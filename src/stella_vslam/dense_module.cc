#include "stella_vslam/type.h"
#include "stella_vslam/dense_module.h"
#include "stella_vslam/camera/base.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/dense_point.h"
#include "stella_vslam/data/map_database.h"
#include "stella_vslam/util/image_converter.h"
#include "stella_vslam/util/keyframe_converter.h"
#include "stella_vslam/util/yaml.h"

#include "patch_match.hpp"

#include <thread>

#include <spdlog/spdlog.h>

namespace stella_vslam {

using patch_match_config = dense::PatchMatch::Config;

dense_module::dense_module(const YAML::Node& yaml_node, camera::base* camera, data::map_database* map_db)
        : map_db_(map_db), camera_(camera) {
    spdlog::debug("CONSTRUCT: dense_module");

    const YAML::Node& local_yaml_node = util::yaml_optional_ref(yaml_node, "PatchMatch");
    if (!local_yaml_node["enabled"].as<bool>(false)) {
        is_paused_ = true;
        spdlog::debug("patch match is disabled");
    }
    else if (camera->model_type_ != camera::model_type_t::Equirectangular) {
        is_paused_ = true;
        spdlog::warn("dense module with patch match is only supported with equirectangular cameras, disabling dense module");
    }
    else {
        spdlog::debug("load patch match parameters");

        std::shared_ptr<patch_match_config> pmd_cfg = std::make_shared<patch_match_config>();
        pmd_cfg->min_patch_standard_deviation = local_yaml_node["min_patch_std_dev"].as<float>(0);
        pmd_cfg->patch_size = local_yaml_node["patch_size"].as<int>(7);
        pmd_cfg->patchmatch_iterations = local_yaml_node["patchmatch_iterations"].as<int>(4);
        pmd_cfg->min_score = local_yaml_node["min_score"].as<double>(0.1);
        pmd_cfg->min_consistent_views = local_yaml_node["min_consistent_views"].as<int>(3);
        pmd_cfg->depthmap_queue_size = local_yaml_node["depthmap_queue_size"].as<size_t>(5);
        pmd_cfg->depthmap_same_depth_threshold = local_yaml_node["depthmap_same_depth_threshold"].as<float>(0.08);
        pmd_cfg->min_views = local_yaml_node["min_views"].as<int>(1);
        pmd_cfg->pointcloud_queue_size = local_yaml_node["pointcloud_queue_size"].as<size_t>(5);
        pmd_cfg->pointcloud_same_depth_threshold = local_yaml_node["pointcloud_same_depth_threshold"].as<float>(0.08);
        pmd_cfg->min_stereo_score = local_yaml_node["min_stereo_score"].as<double>(0);
        patch_match_ = new patch_match_module(pmd_cfg);

        count_init_frames_ = patch_match_->numDropped();

        int cols = local_yaml_node["cols"].as<int>(0);
        int rows = local_yaml_node["rows"].as<int>(0);
        if (cols == 0 && rows == 0) {
            cols = camera->cols_;
            rows = camera->rows_;
        } else if (cols == 0) {
            cols = ((double) rows / (double) camera->rows_) * camera->cols_;
        } else if (rows == 0) {
            rows = ((double) cols / (double) camera->cols_) * camera->rows_;
        }
        depth_res_ = cv::Size(cols, rows);
    }
}

dense_module::~dense_module() {
    spdlog::debug("DESTRUCT: dense_module");

    if (patch_match_) {
        delete patch_match_;
        patch_match_ = nullptr;
    }
}

void dense_module::run() {
    spdlog::info("start dense module");

    is_terminated_ = false;

    while (true) {
        // waiting time for the other threads
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        // check if termination is requested
        if (terminate_is_requested()) {
            // wait for running processing to
            if (patch_match_) {
                while (!patch_match_->isIdle()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(3));
                    retrieve_latest_keyframe();
                }
            }
            // terminate and break
            SPDLOG_TRACE("dense_module: terminate");
            terminate();
            break;
        }

        // check if reset is requested
        if (reset_is_requested()) {
            // reset and continue
            reset();
            continue;
        }

        // idle if patch match is disabled
        if (!patch_match_) {
            continue;
        }

        // check if new dense infromation is available
        retrieve_latest_keyframe();

        // check if pause is requested and not prevented
        if (pause_is_requested()) {
            if (!keyframe_is_queued()) {
                pause();
                SPDLOG_TRACE("dense_module: waiting");
                // check if termination or reset is requested during pause
                while (is_paused() && !terminate_is_requested() && !reset_is_requested()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(3));
                }
                SPDLOG_TRACE("dense_module: resume");
            }
        }

        // if the queue is empty, the following process is not needed
        if (!keyframe_is_queued()) {
            continue;
        }

        // preprocess the new keyframe and input into patch match pipeline
        dispatch_new_keyframe();
    }

    spdlog::info("terminate dense module");
}

std::shared_future<void> dense_module::async_add_keyframe(const std::shared_ptr<data::keyframe>& keyfrm) {
    if (patch_match_) {
        std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
        keyfrms_queue_.push_back(keyfrm);
        promise_add_keyfrm_queue_.emplace_back();
        return promise_add_keyfrm_queue_.back().get_future().share();
    }
    promise_add_keyfrm_queue_.emplace_back();
    promise_add_keyfrm_queue_.back().set_value();
    auto retval = promise_add_keyfrm_queue_.back().get_future().share();
    promise_add_keyfrm_queue_.pop_back();
    return retval;
}

unsigned int dense_module::get_num_queued_keyframes() const {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    return keyfrms_queue_.size();
}

bool dense_module::keyframe_is_queued() const {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    return !keyfrms_queue_.empty();
}

void dense_module::dispatch_new_keyframe() {
    std::shared_ptr<data::keyframe> keyfrm;

    // dequeue
    {
        std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
        // dequeue -> cur_keyfrm_
        keyfrm = keyfrms_queue_.front();
        keyfrms_queue_.pop_front();
    }

    cv::Matx33d R;
    cv::Vec3d t;
    std::vector<cv::Vec3d> sparse;
    cv::Mat img;
    cv::Mat mask;
    util::convert_to_patch_match(*keyfrm, R, t, sparse);
    util::resize(keyfrm->img_, img, depth_res_);
    util::resize(keyfrm->mask_, mask, depth_res_);
    util::convert_to_bgr(img, camera_->color_order_);
    auto retval = patch_match_->addView(R, t, img, mask, sparse);
    if (retval == patch_match_module::SUCCESS) {
        spdlog::trace("patch_match_module::addView: {}", patch_match_module::retval_to_cstr(retval));
        if (count_init_frames_ > 0) {
            --count_init_frames_;
            std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
            promise_add_keyfrm_queue_.front().set_value();
            promise_add_keyfrm_queue_.pop_front();
        }
        else {
            proc_queue_.push_back(keyfrm);
        }
    }
    else {
        spdlog::debug("patch_match_module::addView: {}", patch_match_module::retval_to_cstr(retval));
    }
}

void dense_module::retrieve_latest_keyframe() {
    cv::Mat depthmap;
    std::vector<cv::Vec3d> dense;
    std::vector<cv::Vec3b> color;
    static patch_match_module::ReturnValue retval_last = patch_match_module::SUCCESS;
    patch_match_module::ReturnValue retval = patch_match_->getDepth(depthmap, dense, color, false);
    if (retval == patch_match_module::SUCCESS) {
        spdlog::trace("patch_match_module::getDepth: {}", patch_match_module::retval_to_cstr(retval));
        std::shared_ptr<data::keyframe> keyfrm = proc_queue_.front();
        proc_queue_.pop_front();

        keyfrm->depth_ = depthmap;

        assert(dense.size() == color.size());
        auto dense_iter = dense.begin();
        auto color_iter = color.begin();
        {
            std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);
            for (; dense_iter != dense.end(); ++dense_iter, ++color_iter) {
                const cv::Vec3d &p = *dense_iter;
                const cv::Vec3b &c = *color_iter;
                auto point = std::make_shared<data::dense_point>(map_db_->next_dense_point_id_, Vec3_t(p[0], p[1], p[2]), Color_t(c[0], c[1], c[2]), keyfrm);
                map_db_->add_dense_point(point, map_db_->next_dense_point_id_);
                map_db_->next_dense_point_id_ += 1;
            }
        }
        {
            std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
            promise_add_keyfrm_queue_.front().set_value();
            promise_add_keyfrm_queue_.pop_front();
        }
    }
    else if (retval != retval_last) {
        spdlog::trace("patch_match_module::addView: {}", patch_match_module::retval_to_cstr(retval));
    }
    retval_last = retval;
}

std::shared_future<void> dense_module::async_reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    reset_is_requested_ = true;
    if (!future_reset_.valid()) {
        future_reset_ = promise_reset_.get_future().share();
    }
    return future_reset_;
}

bool dense_module::reset_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    return reset_is_requested_;
}

void dense_module::reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    spdlog::info("reset dense module");
    if (patch_match_) {
        patch_match_->reset();
    }
    keyfrms_queue_.clear();
    {
        std::lock_guard<std::mutex> lock_keyfrm_queue(mtx_keyfrm_queue_);
        while (!promise_add_keyfrm_queue_.empty()) {
            promise_add_keyfrm_queue_.front().set_value();
            promise_add_keyfrm_queue_.pop_front();
        }
    }
    proc_queue_.clear();
    reset_is_requested_ = false;
    promise_reset_.set_value();
    promise_reset_ = std::promise<void>();
    future_reset_ = std::shared_future<void>();
}

std::shared_future<void> dense_module::async_pause() {
    std::lock_guard<std::mutex> lock_pause(mtx_pause_);
    pause_is_requested_ = true;
    if (!future_pause_.valid()) {
        future_pause_ = promise_pause_.get_future().share();
    }

    std::lock_guard<std::mutex> lock_terminate(mtx_terminate_);
    SPDLOG_TRACE("dense_module::async_pause is_terminated_={} is_paused_={}", is_terminated_, is_paused_);
    std::shared_future<void> future_pause = future_pause_;
    if (is_terminated_ || is_paused_) {
        promise_pause_.set_value();
        // Clear request
        promise_pause_ = std::promise<void>();
        future_pause_ = std::shared_future<void>();
    }
    return future_pause;
}

bool dense_module::is_paused() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return is_paused_;
}

bool dense_module::pause_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return pause_is_requested_;
}

void dense_module::pause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    spdlog::info("pause dense module");
    is_paused_ = true;
    promise_pause_.set_value();
    promise_pause_ = std::promise<void>();
    future_pause_ = std::shared_future<void>();
}

void dense_module::resume() {
    // never resume if patch match is disabled
    if (!patch_match_) {
        return;
    }

    std::lock_guard<std::mutex> lock1(mtx_pause_);
    std::lock_guard<std::mutex> lock2(mtx_terminate_);

    // if it has been already terminated, cannot resume
    if (is_terminated_) {
        return;
    }

    assert(keyfrms_queue_.empty());

    is_paused_ = false;
    pause_is_requested_ = false;

    spdlog::info("resume dense module");
}

std::shared_future<void> dense_module::async_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
    if (!future_terminate_.valid()) {
        future_terminate_ = promise_terminate_.get_future().share();
    }
    return future_terminate_;
}

bool dense_module::is_terminated() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return is_terminated_;
}

bool dense_module::terminate_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

void dense_module::terminate() {
    {
        std::lock_guard<std::mutex> lock_pause(mtx_pause_);
        is_paused_ = true;
        promise_pause_.set_value();
        promise_pause_ = std::promise<void>();
        future_pause_ = std::shared_future<void>();
    }
    {
        std::lock_guard<std::mutex> lock_terminate(mtx_terminate_);
        is_terminated_ = true;
        promise_terminate_.set_value();
        promise_terminate_ = std::promise<void>();
        future_terminate_ = std::shared_future<void>();
    }
}

bool dense_module::is_available() const {
    return patch_match_ != nullptr;
}

} // namespace stella_vslam

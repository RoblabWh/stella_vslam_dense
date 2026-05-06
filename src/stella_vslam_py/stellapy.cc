#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "stella_vslam/system.h"
#include "stella_vslam/config.h"
#include "stella_vslam/publish/frame_publisher.h"
#include "stella_vslam/publish/map_publisher.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/data/dense_point.h"

namespace py = pybind11;

template<typename T>
using ndarray = py::array_t<T, py::array::c_style | py::array::forcecast>;
using PyImage = ndarray<uint8_t>;
using PyPoints = ndarray<float>;
using PyPose = std::tuple<ndarray<float>, ndarray<float>>; // position, orientation

namespace stella_vslam {

using logging_callback_t = std::function<void(const std::string&)>;

class python_callback_sink_mt : public spdlog::sinks::base_sink<std::mutex> {
public:
    python_callback_sink_mt(logging_callback_t&& callback, bool color)
        : callback_(std::move(callback)), color_(color) {}

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        fmt::memory_buffer formatted;
        formatter_->format(msg, formatted);

        std::string out;
        std::string_view view(formatted.data(), formatted.size());
        if (!view.empty() && view.back() == '\n') {
            view.remove_suffix(1);
        }
        if (color_ && msg.color_range_end > msg.color_range_start) {
            out = fmt::format("{}{}{}\033[0m{}",
                              view.substr(0, msg.color_range_start),
                              level_color(msg.level),
                              view.substr(msg.color_range_start, msg.color_range_end - msg.color_range_start),
                              view.substr(msg.color_range_end));
        }
        else {
            out.assign(view);
        }

        try {
            callback_(out);
        }
        catch (py::error_already_set& e) {
            std::fprintf(stderr, "[stella_vslam] log callback raised: %s\n", e.what());
        }
    }

    void flush_() override {}

private:
    logging_callback_t callback_;
    bool color_;

    static constexpr std::array<const char*, 7> level_color_ = {
        "\033[37m",        // trace    - white
        "\033[36m",        // debug    - cyan
        "\033[32m",        // info     - green
        "\033[33m\033[1m", // warn     - yellow bold
        "\033[31m\033[1m", // error    - red bold
        "\033[1m\033[41m", // critical - bold on red
        "",                // off
    };

    static std::string_view level_color(spdlog::level::level_enum level) {
        const auto i = static_cast<size_t>(level);
        return i < level_color_.size()
                   ? std::string_view{level_color_[i]}
                   : std::string_view{};
    }
};

class StellaVSLAM {
public:
    StellaVSLAM(const std::string& config_file_path, const std::string& vocab_file_path, const std::string& log_level = "info") {
        system::get_logger()->set_level(spdlog::level::from_str(log_level));
        auto cfg = std::make_shared<config>(config_file_path);

        system_ = std::make_unique<system>(cfg, vocab_file_path);
        frame_publisher_ = system_->get_frame_publisher();
        map_publisher_ = system_->get_map_publisher();
    }

    ~StellaVSLAM() {
        set_log_callback(std::nullopt, true);
    }

    // System API

    void startup(bool need_initialize = true) {
        py::gil_scoped_release release;
        system_->startup(need_initialize);
    }
    void shutdown() {
        py::gil_scoped_release release;
        system_->shutdown();
    }
    void pause() {
        py::gil_scoped_release release;
        system_->pause_tracker();
    }
    bool is_paused() const {
        py::gil_scoped_release release;
        return system_->tracker_is_paused();
    }
    void unpause() {
        py::gil_scoped_release release;
        system_->resume_tracker();
    }
    void reset() {
        py::gil_scoped_release release;
        system_->request_reset();
    }
    bool reset_is_requested() const {
        py::gil_scoped_release release;
        return system_->reset_is_requested();
    }
    void terminate() {
        py::gil_scoped_release release;
        system_->request_terminate();
    }
    bool terminate_is_requested() const {
        py::gil_scoped_release release;
        return system_->terminate_is_requested();
    }

    void enable_mapping() {
        py::gil_scoped_release release;
        system_->enable_mapping_module();
    }
    void disable_mapping() {
        py::gil_scoped_release release;
        system_->disable_mapping_module();
    }
    bool mapping_is_enabled() const {
        py::gil_scoped_release release;
        return system_->mapping_module_is_enabled();
    }
    void enable_temporal_mapping() {
        py::gil_scoped_release release;
        system_->enable_temporal_mapping();
    }
    void enable_dense_reconstruction() {
        py::gil_scoped_release release;
        system_->enable_dense_module();
    }
    void disable_dense_reconstruction() {
        py::gil_scoped_release release;
        system_->disable_dense_module();
    }
    bool dense_reconstruction_is_enabled() const {
        py::gil_scoped_release release;
        return system_->dense_module_is_enabled();
    }
    bool dense_reconstruction_is_available() const {
        py::gil_scoped_release release;
        return system_->dense_module_is_available();
    }
    void enable_loop_detection() {
        py::gil_scoped_release release;
        system_->enable_loop_detector();
    }
    void disable_loop_detection() {
        py::gil_scoped_release release;
        system_->disable_loop_detector();
    }
    bool loop_detection_is_enabled() const {
        py::gil_scoped_release release;
        return system_->loop_detector_is_enabled();
    }
    bool loop_ba_is_running() const {
        py::gil_scoped_release release;
        return system_->loop_BA_is_running();
    }
    void request_loop_closure(int keyfrm1_id, int keyfrm2_id) {
        py::gil_scoped_release release;
        system_->request_loop_closure(keyfrm1_id, keyfrm2_id);
    }

    bool load_map_database(const std::string& path) {
        py::gil_scoped_release release;
        return system_->load_map_database(path);
    }
    bool save_map_database(const std::string& path) {
        py::gil_scoped_release release;
        return system_->save_map_database(path);
    }
    bool save_point_cloud(const std::string& path, std::optional<bool> dense) {
        py::gil_scoped_release release;
        return system_->save_point_cloud(path, dense);
    }
    bool save_keyframes(const std::string& path) {
        py::gil_scoped_release release;
        return system_->save_keyframes(path);
    }
    void save_frame_trajectory(const std::string& path, const std::string& format) {
        py::gil_scoped_release release;
        system_->save_frame_trajectory(path, format);
    }
    void save_keyframe_trajectory(const std::string& path, const std::string& format) {
        py::gil_scoped_release release;
        system_->save_keyframe_trajectory(path, format);
    }

    bool relocalize_by_pose(const PyPose& cam_pose_wc) {
        const auto cam_pose = tuple_to_homogeneous(cam_pose_wc);
        py::gil_scoped_release release;
        return system_->relocalize_by_pose(cam_pose);
    }
    bool relocalize_by_pose_2d(const PyPose& cam_pose_wc, const ndarray<float>& normal_vector) {
        if (normal_vector.size() != 3) {
            throw std::invalid_argument("normal_vector must have exactly 3 elements");
        }
        const auto cam_pose = tuple_to_homogeneous(cam_pose_wc);
        const auto normal_vec = Eigen::Map<const Eigen::Vector3f, Eigen::Unaligned>(normal_vector.data()).cast<double>().eval();
        py::gil_scoped_release release;
        return system_->relocalize_by_pose_2d(cam_pose, normal_vec);
    }

    std::optional<PyPose> feed_monocular_frame(PyImage img, double timestamp, PyImage mask = PyImage()) {
        const auto type = detect_cv_type(img);
        const auto img_ = cv::Mat(img.shape(0), img.shape(1), type, img.mutable_data()).clone();
        return feed_frame([&](const auto& mask_) { return system_->feed_monocular_frame(img_, timestamp, mask_); }, mask);
    }
    std::optional<PyPose> feed_stereo_frame(PyImage left_img, PyImage right_img, double timestamp, PyImage mask = PyImage()) {
        const auto left_type = detect_cv_type(left_img);
        const auto right_type = detect_cv_type(right_img);
        const auto left_img_ = cv::Mat(left_img.shape(0), left_img.shape(1), left_type, left_img.mutable_data()).clone();
        const auto right_img_ = cv::Mat(right_img.shape(0), right_img.shape(1), right_type, right_img.mutable_data()).clone();
        return feed_frame([&](const auto& mask_) { return system_->feed_stereo_frame(left_img_, right_img_, timestamp, mask_); }, mask);
    }
    std::optional<PyPose> feed_rgbd_frame(PyImage rgb_img, ndarray<float> depthmap, double timestamp, PyImage mask = PyImage()) {
        const auto rgb_type = detect_cv_type(rgb_img);
        const auto depthmap_type = detect_cv_type(depthmap);
        const auto rgb_img_ = cv::Mat(rgb_img.shape(0), rgb_img.shape(1), rgb_type, rgb_img.mutable_data()).clone();
        const auto depthmap_ = cv::Mat(depthmap.shape(0), depthmap.shape(1), depthmap_type, depthmap.mutable_data()).clone();
        return feed_frame([&](const auto& mask_) { return system_->feed_RGBD_frame(rgb_img_, depthmap_, timestamp, mask_); }, mask);
    }

    // Frame Publisher API

    PyImage draw_frame() {
        cv::Mat drawn_img;
        {
            py::gil_scoped_release release;
            drawn_img = frame_publisher_->draw_frame();
        }
        auto np_img = PyImage({drawn_img.rows, drawn_img.cols, drawn_img.channels()});
        auto cv_img = cv::Mat(drawn_img.size(), drawn_img.type(), np_img.mutable_data());
        cv::cvtColor(drawn_img, cv_img, cv::COLOR_BGR2RGB);
        return np_img;
    }

    // Map Publisher API

    std::tuple<PyPoints, PyPoints> get_all_landmarks() {
        std::vector<Eigen::Vector3f> all_points;
        std::vector<Eigen::Vector3f> local_points;
        {
            py::gil_scoped_release release;

            std::vector<std::shared_ptr<data::landmark>> all_landmarks;
            std::set<std::shared_ptr<data::landmark>> local_landmarks;
            map_publisher_->get_landmarks(all_landmarks, local_landmarks);

            all_points = landmarks_to_points(all_landmarks);
            local_points = landmarks_to_points(local_landmarks);
        }

        PyPoints np_all_points({all_points.size(), 3UL});
        std::memcpy(np_all_points.mutable_data(), all_points.data(), np_all_points.nbytes());

        PyPoints np_local_points({local_points.size(), 3UL});
        std::memcpy(np_local_points.mutable_data(), local_points.data(), np_local_points.nbytes());

        return {np_all_points, np_local_points};
    }

    std::tuple<PyPoints, PyImage> get_dense_points() {
        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3<uint8_t>> colors;
        {
            py::gil_scoped_release release;

            std::vector<std::shared_ptr<data::dense_point>> dense_points;
            map_publisher_->get_dense_points(dense_points);

            points.reserve(dense_points.size());
            colors.reserve(dense_points.size());

            for (const auto& pt : dense_points) {
                if (pt) {
                    points.emplace_back(pt->get_pos_in_world().cast<float>());
                    colors.emplace_back(pt->get_color_in_rgb());
                }
            }
        }

        PyPoints np_points({points.size(), 3UL});
        std::memcpy(np_points.mutable_data(), points.data(), np_points.nbytes());

        PyImage np_color({colors.size(), 3UL});
        std::memcpy(np_color.mutable_data(), colors.data(), np_color.nbytes());

        return {np_points, np_color};
    }

    std::tuple<std::unordered_map<uint32_t, PyPose>, ndarray<float>, ndarray<float>, ndarray<float>> get_keyframe_graph(const uint32_t min_shared_lms) {
        std::vector<std::pair<uint32_t, Eigen::Matrix4d>> keyframe_pose;
        std::vector<Eigen::Vector3f> spanning_tree_edges;
        std::vector<Eigen::Vector3f> loop_edges;
        std::vector<Eigen::Vector3f> covisibility_edges;
        {
            py::gil_scoped_release release;

            std::vector<std::shared_ptr<data::keyframe>> all_keyframes;
            map_publisher_->get_keyframes(all_keyframes);

            // Reserve enough memory that realloc is mostly unnecessary
            keyframe_pose.reserve(all_keyframes.size());
            spanning_tree_edges.reserve(all_keyframes.size() * 2);
            covisibility_edges.reserve(all_keyframes.size() * 32);

            for (const auto& kf : all_keyframes) {
                if (!kf || kf->will_be_erased()) {
                    continue;
                }

                // get keyframe id
                const auto kf_id = kf->id_;

                // get keyframe pose
                keyframe_pose.emplace_back(kf_id, kf->get_pose_wc());
                const auto kf_pos = kf->get_trans_wc().cast<float>().eval();

                // get spanning tree edges
                const auto spanning_parent = kf->graph_node_->get_spanning_parent();
                if (spanning_parent) {
                    spanning_tree_edges.emplace_back(kf_pos);
                    spanning_tree_edges.emplace_back(spanning_parent->get_trans_wc().cast<float>());
                }

                // get loop edges
                const auto kf_loop_edges = kf->graph_node_->get_loop_edges();
                for (const auto& loop_edge : kf_loop_edges) {
                    if (loop_edge && (loop_edge->id_ >= kf_id)) {
                        loop_edges.emplace_back(kf_pos);
                        loop_edges.emplace_back(loop_edge->get_trans_wc().cast<float>());
                    }
                }

                // get covisibility edges
                const auto kf_covisibility_edges = kf->graph_node_->get_covisibilities_over_min_num_shared_lms(min_shared_lms);
                for (const auto& covisibility_edge : kf_covisibility_edges) {
                    if (covisibility_edge && !covisibility_edge->will_be_erased() && (covisibility_edge->id_ >= kf_id)) {
                        covisibility_edges.emplace_back(kf_pos);
                        covisibility_edges.emplace_back(covisibility_edge->get_trans_wc().cast<float>());
                    }
                }
            }
        }

        std::unordered_map<uint32_t, PyPose> np_keyframe_pose;
        for (const auto& [id, pose] : keyframe_pose) {
            np_keyframe_pose[id] = homogeneous_to_tuple(pose);
        }

        ndarray<float> np_spanning_tree_edges({spanning_tree_edges.size() / 2, 2UL, 3UL});
        std::memcpy(np_spanning_tree_edges.mutable_data(), spanning_tree_edges.data(), np_spanning_tree_edges.nbytes());

        ndarray<float> np_loop_edges({loop_edges.size() / 2, 2UL, 3UL});
        std::memcpy(np_loop_edges.mutable_data(), loop_edges.data(), np_loop_edges.nbytes());

        ndarray<float> np_covisibility_edges({covisibility_edges.size() / 2, 2UL, 3UL});
        std::memcpy(np_covisibility_edges.mutable_data(), covisibility_edges.data(), np_covisibility_edges.nbytes());

        return {np_keyframe_pose, np_spanning_tree_edges, np_loop_edges, np_covisibility_edges};
    }

    static void set_log_callback(std::optional<logging_callback_t> callback, bool color) {
        auto sink = callback
                                    ? spdlog::sink_ptr{std::make_shared<python_callback_sink_mt>(std::move(*callback), color)}
                                : color ? spdlog::sink_ptr{std::make_shared<spdlog::sinks::stdout_color_sink_mt>()}
                                        : spdlog::sink_ptr{std::make_shared<spdlog::sinks::stdout_sink_mt>()};
        system::get_logger()->sinks() = {std::move(sink)};
    }

private:
    std::unique_ptr<system> system_;
    std::shared_ptr<publish::frame_publisher> frame_publisher_;
    std::shared_ptr<publish::map_publisher> map_publisher_;

    template<typename T>
    static inline int detect_cv_type(const ndarray<T>& img) {
        constexpr int base_type = []() {
            if constexpr (std::is_same_v<T, uint8_t>) return CV_8U;
            else if constexpr (std::is_same_v<T, float>) return CV_32F;
            else static_assert(!std::is_same_v<T, T>(), "detect_cv_type: unsupported element type");
        }();

        const auto np_dims = img.ndim();
        if (np_dims < 2 || np_dims > 3) {
            throw std::invalid_argument("image array must be 2D (H, W) or 3D (H, W, C > 1)");
        }
        return CV_MAKETYPE(base_type, np_dims == 2 ? 1 : img.shape(2));
    }

    template<typename F>
    static inline std::optional<PyPose> feed_frame(F&& feed_method, PyImage mask) {
        auto mask_ = cv::Mat();
        if (mask.size() > 0) {
            const auto type = detect_cv_type(mask);
            mask_ = cv::Mat(mask.shape(0), mask.shape(1), type, mask.mutable_data()).clone();
        }

        std::shared_ptr<Eigen::Matrix4d> pose;
        {
            py::gil_scoped_release release;
            pose = feed_method(mask_);
        }
        return pose ? std::optional<PyPose>{homogeneous_to_tuple(*pose)} : std::nullopt;
    }

    // Conversion utilities

    template<typename C>
    static std::vector<Eigen::Vector3f> landmarks_to_points(const C& landmarks) {
        std::vector<Eigen::Vector3f> points;
        points.reserve(landmarks.size());

        for (const auto& lm : landmarks) {
            if (lm) {
                points.emplace_back(lm->get_pos_in_world().template cast<float>());
            }
        }
        return points;
    }

    static inline PyPose homogeneous_to_tuple(const Eigen::Matrix4d& pose) {
        ndarray<float> np_position(3);
        const auto& position = pose.topRightCorner<3, 1>();
        Eigen::Map<Eigen::Vector3f, Eigen::Unaligned>(np_position.mutable_data()) = position.cast<float>();

        ndarray<float> np_orientation(4);
        const auto& orientation = Eigen::Quaterniond(pose.topLeftCorner<3, 3>());
        Eigen::Map<Eigen::Vector4f, Eigen::Unaligned>(np_orientation.mutable_data()) = orientation.coeffs().cast<float>();

        return {np_position, np_orientation};
    }

    static inline Eigen::Matrix4d tuple_to_homogeneous(const PyPose& pose) {
        const auto& position = std::get<0>(pose);
        const auto& orientation = std::get<1>(pose);
        if (position.size() != 3) {
            throw std::invalid_argument("pose position must have exactly 3 elements");
        }
        if (orientation.size() != 4) {
            throw std::invalid_argument("pose orientation must have exactly 4 elements");
        }

        auto eigen_pose = Eigen::Matrix4d::Identity().eval();
        eigen_pose.topRightCorner<3, 1>() = Eigen::Map<const Eigen::Vector3f, Eigen::Unaligned>(position.data()).cast<double>();

        auto quat = Eigen::Quaterniond(Eigen::Map<const Eigen::Vector4f, Eigen::Unaligned>(orientation.data()).cast<double>());
        eigen_pose.topLeftCorner<3, 3>() = quat.toRotationMatrix();

        return eigen_pose;
    }
};
} // namespace stella_vslam

PYBIND11_MODULE(stellapy, m) {
    m.doc() = "stella_vslam python bindings";

    py::class_<stella_vslam::StellaVSLAM>(m, "StellaVSLAM")
        .def(py::init<const std::string&, const std::string&, const std::string&>(),
             py::arg("config_file_path"),
             py::arg("vocab_file_path"),
             py::arg("log_level") = "info")

        .def("startup", &stella_vslam::StellaVSLAM::startup, py::arg("need_initialize") = true)
        .def("shutdown", &stella_vslam::StellaVSLAM::shutdown)
        .def("pause", &stella_vslam::StellaVSLAM::pause)
        .def("is_paused", &stella_vslam::StellaVSLAM::is_paused)
        .def("unpause", &stella_vslam::StellaVSLAM::unpause)
        .def("reset", &stella_vslam::StellaVSLAM::reset)
        .def("reset_is_requested", &stella_vslam::StellaVSLAM::reset_is_requested)
        .def("terminate", &stella_vslam::StellaVSLAM::terminate)
        .def("terminate_is_requested", &stella_vslam::StellaVSLAM::terminate_is_requested)

        .def("enable_mapping", &stella_vslam::StellaVSLAM::enable_mapping)
        .def("disable_mapping", &stella_vslam::StellaVSLAM::disable_mapping)
        .def("mapping_is_enabled", &stella_vslam::StellaVSLAM::mapping_is_enabled)
        .def("enable_temporal_mapping", &stella_vslam::StellaVSLAM::enable_temporal_mapping)
        .def("enable_dense_reconstruction", &stella_vslam::StellaVSLAM::enable_dense_reconstruction)
        .def("disable_dense_reconstruction", &stella_vslam::StellaVSLAM::disable_dense_reconstruction)
        .def("dense_reconstruction_is_enabled", &stella_vslam::StellaVSLAM::dense_reconstruction_is_enabled)
        .def("dense_reconstruction_is_available", &stella_vslam::StellaVSLAM::dense_reconstruction_is_available)
        .def("enable_loop_detection", &stella_vslam::StellaVSLAM::enable_loop_detection)
        .def("disable_loop_detection", &stella_vslam::StellaVSLAM::disable_loop_detection)
        .def("loop_detection_is_enabled", &stella_vslam::StellaVSLAM::loop_detection_is_enabled)
        .def("loop_ba_is_running", &stella_vslam::StellaVSLAM::loop_ba_is_running)
        .def("request_loop_closure", &stella_vslam::StellaVSLAM::request_loop_closure,
             py::arg("keyfrm1_id"), py::arg("keyfrm2_id"))

        .def("load_map_database", &stella_vslam::StellaVSLAM::load_map_database, py::arg("path"))
        .def("save_map_database", &stella_vslam::StellaVSLAM::save_map_database, py::arg("path"))
        .def("save_point_cloud", &stella_vslam::StellaVSLAM::save_point_cloud, py::arg("path"), py::arg("dense") = std::nullopt)
        .def("save_keyframes", &stella_vslam::StellaVSLAM::save_keyframes, py::arg("path"))
        .def("save_frame_trajectory", &stella_vslam::StellaVSLAM::save_frame_trajectory,
             py::arg("path"), py::arg("format"))
        .def("save_keyframe_trajectory", &stella_vslam::StellaVSLAM::save_keyframe_trajectory,
             py::arg("path"), py::arg("format"))

        .def("relocalize_by_pose", &stella_vslam::StellaVSLAM::relocalize_by_pose, py::arg("cam_pose_wc"))
        .def("relocalize_by_pose_2d", &stella_vslam::StellaVSLAM::relocalize_by_pose_2d, py::arg("cam_pose_wc"), py::arg("normal_vector"))
        .def("feed_monocular_frame", &stella_vslam::StellaVSLAM::feed_monocular_frame,
             py::arg("img"), py::arg("timestamp"), py::arg("mask") = PyImage())
        .def("feed_stereo_frame", &stella_vslam::StellaVSLAM::feed_stereo_frame,
             py::arg("left_img"), py::arg("right_img"), py::arg("timestamp"), py::arg("mask") = PyImage())
        .def("feed_rgbd_frame", &stella_vslam::StellaVSLAM::feed_rgbd_frame,
             py::arg("rgb_img"), py::arg("depthmap"), py::arg("timestamp"), py::arg("mask") = PyImage())

        .def("draw_frame", &stella_vslam::StellaVSLAM::draw_frame)
        .def("get_all_landmarks", &stella_vslam::StellaVSLAM::get_all_landmarks)
        .def("get_dense_points", &stella_vslam::StellaVSLAM::get_dense_points)
        .def("get_keyframe_graph", &stella_vslam::StellaVSLAM::get_keyframe_graph, py::arg("min_shared_lms") = 100)

        .def_static("set_log_callback", &stella_vslam::StellaVSLAM::set_log_callback, py::arg("callback"), py::arg("color") = true);
}

#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/map_database.h"
#include "stella_vslam/io/keyframe_io_opencv.h"
#include "stella_vslam/util/image_converter.h"

#include <spdlog/spdlog.h>

#include <opencv2/imgcodecs.hpp>

#include "filesystem"

namespace stella_vslam {
namespace io {

bool keyframe_io_opencv::save(const std::string& path,
                           const data::map_database* const map_db) {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    assert(map_db);

    auto keyfrms = map_db->get_all_keyframes();

    std::error_code ec;
    std::filesystem::path path_checked(path);
    if (!std::filesystem::create_directory(path_checked, ec) && ec)
    {
        return false;
    }

    for (auto keyfrm : keyfrms)
    {
        if (!keyfrm->img_.empty())
        {
            cv::imwrite(path_checked / (std::string("image") + std::to_string(keyfrm->id_) + '.' + format_), keyfrm->img_);
        }
        if (!keyfrm->depth_.empty())
        {
            cv::Mat depthvis;
            cv::Mat mask;
            util::resize(keyfrm->mask_, mask, keyfrm->depth_.rows, keyfrm->depth_.cols);
            cv::normalize(keyfrm->depth_, depthvis, 255, 0, cv::NORM_MINMAX, CV_8UC1, mask);
            cv::imwrite(path_checked / (std::string("depthvis") + std::to_string(keyfrm->id_) + '.' + format_), depthvis);
            cv::imwrite(path_checked / (std::string("depth") + std::to_string(keyfrm->id_) + std::string(".tiff")), keyfrm->depth_);
        }
        if (!keyfrm->mask_.empty())
        {
            cv::imwrite(path_checked / (std::string("mask") + std::to_string(keyfrm->id_) + '.' + format_), keyfrm->mask_);
        }
    }

    return true;
}

} // namespace io
} // namespace stella_vslam

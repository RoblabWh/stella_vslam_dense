#ifndef STELLA_VSLAM_IO_KEYFRAME_IO_FACTORY_H
#define STELLA_VSLAM_IO_KEYFRAME_IO_FACTORY_H

#include "stella_vslam/io/keyframe_io_base.h"
#include "stella_vslam/io/keyframe_io_opencv.h"

#include <string>

namespace stella_vslam {

namespace data {
class map_database;
} // namespace data

namespace io {

class keyframe_io_factory {
public:
    static std::shared_ptr<keyframe_io_base> create(const std::string& keyframe_format) {
        std::shared_ptr<keyframe_io_base> keyframe_io;
        if (keyframe_format == "png" || keyframe_format == "jpg" || keyframe_format == "jpeg" || keyframe_format == "tiff") {
            keyframe_io = std::make_shared<io::keyframe_io_opencv>(keyframe_format);
        }
        else {
            throw std::runtime_error("Invalid keyframe format: " + keyframe_format);
        }
        return keyframe_io;
    }
};

} // namespace io
} // namespace stella_vslam

#endif // STELLA_VSLAM_IO_KEYFRAME_IO_FACTORY_H

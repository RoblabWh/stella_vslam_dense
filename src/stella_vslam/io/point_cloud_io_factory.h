#ifndef STELLA_VSLAM_IO_POINT_CLOUD_IO_FACTORY_H
#define STELLA_VSLAM_IO_POINT_CLOUD_IO_FACTORY_H

#include "stella_vslam/io/point_cloud_io_base.h"
#include "stella_vslam/io/point_cloud_io_ply_ascii.h"

#include <string>

namespace stella_vslam {

namespace data {
class map_database;
} // namespace data

namespace io {

class point_cloud_io_factory {
public:
    static std::shared_ptr<point_cloud_io_base> create(const std::string& point_cloud_format) {
        std::shared_ptr<point_cloud_io_base> point_cloud_io;
        if (point_cloud_format == "ply_ascii") {
            point_cloud_io = std::make_shared<io::point_cloud_io_ply_ascii>();
        }
        else {
            throw std::runtime_error("Invalid point cloud format: " + point_cloud_format);
        }
        return point_cloud_io;
    }
};

} // namespace io
} // namespace stella_vslam

#endif // STELLA_VSLAM_IO_POINT_CLOUD_IO_FACTORY_H

#ifndef STELLA_VSLAM_IO_POINT_CLOUD_IO_PLY_ASCII_H
#define STELLA_VSLAM_IO_POINT_CLOUD_IO_PLY_ASCII_H

#include "stella_vslam/io/point_cloud_io_base.h"

#include <string>

namespace stella_vslam {

namespace data {
class map_database;
} // namespace data

namespace io {

class point_cloud_io_ply_ascii : public point_cloud_io_base {
public:
    /**
     * Constructor
     */
    point_cloud_io_ply_ascii() = default;

    /**
     * Destructor
     */
    virtual ~point_cloud_io_ply_ascii() = default;

    /**
     * Save the point cloud as PLY
     */
    bool save(const std::string& path,
              const data::map_database* const map_db) override;
};

} // namespace io
} // namespace stella_vslam

#endif // STELLA_VSLAM_IO_POINT_CLOUD_IO_PLY_ASCII_H

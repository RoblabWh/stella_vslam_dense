#ifndef STELLA_VSLAM_IO_POINT_CLOUD_IO_BASE_H
#define STELLA_VSLAM_IO_POINT_CLOUD_IO_BASE_H

#include <string>

namespace stella_vslam {

namespace data {
class map_database;
} // namespace data

namespace io {

class point_cloud_io_base {
public:
    /**
     * Save the point cloud
     */
    virtual bool save(const std::string& path,
                      const data::map_database* const map_db,
                      bool dense)
        = 0;
};

} // namespace io
} // namespace stella_vslam

#endif // STELLA_VSLAM_IO_POINT_CLOUD_IO_BASE_H

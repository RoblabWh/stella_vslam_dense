#include "stella_vslam/data/dense_point.h"
#include "stella_vslam/data/map_database.h"
#include "stella_vslam/io/point_cloud_io_ply_ascii.h"

#include <spdlog/spdlog.h>

#include <fstream>

namespace stella_vslam {
namespace io {

bool point_cloud_io_ply_ascii::save(const std::string& path,
                                   const data::map_database* const map_db) {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    assert(map_db);

    std::ofstream ofs(path, std::ios::out);

    if (ofs.is_open()) {
        spdlog::info("save the PLY file of point cloud to {}", path);

        ofs << "ply\n"
               "format ascii 1.0\n"
               "element vertex " << map_db->get_num_dense_points() << '\n';
        ofs << "property float x\n"
               "property float y\n"
               "property float z\n"
               "property uchar red\n"
               "property uchar green\n"
               "property uchar blue\n"
               "end_header\n"
            << std::fixed;

        std::vector<std::shared_ptr<data::dense_point>> points;
        points = map_db->get_all_dense_points();
        for (const auto& point : points) {
            const Vec3_t &pos_w = point->get_pos_in_world();
            const Color_t &color = point->get_color_in_rgb();

            ofs << pos_w[0] << ' ' << pos_w[1] << ' ' << pos_w[2] << ' '
                << (uint16_t) color[0] << ' ' << (uint16_t) color[1] << ' ' << (uint16_t) color[2] << '\n';
        }
        ofs.flush();
        ofs.close();
        return true;
    }
    else {
        spdlog::critical("cannot create a file at {}", path);
        return false;
    }
}

} // namespace io
} // namespace stella_vslam

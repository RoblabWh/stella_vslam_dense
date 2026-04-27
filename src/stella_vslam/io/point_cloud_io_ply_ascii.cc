#include "stella_vslam/data/landmark.h"
#include "stella_vslam/data/dense_point.h"
#include "stella_vslam/data/map_database.h"
#include "stella_vslam/io/point_cloud_io_ply_ascii.h"

#include <spdlog/spdlog.h>

#include <fstream>

namespace stella_vslam {
namespace io {

template<bool D, typename C>
void write_points_to_file(const C& points, std::ofstream& ofs) {
    for (const auto& point : points) {
        assert(point);

        const Vec3_t& pos_w = point->get_pos_in_world();
        ofs << pos_w[0] << ' ' << pos_w[1] << ' ' << pos_w[2];

        if constexpr (D) {
            const Color_t& color = point->get_color_in_rgb();
            ofs << ' ' << (uint16_t)color[0] << ' ' << (uint16_t)color[1] << ' ' << (uint16_t)color[2];
        }
        ofs << '\n';
    }
}

bool point_cloud_io_ply_ascii::save(const std::string& path,
                                    const data::map_database* const map_db,
                                    bool dense) {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    assert(map_db);

    std::ofstream ofs(path, std::ios::out);

    if (ofs.is_open()) {
        spdlog::info("save {} point cloud as PLY to {}", dense ? "dense" : "sparse", path);

        ofs << "ply\n"
               "format ascii 1.0\n"
               "element vertex " << (dense ? map_db->get_num_dense_points() : map_db->get_num_landmarks()) << '\n';
        ofs << "property float x\n"
               "property float y\n"
               "property float z\n";
        if (dense) {
            ofs << "property uchar red\n"
                   "property uchar green\n"
                   "property uchar blue\n";
        }
        ofs << "end_header\n"
            << std::fixed;

        if (dense) {
            std::vector<std::shared_ptr<data::dense_point>> points;
            points = map_db->get_all_dense_points();
            write_points_to_file<true>(points, ofs);
        }
        else {
            std::vector<std::shared_ptr<data::landmark>> landmarks;
            landmarks = map_db->get_all_landmarks();
            write_points_to_file<false>(landmarks, ofs);
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

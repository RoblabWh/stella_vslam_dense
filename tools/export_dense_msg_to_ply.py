#!/usr/bin/env python3

import msgpack
import argparse
import os

def point_cloud_to_ply(output, points, colors, scale=1):
    """Export depthmap points as a PLY string"""
    with open(output, "w", encoding="utf-8") as fp:
        lines = _point_cloud_to_ply_lines(points, colors, scale)
        fp.writelines(lines)


def _point_cloud_to_ply_lines(points,colors, scale):
    yield "ply\n"
    yield "format ascii 1.0\n"
    yield "element vertex {}\n".format(len(points))
    yield "property float x\n"
    yield "property float y\n"
    yield "property float z\n"
    yield "property uchar red\n"
    yield "property uchar green\n"
    yield "property uchar blue\n"
    yield "end_header\n"

    template = "{:.4f} {:.4f} {:.4f} {} {} {}\n"
    for i in range(len(points)):
        p, c = points[i], colors[i]
        #p = points[i]
        yield template.format(
            scale*p[0],
            scale*p[1],
            scale*p[2],
            int(c[0]),
            int(c[1]),
            int(c[2])
        )

def read_dense_pcl_msg(path_to_msg):

    msg = None
    with open(path_to_msg, 'rb') as f:
        mapstr = f.read()
        msg = msgpack.unpackb(mapstr, use_list=True, raw=False)
    return msg

def main():
    arg_parser = argparse.ArgumentParser(
        description='Export msg from OpenVSLAM/PatchMatchDenseVSLAM to ply')
    arg_parser.add_argument('-i', '--in_path_MsgMap', type=str, help='Path to *.msg file (OpenVSLAM/PatchMatchDenseVSLAM)', required=True)
    arg_parser.add_argument('-o', '--out_path_plyMap', type=str, help='Path to output *.ply file', required=False)
    arg_parser.add_argument('--nocolor', action='store_true', help='Output *.ply file without color')

    args = arg_parser.parse_args()

    has_color = False
    path_to_msg = args.in_path_MsgMap
    if os.path.exists(path_to_msg + "_dense"):
        has_color = True
        path_to_msg = path_to_msg + "_dense"
    elif os.path.exists(path_to_msg):
        if path_to_msg.endswith("_dense"):
            has_color = True
    else:
        raise FileNotFoundError
    has_color = True

    path_to_output = None
    if args.out_path_plyMap:
        if not os.path.exists(os.path.split(args.out_path_plyMap)[0]):
            raise NotADirectoryError

        path_to_output = args.out_path_plyMap
    else:
        path_to_output = path_to_msg + ".ply"

    msg = read_dense_pcl_msg(path_to_msg)

    #keyfrms = msg["keyframes"]
    lms = msg["dense_points"]

    points = []
    colors = []
    for lm in lms:
        point = lms[lm]['pos_w']
        if has_color and not args.nocolor:
            color = lms[lm]['color']
        else:
            color = (255,255,255)

        points.append(point)
        colors.append(color)

    point_cloud_to_ply(path_to_output, points, colors)


if __name__=='__main__':
    main()

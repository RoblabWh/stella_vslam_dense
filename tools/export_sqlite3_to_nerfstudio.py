#!/usr/bin/env python3

import os
import math
import argparse
import json
import sqlite3
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R
from numba import jit


def point_cloud_to_ply(output, points, colors, scale=1):
    with open(output, "w", encoding="utf-8") as fp:
        lines = _point_cloud_to_ply_lines(points, colors, scale)
        fp.writelines(lines)

def _point_cloud_to_ply_lines(points, colors, scale):
    yield "ply\n"
    yield "format ascii 1.0\n"
    yield "element vertex {}\n".format(len(points))
    yield "property float x\n"
    yield "property float y\n"
    yield "property float z\n"
    if colors:
        yield "property uchar red\n"
        yield "property uchar green\n"
        yield "property uchar blue\n"
    yield "end_header\n"

    if colors:
        template = "{:.4f} {:.4f} {:.4f} {} {} {}\n"
        for i in range(len(points)):
            p, c = points[i], colors[i]
            yield template.format(
                scale*p[0],
                scale*p[1],
                scale*p[2],
                int(c[2]),
                int(c[1]),
                int(c[0])
            )
    else:
        template = "{:.4f} {:.4f} {:.4f}\n"
        for p in points:
            yield template.format(
                scale*p[0],
                scale*p[1],
                scale*p[2]
            )

def tfCamToWorld(pos):
    pos_w = [pos[2], -pos[0], -pos[1]]
    # adjust world to whatever is happening in other transform
    return [-pos_w[1], pos_w[0], pos_w[2]]

@jit(nopython=True)
def clip(value, min, max):
    if value > min and value < max:
        return value
    elif value < min:
        return min
    else:
        return max

# get x,y,z coords from out image pixels coords
# i,j are pixel coords
# face is face number
# edge is edge length
@jit(nopython=True)
def outImgToXYZ(i,j,face,edge):
    a = 2.0*float(i)/edge
    b = 2.0*float(j)/edge
    if face==0: # back
        (x,y,z) = (-1.0, 1.0-a, 3.0 - b)
    elif face==1: # left
        (x,y,z) = (a-3.0, -1.0, 3.0 - b)
    elif face==2: # front
        (x,y,z) = (1.0, a - 5.0, 3.0 - b)
    elif face==3: # right
        (x,y,z) = (7.0-a, 1.0, 3.0 - b)
    elif face==4: # top
        (x,y,z) = (b-1.0, a -5.0, 1.0)
    elif face==5: # bottom
        (x,y,z) = (5.0-b, a-5.0, -1.0)
    return (x,y,z)

# convert using an inverse transformation
@jit(nopython=True)
def convertBack(inSize, outSize, inPix, outPix):

    edge = int(inSize[1]/4)   # the length of each edge in pixels
    for i in range(outSize[1]):
        face = int(i/edge) # 0 - back, 1 - left 2 - front, 3 - right
        if face==2:
            rng = range(0,edge*3)
        else:
            rng = range(edge,edge*2)

        for j in rng:
            if j<edge:
                face2 = 4 # top
            elif j>=2*edge:
                face2 = 5 # bottom
            else:
                face2 = face

            (x,y,z) = outImgToXYZ(i,j,face2,edge)
            theta = math.atan2(y,x) # range -pi to pi
            r = math.hypot(x,y)
            phi = math.atan2(z,r) # range -pi/2 to pi/2
            # source img coords
            uf = ( 2.0*edge*(theta + math.pi)/math.pi )
            vf = ( 2.0*edge * (math.pi/2 - phi)/math.pi)
            # Use bilinear interpolation between the four surrounding pixels
            ui = math.floor(uf)  # coord of pixel to bottom left
            vi = math.floor(vf)
            u2 = ui+1       # coords of pixel to top right
            v2 = vi+1
            mu = uf-ui      # fraction of way across pixel
            nu = vf-vi
            # Pixel values of four corners
            A = inPix[int(clip(vi,0,inSize[0]-1)),ui % inSize[1]]
            B = inPix[int(clip(vi,0,inSize[0]-1)),u2 % inSize[1]]
            C = inPix[int(clip(v2,0,inSize[0]-1)),ui % inSize[1]]
            D = inPix[int(clip(v2,0,inSize[0]-1)),u2 % inSize[1]]
            # interpolate
            (r,g,b) = (
              A[0]*(1-mu)*(1-nu) + B[0]*(mu)*(1-nu) + C[0]*(1-mu)*nu+D[0]*mu*nu,
              A[1]*(1-mu)*(1-nu) + B[1]*(mu)*(1-nu) + C[1]*(1-mu)*nu+D[1]*mu*nu,
              A[2]*(1-mu)*(1-nu) + B[2]*(mu)*(1-nu) + C[2]*(1-mu)*nu+D[2]*mu*nu )

            outPix[j,i] = (int(round(r)),int(round(g)),int(round(b)))



def main():
    parser = argparse.ArgumentParser()

    # Path to sqlite3 database
    parser.add_argument('sqlite3_db', type=str, help='Path to sqlite3 database')
    # Path to output folder
    parser.add_argument('output_folder', type=str, help='Path to output folder')
    # Cube configuration
    cube_faces = ['front', 'back', 'left', 'right', 'top', 'bottom']
    parser.add_argument('--faces', type=str, nargs='+', help='Cube faces to be exported', choices=cube_faces, default=cube_faces)

    args = parser.parse_args()

    db_connection = sqlite3.connect(args.sqlite3_db)
    db_cursor = db_connection.cursor()
    keyframes = db_cursor.execute("SELECT * FROM keyframes").fetchall()
    landmarks = db_cursor.execute("SELECT pos_w FROM landmarks").fetchall()
    landmarks = [tfCamToWorld(np.frombuffer(i[0], np.float64)) for i in landmarks]
    dense_points_pos = db_cursor.execute("SELECT pos_w FROM dense_points").fetchall()
    dense_points_pos = [tfCamToWorld(np.frombuffer(i[0], np.float64)) for i in dense_points_pos]
    dense_points_color = db_cursor.execute("SELECT color FROM dense_points").fetchall()
    dense_points_color = [np.frombuffer(i[0], np.uint8) for i in dense_points_color]

    frame_count = len(keyframes)
    frame_nr = 1
    frame_infos = []
    edge = 0

    image_path = os.path.join(args.output_folder, 'images')
    os.makedirs(image_path, exist_ok=True)

    for keyframe in keyframes:
        print("Extracting frame: " + str(frame_nr) + '/' + str(frame_count))

        pose_cw = np.frombuffer(keyframe[4], np.float64).reshape((4, 4)).transpose()
        r = R.from_matrix(np.array(pose_cw[:3, :3]))
        t = pose_cw[:3, 3]
        t = np.matrix([t])
        t = t.transpose()
        r_ = r.as_matrix().T
        t = - r_ * t
        t = np.asarray([[t[0, 0]], [t[1, 0]], [t[2, 0]]])

        r = r.as_euler('xyz')
        r[1] *= -1
        r = R.from_euler('xyz', r)

        frame = cv.imdecode(np.frombuffer(keyframe[10], np.uint8), cv.IMREAD_UNCHANGED)

        inSize = frame.shape
        inPix = frame
        outPix = np.zeros((int(inSize[1] * 3 / 4), inSize[1], 3), np.uint8)
        outSize = outPix.shape

        convertBack(inSize, outSize, inPix, outPix)

        edge = int(inSize[1] / 4)

        # top
        if 'top' in args.faces:
            r_r = r * R.from_euler('Z', 180, degrees=True)
            r_r = r_r * R.from_euler('Y', 180, degrees=True)
            r_r = r_r * R.from_euler('X', 90, degrees=True)
            mat = np.append(r_r.as_matrix(), t, axis=1)
            mat = np.append(mat, [[0, 0, 0, 1]], axis=0)

            T = np.eye(4)
            T[:3, :3] = R.from_euler('X', -90, degrees=True).as_matrix()
            mat = np.matmul(T, mat)

            cv.imwrite(os.path.join(image_path, str(frame_nr) + "_top.png"), outPix[0:edge, edge * 2:edge * 3])

            frame_infos.append((frame_nr, "top", mat))

        # bottom
        if 'bottom' in args.faces:
            r_r = r * R.from_euler('Z', 180, degrees=True)
            r_r = r_r * R.from_euler('Y', 180, degrees=True)
            r_r = r_r * R.from_euler('X', -90, degrees=True)
            mat = np.append(r_r.as_matrix(), t, axis=1)
            mat = np.append(mat, [[0, 0, 0, 1]], axis=0)

            T = np.eye(4)
            T[:3, :3] = R.from_euler('X', -90, degrees=True).as_matrix()
            mat = np.matmul(T, mat)

            cv.imwrite(os.path.join(image_path, str(frame_nr) + "_bottom.png"), outPix[edge * 2:, edge * 2:edge * 3])

            frame_infos.append((frame_nr, "bottom", mat))

        # back
        if 'back' in args.faces:
            r_r = r * R.from_euler('Z', 180, degrees=True)
            mat = np.append(r_r.as_matrix(), t, axis=1)
            mat = np.append(mat, [[0, 0, 0, 1]], axis=0)

            T = np.eye(4)
            T[:3, :3] = R.from_euler('X', -90, degrees=True).as_matrix()
            mat = np.matmul(T, mat)

            cv.imwrite(os.path.join(image_path, str(frame_nr) + "_back.png"), outPix[edge:edge * 2, edge * 0:edge * 1])

            frame_infos.append((frame_nr, "back", mat))

        # left
        if 'left' in args.faces:
            r_r = r * R.from_euler('Z', 180, degrees=True)
            r_r = r_r * R.from_euler('Y', -90, degrees=True)
            mat = np.append(r_r.as_matrix(), t, axis=1)
            mat = np.append(mat, [[0, 0, 0, 1]], axis=0)

            T = np.eye(4)
            T[:3, :3] = R.from_euler('X', -90, degrees=True).as_matrix()
            mat = np.matmul(T, mat)

            cv.imwrite(os.path.join(image_path, str(frame_nr) + "_left.png"), outPix[edge:edge * 2, edge * 1:edge * 2])

            frame_infos.append((frame_nr, "left", mat))

        # front
        if 'front' in args.faces:
            r_r = r * R.from_euler('Z', 180, degrees=True)
            r_r = r_r * R.from_euler('Y', 180, degrees=True)

            mat = np.append(r_r.as_matrix(), t, axis=1)
            mat = np.append(mat, [[0, 0, 0, 1]], axis=0)

            T = np.eye(4)
            T[:3, :3] = R.from_euler('X', -90, degrees=True).as_matrix()
            mat = np.matmul(T, mat)

            cv.imwrite(os.path.join(image_path, str(frame_nr) + "_front.png"), outPix[edge:edge * 2, edge * 2:edge * 3])

            frame_infos.append((frame_nr, "front", mat))

        # right
        if 'right' in args.faces:
            r_r = r * R.from_euler('Z', 180, degrees=True)
            r_r = r_r * R.from_euler('Y', 90, degrees=True)
            mat = np.append(r_r.as_matrix(), t, axis=1)
            mat = np.append(mat, [[0, 0, 0, 1]], axis=0)

            T = np.eye(4)
            T[:3, :3] = R.from_euler('X', -90, degrees=True).as_matrix()
            mat = np.matmul(T, mat)

            cv.imwrite(os.path.join(image_path, str(frame_nr) + "_right.png"), outPix[edge:edge * 2, edge * 3:edge * 4])

            frame_infos.append((frame_nr, "right", mat))

        frame_nr += 1

    max_x = -math.inf
    min_x = math.inf
    max_y = -math.inf
    min_y = math.inf
    max_z = -math.inf
    min_z = math.inf

    for info in frame_infos:
        if info[2][0][3] > max_x:
            max_x = info[2][0][3]
        if info[2][0][3] < min_x:
            min_x = info[2][0][3]
        if info[2][1][3] > max_y:
            max_y = info[2][1][3]
        if info[2][1][3] < min_y:
            min_y = info[2][1][3]
        if info[2][2][3] > max_z:
            max_z = info[2][2][3]
        if info[2][2][3] < min_z:
            min_z = info[2][2][3]

    for point in landmarks:
        if point[0] > max_x:
            max_x = point[0]
        if point[0] < min_x:
            min_x = point[0]
        if point[1] > max_y:
            max_y = point[1]
        if point[1] < min_y:
            min_y = point[1]
        if point[2] > max_z:
            max_z = point[2]
        if point[2] < min_z:
            min_z = point[2]

    for point in dense_points_pos:
        if point[0] > max_x:
            max_x = point[0]
        if point[0] < min_x:
            min_x = point[0]
        if point[1] > max_y:
            max_y = point[1]
        if point[1] < min_y:
            min_y = point[1]
        if point[2] > max_z:
            max_z = point[2]
        if point[2] < min_z:
            min_z = point[2]


    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z

    scale_factor = 1 / max(x_range, y_range, z_range)

    for info in frame_infos:
        info[2][0][3] -= max_x - x_range / 2
        info[2][1][3] -= max_y - y_range / 2
        info[2][2][3] -= max_z - z_range / 2

        info[2][0][3] *= scale_factor * 2
        info[2][1][3] *= scale_factor * 2
        info[2][2][3] *= scale_factor * 2

    for point in landmarks:
        point[0] -= max_x - x_range / 2
        point[1] -= max_y - y_range / 2
        point[2] -= max_z - z_range / 2

        point[0] *= scale_factor * 2
        point[1] *= scale_factor * 2
        point[2] *= scale_factor * 2

    for point in dense_points_pos:
        point[0] -= max_x - x_range / 2
        point[1] -= max_y - y_range / 2
        point[2] -= max_z - z_range / 2

        point[0] *= scale_factor * 2
        point[1] *= scale_factor * 2
        point[2] *= scale_factor * 2


    point_cloud_to_ply(os.path.join(args.output_folder, "sparse.ply"), landmarks, list())
    point_cloud_to_ply(os.path.join(args.output_folder, "dense.ply"), dense_points_pos, dense_points_color)

    json_data = dict()

    json_data["ply_file_path"] = "dense.ply"

    json_data["fl_x"] = float(edge / 2)
    json_data["fl_y"] = float(edge / 2)

    json_data["cx"] = float(edge / 2)
    json_data["cy"] = float(edge / 2)

    json_data["w"] = float(edge)
    json_data["h"] = float(edge)

    json_data["frames"] = list()
    for info in frame_infos:
        json_data['frames'].append({"file_path": 'images/' + str(info[0]) + "_" + info[1] + ".png", "transform_matrix": info[2].tolist()})

    with open(os.path.join(args.output_folder, "transforms.json"), 'w') as outfile:
        json.dump(json_data, outfile, indent=4)


if __name__ == "__main__":
    main()

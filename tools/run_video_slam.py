#!/usr/bin/env python3

import os
import time
import cv2 as cv
import numpy as np
import color_scheme as cs
from argparse import ArgumentParser
from stellapy import StellaVSLAM
from viser import ViserServer

def main():
    # Parse arguments
    parser = ArgumentParser("StellaVSLAM")

    # General options
    parser.add_argument("--log-level", default="info", help="log level")
    parser.add_argument("--start-paused", action="store_true", help="start the SLAM process in paused state")
    parser.add_argument("--auto-term", action="store_true", help="automatically terminate when the video ends")

    # Input options
    parser.add_argument("-v", "--vocab", required=True, help="vocabulary file path")
    parser.add_argument("-c", "--config", required=True, help="config file path")
    parser.add_argument("-m", "--video", required=True, help="video file path")
    parser.add_argument("--mask", default="", help="mask image path")
    parser.add_argument("--frame-step", type=int, default=1, help="step size of frame")
    parser.add_argument("--wait", action="store_true", help="wait to enforce real-time processing")
    parser.add_argument("-s", "--start-time", type=int, default=0, help="time to start playing [milli seconds]")
    parser.add_argument("--start-timestamp", type=float, default=0.0, help="timestamp of the start of the video capture")

    # Mapping options
    parser.add_argument("--disable-mapping", action="store_true", help="disable mapping")
    parser.add_argument("--temporal-mapping", action="store_true", help="enable temporal mapping")
    parser.add_argument("--wait-loop-ba", action="store_true", help="wait until the loop BA is finished")

    # Output options
    parser.add_argument("-p", "--pc-out", default="", help="store point cloud at this path after slam")
    parser.add_argument("-k", "--kf-out", default="", help="store keyframes in this folder after slam")
    parser.add_argument("-i", "--map-db-in", default="", help="load a map from this path")
    parser.add_argument("-o", "--map-db-out", default="", help="store a map database at this path after slam")
    parser.add_argument("--eval-log-dir", default="", help="store trajectory and tracking times at this path (Specify the directory where it exists.)")

    # Parse arguments
    args = parser.parse_args()

    # Open video
    cap = cv.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(1)
    cap.set(cv.CAP_PROP_POS_MSEC, args.start_time)

    # Load mask if provided
    if args.mask:
        mask = cv.imread(args.mask, cv.IMREAD_GRAYSCALE)
        if mask is None:
            print("Error: Could not load mask image.")
            exit(1)
    else:
        mask = np.array([])

    # Setup Viewer
    server = ViserServer()
    server.scene.set_up_direction("-y")
    server.gui.configure_theme(control_width="large", dark_mode=True, brand_color=cs.WH_LOGO)

    # Add GUI elements
    image_view = server.gui.add_image(np.zeros((1, 2, 3), dtype="uint8"))
    # camera_mode = server.gui.add_dropdown("Camera Mode", ["Free", "LookAt", "Follow", "Lock"], "Free", hint="Set camera mode")
    mapping = server.gui.add_checkbox("Mapping", not args.disable_mapping, hint="Enable/Disable mapping")
    temporal_mapping = server.gui.add_checkbox("Temporal Mapping", args.temporal_mapping, disabled=True, hint="Enable/Disable temporal mapping")
    dense_reconstruction = server.gui.add_checkbox("Dense Reconstruction", True, hint="Enable/Disable dense reconstruction")
    loop_detection = server.gui.add_checkbox("Loop Detection", not args.temporal_mapping, disabled=args.temporal_mapping, hint="Enable/Disable loop detection")
    wait_loop_ba = server.gui.add_checkbox("Wait Loop BA", args.wait_loop_ba, hint="Enable/Disable waiting for loop BA")
    covisibility_min_shared = server.gui.add_slider("Covisibility Minimum Shared Landmarks", 10, 500, 10, 100, hint="Minimum shared landmarks for covisibility edge")
    world_scale = server.gui.add_slider("World Scale", 0.01, 10.0, 0.1, 1.0, hint="Scale of the world visualization")
    pause = server.gui.add_button("Unpause" if args.start_paused else "Pause", hint="Pause/Resume the SLAM process")
    step = server.gui.add_button("Step", hint="Process one frame when paused", disabled=not args.start_paused)
    reset = server.gui.add_button("Reset SLAM", hint="Request a full reset of the SLAM system")
    terminate = server.gui.add_button("Terminate SLAM", hint="Request termination of the SLAM system")

    # Add scene elements
    camera = server.scene.add_camera_frustum("camera", np.pi/2, 2/1, color=cs.CAMERA)
    landmarks_all = server.scene.add_point_cloud("all_landmarks", np.zeros((1, 3), dtype="float32"), cs.LANDMARK_ALL, point_shape="rounded")
    landmarks_local = server.scene.add_point_cloud("local_landmarks", np.zeros((1, 3), dtype="float32"), cs.LANDMARK_LOCAL, point_shape="sparkle")
    dense_points = server.scene.add_point_cloud("dense_points", np.zeros((1, 3), dtype="float32"), np.zeros((1, 3), dtype="uint8"), point_shape="circle")

    spanning_tree = server.scene.add_line_segments("spanning_tree", np.zeros((0, 2, 3), dtype="float32"), cs.SPANNING_TREE)
    loop = server.scene.add_line_segments("loop", np.zeros((0, 2, 3), dtype="float32"), cs.LOOP)
    covisibility = server.scene.add_line_segments("covisibility", np.zeros((0, 2, 3), dtype="float32"), cs.GRAPH)
    trajectory = server.scene.add_line_segments("trajectory", np.zeros((0, 2, 3), dtype="float32"), cs.TRAJECTORY)

    # DEBUGGING: Add time profiling plot
    runtime_profiling = server.gui.add_uplot(
        (
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
        ),
        (
            cs.SERIES_TIME,
            cs.SERIES_TRACKING,
            cs.SERIES_VISUALIZATION,
            cs.SERIES_IMAGE_VIEW,
            cs.SERIES_LANDMARKS,
            cs.SERIES_DENSE_POINTS,
            cs.SERIES_KEYFRAME_GRAPH,
            cs.SERIES_TRAJECTORY,
            cs.SERIES_PROCESSING,
            cs.SERIES_RT_DEADLINE,
        ),
        aspect=2/1,
        visible=args.log_level == "debug",
    )
    timestamps = list()
    times_tracking = list()
    times_image_view = list()
    times_landmarks = list()
    times_dense_points = list()
    times_keyframe_graph = list()
    times_trajectory = list()
    times_visualization = list()
    times_processing = list()
    runtime_image_view = 0.0
    runtime_landmarks = 0.0
    runtime_dense_points = 0.0
    runtime_keyframe_graph = 0.0
    runtime_trajectory = 0.0
    runtime_visualization = 0.0
    runtime_processing = 0.0

    # Bringup SLAM
    slam = StellaVSLAM(args.config, args.vocab, args.log_level)
    slam.startup(not args.map_db_in)
    if args.map_db_in:
        slam.load_map_database(args.map_db_in)
    if args.disable_mapping:
        slam.disable_mapping()
    if args.temporal_mapping:
        slam.enable_temporal_mapping()
        slam.disable_loop_detection()
    if slam.dense_reconstruction_is_enabled():
        dense_reconstruction.value = True
        dense_reconstruction.disabled = False
        dense_reconstruction.on_update(lambda v: slam.enable_dense_reconstruction() if v.target.value else slam.disable_dense_reconstruction())
    else:
        dense_reconstruction.value = False
        dense_reconstruction.disabled = True

    # Wire SLAM inputs
    reset.on_click(lambda _: slam.reset())
    terminate.on_click(lambda _: slam.terminate())
    loop_detection.on_update(lambda v: slam.enable_loop_detection() if v.target.value else slam.disable_loop_detection())
    mapping.on_update(lambda v: slam.enable_mapping() if v.target.value else slam.disable_mapping())

    # Wire control variables
    paused = args.start_paused
    def toggle_pause(_):
        nonlocal paused
        if paused:
            paused = False
            pause.label = "Pause"
            step.disabled = True
        else:
            paused = True
            pause.label = "Unpause"
            step.disabled = False
    pause.on_click(toggle_pause)
    stepping = False
    def step_once(_):
        nonlocal stepping
        stepping = True
    step.on_click(step_once)

    # Constant variables
    frame_duration = args.frame_step / cap.get(cv.CAP_PROP_FPS)

    # Loop variables
    keyframes = dict()
    trajectory_poses = list()
    timestamp = args.start_timestamp

    # Main loop
    while not slam.terminate_is_requested():
        start_processing = time.time()

        # Process next frame
        if not paused or stepping:
            stepping = False

            # Read next frame
            ok, img = cap.read()

            if ok:
                # DEBUGGING: Update time profiling plot
                if timestamp > args.start_timestamp:
                    timestamps.append(timestamp)
                    times_image_view.append(runtime_image_view)
                    times_landmarks.append(runtime_landmarks)
                    times_dense_points.append(runtime_dense_points)
                    times_keyframe_graph.append(runtime_keyframe_graph)
                    times_trajectory.append(runtime_trajectory)
                    times_visualization.append(runtime_visualization)
                    times_processing.append(runtime_processing)

                    runtime_profiling.data = (
                        np.array(timestamps),
                        np.array(times_tracking),
                        np.array(times_visualization),
                        np.array(times_image_view),
                        np.array(times_landmarks),
                        np.array(times_dense_points),
                        np.array(times_keyframe_graph),
                        np.array(times_trajectory),
                        np.array(times_processing),
                        np.array([frame_duration for _ in timestamps]),
                    )

                # Wait for loop BA if requested
                if wait_loop_ba.value:
                    while slam.loop_ba_is_running():
                        time.sleep(0.01)

                # Clear visualizations if reset is requested
                if slam.reset_is_requested():
                    trajectory_poses.clear()

                # Track next frame
                start_tracking = time.time()
                position, orientation = slam.feed_monocular_frame(img, timestamp, mask)
                tracking_time = time.time() - start_tracking
                times_tracking.append(tracking_time)

                # Update viewer pose
                camera.position = position * world_scale.value
                camera.wxyz = orientation

                # Append current pose to trajectory
                trajectory_poses.append(position)

                # Advance timestamp
                timestamp += frame_duration

                # Skip frames if requested
                for _ in range(args.frame_step - 1):
                    cap.grab()

            # Terminate when video ends if requested
            elif args.auto_term:
                print("End of video.")
                slam.terminate()

        start_visualization = time.time()

        # Update tracking image
        start_image_view = time.time()
        image_view.image = slam.draw_frame()
        runtime_image_view = time.time() - start_image_view

        # Update landmarks
        start_landmarks = time.time()
        all_lms, local_lms = slam.get_all_landmarks()
        landmarks_all.points = all_lms * world_scale.value
        landmarks_local.points = local_lms * world_scale.value
        runtime_landmarks = time.time() - start_landmarks

        # Update dense points
        start_dense_points = time.time()
        points, colors = slam.get_dense_points()
        dense_points.points = points * world_scale.value
        dense_points.colors = colors
        runtime_dense_points = time.time() - start_dense_points

        # Get new keyframes data
        start_keyframe_graph = time.time()
        keyframe_pose, spanning_tree_edges, loop_edges, covisibility_edges = slam.get_keyframe_graph(covisibility_min_shared.value)

        # Update keyframes
        # deleted
        for id in keyframes.keys() - keyframe_pose.keys():
            keyframes[id].remove()
            keyframes.pop(id)
        # new
        for id in keyframe_pose.keys() - keyframes.keys():
            position, orientation = keyframe_pose[id]
            keyframe = server.scene.add_camera_frustum(f"keyframes/{id}", np.pi/2, 2/1, color=cs.KEYFRAME, position=position * world_scale.value, wxyz=orientation)
            keyframes[id] = keyframe
        # updated
        for id in keyframe_pose.keys() & keyframes.keys():
            position, orientation = keyframe_pose[id]
            keyframes[id].position = position * world_scale.value
            keyframes[id].wxyz = orientation

        # Update spanning tree
        spanning_tree.points = spanning_tree_edges * world_scale.value
        # Update loop edges
        loop.points = loop_edges * world_scale.value
        # Update covisibility edges
        covisibility.points = covisibility_edges * world_scale.value

        runtime_keyframe_graph = time.time() - start_keyframe_graph

        # Update trajectory
        start_trajectory = time.time()
        trajectory.points = np.stack([trajectory_poses[:-1], trajectory_poses[1:]], axis=1) * world_scale.value
        runtime_trajectory = time.time() - start_trajectory

        runtime_visualization = time.time() - start_visualization

        # Sleep to enforce real-time processing if requested
        runtime_processing = time.time() - start_processing
        sleep_duration = frame_duration - runtime_processing
        if sleep_duration > 0 and (paused or args.wait):
            time.sleep(sleep_duration)

    # Stop SLAM
    slam.shutdown()

    # Save reqested outputs
    if args.map_db_out:
        slam.save_map_database(args.map_db_out)
    if args.pc_out:
        slam.save_point_cloud(args.pc_out)
    if args.kf_out:
        slam.save_keyframes(args.kf_out)
    if args.eval_log_dir:
        try:
            os.mkdir(args.eval_log_dir)
        except FileExistsError:
            pass
        slam.save_frame_trajectory(args.eval_log_dir + "/frame_trajectory.txt", "TUM")
        slam.save_keyframe_trajectory(args.eval_log_dir + "/keyframe_trajectory.txt", "TUM")
        with open(args.eval_log_dir + "/track_times.txt", "w") as f:
            for t in times_tracking:
                f.write(f"{t}\n")

if __name__ == "__main__":
    main()

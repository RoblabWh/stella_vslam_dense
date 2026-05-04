#!/usr/bin/env python3

import os
import time
import cv2 as cv
import numpy as np
import color_scheme as cs
from argparse import ArgumentParser
from stellapy import StellaVSLAM
from tqdm import tqdm
from viser import ViserServer
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, wait

def main():
    ## Parse arguments
    parser = ArgumentParser("StellaVSLAM")

    # General options
    parser.add_argument("--log-level", default="info", help="log level", choices=cs._SPDLOG_LEVEL_TO_PY.keys())
    parser.add_argument("--start-paused", action="store_true", help="start the SLAM process in paused state")
    parser.add_argument("--auto-term", action="store_true", help="automatically terminate when the video ends")
    parser.add_argument("--disable-viewer", action="store_true", help="disable viewer and run SLAM headless")

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
    parser.add_argument("--disable-dense", action="store_true", help="disable dense reconstruction")
    parser.add_argument("--wait-loop-ba", action="store_true", help="wait until the loop BA is finished")

    # Output options
    parser.add_argument("-p", "--pc-out", default="", help="store point cloud at this path after slam")
    parser.add_argument("-k", "--kf-out", default="", help="store keyframes in this folder after slam")
    parser.add_argument("-i", "--map-db-in", default="", help="load a map from this path")
    parser.add_argument("-o", "--map-db-out", default="", help="store a map database at this path after slam")
    parser.add_argument("--eval-log-dir", default="", help="store trajectory and tracking times at this path (Specify the directory where it exists.)")

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    logger = cs.setup_logger(args.log_level)

    # Open video
    if not os.path.isfile(args.video):
        parser.error(f"Video file not found: {args.video}")
    cap = cv.VideoCapture(args.video)
    if not cap.isOpened():
        parser.error(f"Could not open video: {args.video}")
    if args.start_time and not cap.set(cv.CAP_PROP_POS_MSEC, args.start_time):
        logger.warning("Could not seek to %d ms", args.start_time)

    # Load mask if provided
    if args.mask:
        if not os.path.isfile(args.mask):
            parser.error(f"Mask file not found: {args.mask}")
        mask = cv.imread(args.mask, cv.IMREAD_GRAYSCALE)
        if mask is None:
            parser.error(f"Could not decode mask image: {args.mask}")
    else:
        mask = np.array([])


    ## Viewer Setup
    viewer_enabled = not args.disable_viewer
    if viewer_enabled:
        # Setup Viewer
        server = ViserServer()
        server.scene.set_up_direction("-y")
        server.gui.configure_theme(control_width="large", dark_mode=True, brand_color=cs.WH_LOGO)

        # Add GUI elements
        image_view = server.gui.add_image(np.zeros((1, 2, 3), dtype="uint8"))
        progress = server.gui.add_progress_bar(0.0, animated=not args.start_paused)
        # camera_mode = server.gui.add_dropdown("Camera Mode", ["Free", "LookAt", "Follow", "Lock"], "Free", hint="Set camera mode")
        mapping = server.gui.add_checkbox("Mapping", not args.disable_mapping, hint="Enable/Disable mapping")
        temporal_mapping = server.gui.add_checkbox("Temporal Mapping", args.temporal_mapping, disabled=True, hint="Enable/Disable temporal mapping")
        dense_reconstruction = server.gui.add_checkbox("Dense Reconstruction", not args.disable_dense, hint="Enable/Disable dense reconstruction")
        loop_detection = server.gui.add_checkbox("Loop Detection", not args.temporal_mapping, disabled=args.temporal_mapping, hint="Enable/Disable loop detection")
        wait_loop_ba = server.gui.add_checkbox("Wait Loop BA", args.wait_loop_ba, hint="Enable/Disable waiting for loop BA")
        wait_real_time = server.gui.add_checkbox("Wait Real-Time", args.wait, hint="Enable/Disable waiting to enforce real-time processing")
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

        #DEBUG: Add time profiling plot
        #NOTE: This relies on the playback never beeing paused and is still a bit hacky
        runtime_profiling = None
        if args.log_level == "debug":
            runtime_profiling = server.gui.add_uplot(
                ( np.array([0.0]), ) * len(cs.RUNTIME_SERIES),
                cs.RUNTIME_SERIES,
                aspect=2/1,
            )
        times_visualization = list()
        times_image_view = Queue()
        times_landmarks = Queue()
        times_dense_points = Queue()
        times_keyframe_graph = Queue()
        times_trajectory = Queue()

    times_tracking = list()
    times_processing = list()
    times_frame_read = Queue()
    times_frame_skip = Queue()


    ## Bringup SLAM
    slam = StellaVSLAM(args.config, args.vocab, args.log_level)
    slam.startup(not args.map_db_in)
    if args.map_db_in:
        slam.load_map_database(args.map_db_in)
    if args.disable_mapping:
        slam.disable_mapping()
    if args.temporal_mapping:
        slam.enable_temporal_mapping()
        slam.disable_loop_detection()
    if slam.dense_reconstruction_is_enabled() and args.disable_dense:
        slam.disable_dense_reconstruction()

    paused = False
    stepping = False
    if viewer_enabled:
        paused = args.start_paused
        if slam.dense_reconstruction_is_available():
            dense_reconstruction.on_update(lambda v: slam.enable_dense_reconstruction() if v.target.value else slam.disable_dense_reconstruction())
        else:
            dense_reconstruction.disabled = True
            dense_reconstruction.value = False

        # Wire SLAM inputs
        reset.on_click(lambda _: slam.reset())
        terminate.on_click(lambda _: slam.terminate())
        loop_detection.on_update(lambda v: slam.enable_loop_detection() if v.target.value else slam.disable_loop_detection())
        mapping.on_update(lambda v: slam.enable_mapping() if v.target.value else slam.disable_mapping())

        # Wire control variables
        def toggle_pause(_):
            nonlocal paused
            if paused:
                paused = False
                pause.label = "Pause"
                step.disabled = True
                progress.animated = True
            else:
                paused = True
                pause.label = "Unpause"
                step.disabled = False
                progress.animated = False
        pause.on_click(toggle_pause)
        def step_once(_):
            nonlocal stepping
            stepping = True
        step.on_click(step_once)


    # Constant variables
    frame_duration = args.frame_step / cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    total_steps = max(1, total_frames // args.frame_step)

    # Loop variables
    keyframes = dict()
    trajectory_poses = list()
    timestamp = args.start_timestamp


    ## Setup async processing
    pool = ThreadPoolExecutor()
    futures = dict()

    # Frame reader
    frame_queue = Queue(maxsize=1)
    def frame_reader():
        while not slam.terminate_is_requested():
            # Read next frame
            start_frame_read = time.monotonic()
            ok, img = cap.read()
            times_frame_read.put(time.monotonic() - start_frame_read)
            frame_queue.put(img)
            if not ok:
                break

            # Skip frames if requested
            start_frame_skip = time.monotonic()
            for _ in range(args.frame_step - 1):
                cap.grab()
            times_frame_skip.put(time.monotonic() - start_frame_skip)
        # Unblock main thread on termination if necessary
        if frame_queue.empty():
            frame_queue.put(None)
    pool.submit(frame_reader)


    # Define visualization update functions

    def update_image_view():
        start_image_view = time.monotonic()
        image_view.image = slam.draw_frame()
        times_image_view.put(time.monotonic() - start_image_view)

    def update_landmarks():
        start_landmarks = time.monotonic()
        all_lms, local_lms = slam.get_all_landmarks()
        landmarks_all.points = all_lms * world_scale.value
        landmarks_local.points = local_lms * world_scale.value
        times_landmarks.put(time.monotonic() - start_landmarks)

    def update_dense_points():
        start_dense_points = time.monotonic()
        points, colors = slam.get_dense_points()
        dense_points.points = points * world_scale.value
        dense_points.colors = colors
        times_dense_points.put(time.monotonic() - start_dense_points)

    def update_keyframe_graph():
        start_keyframe_graph = time.monotonic()

        # Get new keyframes data
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

        times_keyframe_graph.put(time.monotonic() - start_keyframe_graph)

    def update_trajectory():
        start_trajectory = time.monotonic()
        trajectory.points = np.stack([trajectory_poses[:-1], trajectory_poses[1:]], axis=1) * world_scale.value
        times_trajectory.put(time.monotonic() - start_trajectory)


    ## Main loop
    pbar = tqdm(total=total_steps, desc="Processing frames", unit=" frames")
    while not slam.terminate_is_requested():
        start_processing = time.monotonic()

        # Process next frame
        if not paused or stepping:
            stepping = False

            # Get next frame
            img = frame_queue.get()

            if img is not None:
                # Wait for loop BA if requested
                if wait_loop_ba.value if viewer_enabled else args.wait_loop_ba:
                    while slam.loop_ba_is_running():
                        time.sleep(0.001)

                # Clear visualizations if reset is requested
                if slam.reset_is_requested():
                    trajectory_poses.clear()

                # Track next frame
                start_tracking = time.monotonic()
                tracking = slam.feed_monocular_frame(img, timestamp, mask)
                tracking_time = time.monotonic() - start_tracking
                times_tracking.append(tracking_time)

                if tracking is not None:
                    position, orientation = tracking

                    # Update viewer pose
                    if viewer_enabled:
                        camera.position = position * world_scale.value
                        camera.wxyz = orientation

                    # Append current pose to trajectory
                    trajectory_poses.append(position)

                # Advance progress
                timestamp += frame_duration
                pbar.update()
                if viewer_enabled:
                    progress.value = pbar.n / total_steps * 100

            # Retry and check for termination if video hasn't ended
            elif pbar.n < total_steps:
                logger.debug("No frame received, but video hasn't ended, retrying...")
                pass

            # Terminate when video ends if requested
            elif args.disable_viewer or args.auto_term:
                logger.info("End of video.")
                slam.terminate()

            # Pause at the end of the video
            else:
                logger.info("End of video, waiting for termination...")
                pause.disabled = True
                step.disabled = True
                pause.label = "End of Video"
                progress.animated = False
                paused = True
                stepping = False


        # Update visualizations
        if viewer_enabled:
            start_visualization = time.monotonic()

            # Update tracking image
            if update_image_view in futures:
                wait([futures[update_image_view]])
            futures[update_image_view] = pool.submit(update_image_view)

            # Update landmarks
            if update_landmarks not in futures or futures[update_landmarks].done():
                futures[update_landmarks] = pool.submit(update_landmarks)
            else:
                logger.debug("Landmarks update is taking too long, skipping update")
                times_landmarks.put(np.nan)

            # Update dense points
            if update_dense_points not in futures or futures[update_dense_points].done():
                futures[update_dense_points] = pool.submit(update_dense_points)
            else:
                logger.debug("Dense points update is taking too long, skipping update")
                times_dense_points.put(np.nan)

            # Update keyframe graph
            if update_keyframe_graph not in futures or futures[update_keyframe_graph].done():
                futures[update_keyframe_graph] = pool.submit(update_keyframe_graph)
            else:
                logger.debug("Keyframe graph update is taking too long, skipping update")
                times_keyframe_graph.put(np.nan)

            # Update trajectory
            if update_trajectory not in futures or futures[update_trajectory].done():
                futures[update_trajectory] = pool.submit(update_trajectory)
            else:
                logger.debug("Trajectory update is taking too long, skipping update")
                times_trajectory.put(np.nan)

            #DEBUG: Update runtime profiling plot
            if runtime_profiling is not None:
                timestamps = np.arange(0, timestamp - frame_duration / 2, frame_duration)
                runtime_profiling.data = (
                    timestamps,
                    np.array(times_processing, like=timestamps),
                    np.array(times_tracking, like=timestamps),
                    np.array(times_visualization, like=timestamps),
                    np.array(times_frame_read.queue, like=timestamps),
                    np.array(times_frame_skip.queue, like=timestamps),
                    np.array(times_image_view.queue, like=timestamps),
                    np.array(times_landmarks.queue, like=timestamps),
                    np.array(times_dense_points.queue, like=timestamps),
                    np.array(times_keyframe_graph.queue, like=timestamps),
                    np.array(times_trajectory.queue, like=timestamps),
                    np.full(len(timestamps), frame_duration),
                )

            times_visualization.append(time.monotonic() - start_visualization)

        # Sleep to enforce real-time processing if requested
        runtime_processing = time.monotonic() - start_processing
        sleep_duration = frame_duration - runtime_processing
        if sleep_duration > 0 and (paused or (wait_real_time.value if viewer_enabled else args.wait)):
            time.sleep(sleep_duration)
        times_processing.append(runtime_processing)

    # Unblock frame reader on termination if necessary
    if frame_queue.full():
        frame_queue.get()

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
        with open(args.eval_log_dir + "/tracking_times.txt", "w") as f:
            for t in times_tracking:
                f.write(f"{t}\n")

if __name__ == "__main__":
    main()

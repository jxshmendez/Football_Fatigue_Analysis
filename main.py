import cv2
import numpy as np
from utilities import load_video, save_video
from tracking import ObjectTracker
from team_classifier import TeamClassifier
from ball_assigner import BallAssigner
from camera_movement import CameraMovement
from camera_transform import CameraTransform
from speedDistance_estimator import SpeedDistanceEstimator
from track_merger import merge_player_tracks
from simulate_speed_history import simulate_speed_history
from PlayerPerformanceVisualiser import PlayerPerformanceVisualizer
from ultralytics import YOLO
import time
import torch
import onnxruntime as ort

def process_footage(input_video, output_path, use_stub=True, stub_path="stubs/track_stubs.pkl"):
    start_time = time.time()
    print("Processing started...")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    # Model setup code remains the same...
    model = YOLO("models/best.pt").to("mps")
    model.export(format="onnx", opset=12)
    model.to(device)
    session = ort.InferenceSession(
        "models/best.onnx",
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    print("ONNXRuntime Execution Provider:", session.get_providers())
    print("Available devices:", torch.cuda.device_count())
    print("Current device:", device)
    print("MPS Available:", torch.backends.mps.is_available())

    # Read frames from the video
    frames = [torch.tensor(frame).to("mps") for frame in load_video(input_video)]

    # Create a tracker for objects
    main_tracker = ObjectTracker('models/best.pt')

    # Retrieve track info (using the stub only if use_stub is True)
    track_data = main_tracker.retrieve_object_tracks(
        frames,
        use_stub=use_stub,
        cache_path=stub_path
    )

    # Insert positional data (e.g., foot positions) into each track
    main_tracker.assign_positions(track_data)
    
    # Set up camera motion estimator and get per-frame camera movement
    cam_estimator = CameraMovement(frames[0])
    camera_shifts = cam_estimator.calculate_camera_motion(
        frames,
        use_cache=True,
        cache_path='stubs/camera_movement_stub.pkl'
    )
    cam_estimator.apply_camera_shift(track_data, camera_shifts)
    
    # Transform the view to real-world coordinates
    transformer = CameraTransform()
    transformer.apply_transform_to_tracks(track_data)
    
    # Merge tracks for players to handle re-identification issues
    # After assigning positions and applying camera motion/transform
    #track_data["players"] = merge_player_tracks(track_data["players"], frames, max_distance=50, hist_threshold=0.5)
    
    
    # Fill in any missing ball positions
    
    #track_data["ball"] = main_tracker.smooth_ball_trajectory(track_data["ball"])
    # track_data["players"] = main_tracker.player_trajectory(track_data["players"]) #player smoothing test code
    
    # Compute speed and distance for players
    # Compute speed and distance for players
    speed_distance_calc = SpeedDistanceEstimator()
    speed_distance_calc.compute_speed_and_distance(track_data)

    # Assign team colors (only if player tracks exist)
    if 'players' in track_data and len(track_data['players']) > 0 and len(track_data['players'][0]) > 0:
        team_finder = TeamClassifier()
        team_finder.define_team_colors(frames[0], track_data['players'][0])
        # Loop only up to the minimum number of frames between frames and track_data
        min_frames = min(len(frames), len(track_data['players']))
        for idx in range(min_frames):
            player_info = track_data['players'][idx]
            for pid, record in player_info.items():
                # Make sure the frame index exists
                if idx < len(frames):
                    team_label = team_finder.classify_player_team(frames[idx], record['bbox'], pid)
                    track_data['players'][idx][pid]['team'] = team_label
                    track_data['players'][idx][pid]['team_color'] = team_finder.team_centers[team_label]
    else:
        print("No player tracks found; skipping team color assignment.")


            
    # Determine ball possession
    ball_assigner = BallAssigner()
    
    '''
    possession_sequence = []
    for idx, player_info in enumerate(track_data['players']):
        ball_box = track_data['ball'][idx][1]['bbox']
        player_with_ball = ball_assigner.find_ball_holder(player_info, ball_box)
        if player_with_ball != -1:
            track_data['players'][idx][player_with_ball]['has_ball'] = True
            possession_sequence.append(track_data['players'][idx][player_with_ball]['team'])
        else:
            possession_sequence.append(possession_sequence[-1] if possession_sequence else -1)
    possession_sequence = np.array(possession_sequence)
    '''
    
    # Annotate frames with tracks and ball control
    annotated = main_tracker.annotate_video_frames(frames, track_data) #possession_sequence - add this to the param when using possession tracker
    
    # Overlay camera movement info
    annotated = cam_estimator.overlay_camera_motion(annotated, camera_shifts)
    
    # Overlay speed and distance info
    speed_distance_calc.overlay_speed_and_distance(annotated, track_data)
    
    # Optionally, overlay source points for debugging
    for i in range(len(annotated)):
        annotated[i] = transformer.overlay_src_points(annotated[i])
        
    # Save the final annotated video
    save_video(annotated, output_path)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Processing completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    
if __name__ == '__main__':
    input_video = "inputVid/1min30.mp4"  # Change this to your actual input video path
    output_video = "outputVid/processed_video.mp4"  # Change this to your desired output path

    process_footage(input_video, output_video)
    
    visualizer = PlayerPerformanceVisualizer("outputVid/player_5_speed_distance.csv")
    visualizer.plot_all()


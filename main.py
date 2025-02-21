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
import torch_directml
import onnxruntime as ort


def process_footage(video_path, output_path):
    start_time = time.time()
    print("Processing started...")
    device = torch_directml.device()
    print("Using DirectML device:", device)

    # Move YOLO model to DirectML device
    model = YOLO("models/best.pt")
    model.export(format="onnx", opset=12, half=True)
    model.to(device)
    session = ort.InferenceSession("models/best.onnx", providers=["DmlExecutionProvider"])
    print("ONNXRuntime Execution Provider:", session.get_providers())
    # Check if DirectML is working
    print("Available devices:", torch.cuda.device_count())  # Should be at least 1 if working
    print("Current device:", device)
    # Read frames from the video
    frames = load_video(video_path)  # ✅ Uses passed parameter

    
    # Create a tracker for objects (players, ball, referees)
    main_tracker = ObjectTracker('models/best.pt')
    
    # Retrieve track info from cache or by running detection
    track_data = main_tracker.retrieve_object_tracks(
        frames,
        use_stub=True,
        cache_path='stubs/track_stubs.pkl'
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
    speed_distance_calc = SpeedDistanceEstimator()
    speed_distance_calc.compute_speed_and_distance(track_data)
    
      
    # Assign team colors
    team_finder = TeamClassifier()
    team_finder.define_team_colors(frames[0], track_data['players'][0])
    
    # Loop through frames and assign teams
    for idx, player_info in enumerate(track_data['players']):
        for pid, record in player_info.items():
            team_label = team_finder.classify_player_team(frames[idx], record['bbox'], pid)
            track_data['players'][idx][pid]['team'] = team_label
            track_data['players'][idx][pid]['team_color'] = team_finder.team_centers[team_label]
            
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
    process_footage()
    visualizer = PlayerPerformanceVisualizer("outputVid/player_5_speed_distance.csv")
    visualizer.plot_all()  # Call this to plot all graphs at once

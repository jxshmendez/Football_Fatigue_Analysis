import os
import cv2
import numpy as np
import time
import torch
import onnxruntime as ort
from glob import glob
from utilities import load_video, save_video
from tracking import ObjectTracker
from team_classifier import TeamClassifier
from camera_movement import CameraMovement
from camera_transform import CameraTransform
from speedDistance_estimator import SpeedDistanceEstimator, SpeedDistanceEstimatorExtended
from PlayerPerformanceVisualiser import PlayerPerformanceVisualiser
from ultralytics import YOLO

def process_footage(input_video, output_path, single_player_calc, team_calc, use_stub=True, stub_path="stubs/track_stubs.pkl", frame_offset=0):
    start_time = time.time()
    print(f"Processing segment: {input_video}")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    # Model setup
    model = YOLO("models/best.pt").to(device)
    model.export(format="onnx", opset=12)
    model.to(device)
    session = ort.InferenceSession("models/best.onnx", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])

    # Read frames and process without tracking gradients
    with torch.no_grad():
        frames = [torch.tensor(frame).to(device) for frame in load_video(input_video)]
        
        # Run inference or other operations that don't require gradients.
        # For example, if you were to do a forward pass with your model:
        # outputs = model(frames)   # This now doesn't compute or store gradients.

    # The rest of your code (tracking, camera motion, etc.) can remain outside if they require gradient tracking,
    # but typically these are also inference steps, so consider wrapping any additional model-based operations similarly.

    # Object tracking and subsequent processing...
    main_tracker = ObjectTracker('models/best.pt')
    track_data = main_tracker.retrieve_object_tracks(frames, use_stub=use_stub, cache_path=stub_path)
    main_tracker.assign_positions(track_data)

    # Camera movement
    cam_estimator = CameraMovement(frames[0])
    camera_shifts = cam_estimator.calculate_camera_motion(frames, use_cache=True, cache_path='stubs/camera_movement_stub.pkl')
    cam_estimator.apply_camera_shift(track_data, camera_shifts)

    # Coordinate transformation
    transformer = CameraTransform()
    transformer.apply_transform_to_tracks(track_data)

    # Compute single-player metrics
    single_player_calc.compute_speed_and_distance(track_data, frame_offset=frame_offset)

    # Team classification (if this part only classifies without learning, it can be under no_grad as well)
    if 'players' in track_data and len(track_data['players']) > 0 and len(track_data['players'][0]) > 0:
        team_finder = TeamClassifier()
        team_finder.define_team_colors(frames[0], track_data['players'][0])
        min_frames = min(len(frames), len(track_data['players']))
        for idx in range(min_frames):
            for pid, record in track_data['players'][idx].items():
                team_label = team_finder.classify_player_team(frames[idx], record['bbox'], pid)
                track_data['players'][idx][pid]['team'] = team_label
                track_data['players'][idx][pid]['team_color'] = team_finder.team_centers[team_label]
    else:
        print("No player tracks found; skipping team color assignment.")

    # Compute team-based metrics
    team_metrics = team_calc.compute_all_players_speed_distance(track_data, frame_offset=frame_offset)
    print(f"DEBUG: Accumulated team data rows: {len(team_calc.all_player_data)}")

    # Save team CSV (accumulated over segments)
    team_csv_path = os.path.join("outputVid", "team_performance.csv")
    team_calc.save_team_csv(team_csv_path)

    # Annotate video frames and save video
    annotated = main_tracker.annotate_video_frames(frames, track_data)
    
    #annotated = cam_estimator.overlay_camera_motion(annotated, camera_shifts)
    annotated = single_player_calc.overlay_speed_and_distance(annotated, track_data)
    save_video(annotated, output_path)
    end_time = time.time()
    print(f"Segment processed in {end_time - start_time:.2f} seconds")



if __name__ == '__main__':
    # Example usage for processing a single segment.
    # (If you have a separate chunking file, use that instead.)
    input_video = "path/to/your/segment.mp4"  # Replace with your actual segment
    output_path = "outputVid/processed/segment_output.mp4"
    
    # Create instances for single-player and team-based calculations
    single_player_calc = SpeedDistanceEstimator()
    team_calc = SpeedDistanceEstimatorExtended()
    
    # Process the single segment
    process_footage(input_video, output_path, single_player_calc, team_calc, use_stub=False, frame_offset=0)
    
    # After processing, the team performance CSV will be saved to outputVid/team_performance.csv.
    # You can then use your separate visualization/analysis scripts to plot the data.

import cv2
import sys
import numpy as np
import pandas as pd
sys.path.append('../')
from utilities import compute_euclidean_distance, compute_foot_position

class SpeedDistanceEstimator:
    def __init__(self):
        self.interval = 5  # How many frames to skip between measurements
        self.fps = 24  # Frames per second of the video
        self.smoothing_window = 5  # Number of frames for rolling average
        self.max_speed_jump = 5  # Max speed increase allowed per frame (km/h)
        self.target_player_id = 5  # Only store speed for this player ID

    def compute_speed_and_distance(self, track_data):
        """
        Computes speed (km/h) and distance (m) for Player ID 5 and stores results.
        """
        accumulated_distance = {}  # Stores total distance traveled
        speed_history = {}  # Stores past speed values for smoothing
        speed_distance_data = []

        for obj_type, obj_frames in track_data.items():
            if obj_type in ("ball", "referees"):  # Skip ball and referees
                continue

            total_frames = len(obj_frames)
            for start_idx in range(0, total_frames, self.interval):
                end_idx = min(start_idx + self.interval, total_frames - 1)

                for track_id, _ in obj_frames[start_idx].items():
                    if track_id != self.target_player_id:  # Only process player ID 5
                        continue

                    if track_id not in obj_frames[end_idx]:
                        continue

                    start_pos = obj_frames[start_idx][track_id].get('position_transformed')
                    end_pos = obj_frames[end_idx][track_id].get('position_transformed')

                    if not start_pos or not end_pos:
                        continue

                    dist = compute_euclidean_distance(start_pos, end_pos)
                    time_elapsed = (end_idx - start_idx) / self.fps
                    speed_mps = dist / time_elapsed
                    speed_kph = speed_mps * 3.6

                    # Initialize tracking if needed
                    if track_id not in accumulated_distance:
                        accumulated_distance[track_id] = 0
                    if track_id not in speed_history:
                        speed_history[track_id] = []

                    # Check for abnormal speed jumps
                    if speed_history[track_id]:  # If there's previous speed data
                        last_speed = speed_history[track_id][-1]
                        if abs(speed_kph - last_speed) > self.max_speed_jump:
                            speed_kph = (last_speed + speed_kph) / 2  

                    # Maintain rolling speed history
                    speed_history[track_id].append(speed_kph)
                    if len(speed_history[track_id]) > self.smoothing_window:
                        speed_history[track_id].pop(0)

                    # Apply rolling average for final smooth speed
                    smooth_speed = round(np.mean(speed_history[track_id]), 3)  # ✅ Round to 3 decimal places
                    accumulated_distance[track_id] += dist  # Update distance
                    accumulated_distance[track_id] = round(accumulated_distance[track_id], 3)  # ✅ Round distance

                    # Store speed & distance in each frame ONLY for Player ID 5
                    for frame_num in range(start_idx, end_idx):
                        if track_id not in track_data[obj_type][frame_num]:
                            continue

                        track_data[obj_type][frame_num][track_id]['speed'] = smooth_speed
                        track_data[obj_type][frame_num][track_id]['distance'] = accumulated_distance[track_id]

                        # ✅ **Only append Player ID 5's data to the CSV**
                        if track_id == self.target_player_id:
                            speed_distance_data.append({
                                "Frame": frame_num,
                                "Player ID": track_id,
                                "Speed (km/h)": smooth_speed,
                                "Distance Covered (m)": accumulated_distance[track_id]
                            })

        # ✅ **Only save data for Player ID 5**
        df = pd.DataFrame(speed_distance_data)
        df.to_csv("outputVid/player_5_speed_distance.csv", index=False)
        print("Speed and distance data for Player ID 5 saved to CSV.")

    def overlay_speed_and_distance(self, frames, track_data):
        """
        Overlays speed (km/h) and distance (m) on the video frames for Player ID 5.
        """
        annotated_frames = []

        for idx, frame in enumerate(frames):
            for obj_type, obj_frames in track_data.items():
                if obj_type in ("ball", "referees"):
                    continue

                for track_id, track_info in obj_frames[idx].items():
                    if track_id != self.target_player_id:  # Only overlay for Player ID 5
                        continue

                    spd = track_info.get('speed')
                    dist = track_info.get('distance')

                    if spd is None or dist is None:
                        continue

                    # ✅ Format speed & distance to 3 decimal places for display
                    spd_text = f"{spd:.3f} km/h"
                    dist_text = f"{dist:.3f} m"

                    # Display text near the player's feet
                    bbox = track_info['bbox']
                    foot_pos = list(compute_foot_position(bbox))
                    foot_pos[1] += 40
                    foot_pos = tuple(map(int, foot_pos))

                    cv2.putText(frame, spd_text, foot_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, dist_text, (foot_pos[0], foot_pos[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            annotated_frames.append(frame)

        return annotated_frames

    def calculate_fatigue_index(self, speed_data, baseline_speed, window_size=10):
        """
        Computes fatigue index using rolling window of past speed values.
        - Fatigue Index = baseline speed - average speed over last N frames
        """
        if len(speed_data) < window_size:
            current_avg_speed = np.mean(speed_data)  # Use all available data if too short
        else:
            current_avg_speed = np.mean(speed_data[-window_size:])  # Rolling average

        fatigue_index = round(baseline_speed - current_avg_speed, 3)  # ✅ Round to 3 decimal places
        return fatigue_index

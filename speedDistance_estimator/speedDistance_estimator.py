import cv2
import sys
import numpy as np
import pandas as pd
from collections import deque
sys.path.append('../')
from utilities import compute_euclidean_distance, compute_foot_position

class SpeedDistanceEstimator:
    def __init__(self):
        self.interval = 5  # Frames skipped between measurements
        self.fps = 24  # Video FPS
        self.smoothing_window = 5  # Rolling average window
        self.max_acceleration = 3  # Maximum acceleration in m/s²
        self.max_deceleration = 6  # Maximum deceleration in m/s²
        self.max_speed = 31.0  # Max sprint speed (km/h)
        self.target_player_id = 5  # Focus on this player
        self.speed_history = {}  # Store speed history for each player
        self.player_baseline_speed = {}  # Dynamically calculated baseline speeds

    def compute_speed_and_distance(self, track_data):
        """
        Computes speed (km/h), distance (m), and fatigue index while applying acceleration & deceleration capping.
        """
        accumulated_distance = {}  
        speed_distance_data = []
        rolling_window = {}  # Store last N speed values for smoothing

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

                    # ✅ Prevent division by zero
                    if time_elapsed == 0:
                        time_elapsed = 1e-6  # Small nonzero value

                    speed_mps = dist / time_elapsed
                    raw_speed_kph = speed_mps * 3.6  # Convert to km/h

                    # Initialize tracking if needed
                    if track_id not in accumulated_distance:
                        accumulated_distance[track_id] = 0
                    if track_id not in rolling_window:
                        rolling_window[track_id] = deque(maxlen=self.smoothing_window)

                    # Get previous speed for acceleration control
                    prev_speed_kph = rolling_window[track_id][-1] if rolling_window[track_id] else 0

                    # Compute acceleration
                    acceleration_mps2 = (speed_mps - (prev_speed_kph / 3.6)) / time_elapsed

                    # ✅ Apply acceleration & deceleration constraints
                    if acceleration_mps2 > self.max_acceleration:
                        speed_mps = (prev_speed_kph / 3.6) + (self.max_acceleration * time_elapsed)
                    elif acceleration_mps2 < -self.max_deceleration:
                        speed_mps = (prev_speed_kph / 3.6) - (self.max_deceleration * time_elapsed)

                    # ✅ Convert back to km/h & cap top speed
                    speed_kph = round(min(speed_mps * 3.6, self.max_speed), 3)

                    # Store first 100 frames of speed data to determine baseline
                    if track_id not in self.speed_history:
                        self.speed_history[track_id] = []

                    self.speed_history[track_id].append(speed_kph)

                    # ✅ After 100 frames, set player-specific baseline speed
                    if track_id not in self.player_baseline_speed and len(self.speed_history[track_id]) >= 100:
                        self.player_baseline_speed[track_id] = np.mean(self.speed_history[track_id][:100])
                        print(f"Player {track_id} baseline speed set to {self.player_baseline_speed[track_id]:.2f} km/h")

                    # Use player-specific baseline if available, otherwise fallback to 6 km/h
                    baseline_speed = self.player_baseline_speed.get(track_id, 6.0)

                    # Apply **Rolling Median Filter** to smooth sudden jumps
                    rolling_window[track_id].append(speed_kph)
                    median_speed = np.median(rolling_window[track_id])

                    # Apply weighted rolling average
                    smooth_speed = round((0.7 * median_speed) + (0.3 * speed_kph), 3)
                    accumulated_distance[track_id] += dist  # Update distance
                    accumulated_distance[track_id] = round(accumulated_distance[track_id], 3)

                    # ✅ Compute fatigue index using speed history
                    fatigue_index = self.calculate_fatigue_index(list(rolling_window[track_id]), baseline_speed)

                    # Store speed, distance, and fatigue index in each frame ONLY for Player ID 5
                    for frame_num in range(start_idx, end_idx):
                        if track_id not in track_data[obj_type][frame_num]:
                            continue

                        track_data[obj_type][frame_num][track_id]['speed'] = smooth_speed
                        track_data[obj_type][frame_num][track_id]['distance'] = accumulated_distance[track_id]
                        track_data[obj_type][frame_num][track_id]['fatigue_index'] = fatigue_index

                        # ✅ Append only Player 5's data to CSV
                        if track_id == self.target_player_id:
                            speed_distance_data.append({
                                "Frame": frame_num,
                                "Player ID": track_id,
                                "Speed (km/h)": smooth_speed,
                                "Distance Covered (m)": accumulated_distance[track_id],
                                "Fatigue Index": fatigue_index  # ✅ Save fatigue index
                            })

        # ✅ Save only Player 5's data to CSV
        df = pd.DataFrame(speed_distance_data)
        df.to_csv("outputVid/player_5_speed_distance.csv", index=False)
        print("Speed, distance, and fatigue data for Player ID 5 saved to CSV.")

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
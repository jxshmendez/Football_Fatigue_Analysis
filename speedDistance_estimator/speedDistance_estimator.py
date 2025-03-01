import os
import cv2
import numpy as np
import pandas as pd
from collections import deque
from utilities import compute_euclidean_distance, compute_foot_position

class SpeedDistanceEstimator:
    def __init__(self):
        self.interval = 5          # Frames skipped between measurements
        self.fps = 24              # Video FPS
        self.smoothing_window = 5  # Rolling average window
        self.max_acceleration = 3  # Maximum acceleration in m/s²
        self.max_deceleration = 6  # Maximum deceleration in m/s²
        self.max_speed = 31.0      # Max sprint speed (km/h)
        # Retain target_player_id for separate analysis if needed.
        self.target_player_id = 5  
        self.speed_history = {}    # Store speed history for each player
        self.player_baseline_speed = {}  # Baseline speeds once computed
        self.player_data = []      # Accumulate player data across segments

    def compute_speed_and_distance(self, track_data, frame_offset=0):
        # Persist the accumulated distance across segments:
        if not hasattr(self, 'accumulated_distance'):
            self.accumulated_distance = {}
        rolling_window = {}
        speed_distance_data = []

        for obj_type, obj_frames in track_data.items():
            if obj_type in ("ball", "referees"):
                continue

            total_frames = len(obj_frames)
            for start_idx in range(0, total_frames, self.interval):
                end_idx = min(start_idx + self.interval, total_frames - 1)

                for track_id, _ in obj_frames[start_idx].items():
                    # Process all players (no filtering here)
                    if track_id not in obj_frames[end_idx]:
                        continue

                    start_pos = obj_frames[start_idx][track_id].get('position_transformed')
                    end_pos = obj_frames[end_idx][track_id].get('position_transformed')
                    if not start_pos or not end_pos:
                        continue

                    dist = compute_euclidean_distance(start_pos, end_pos)
                    time_elapsed = (end_idx - start_idx) / self.fps
                    if time_elapsed == 0:
                        time_elapsed = 1e-6  # avoid division by zero

                    speed_mps = dist / time_elapsed

                    # Convert to km/h
                    raw_speed_kph = speed_mps * 3.6

                    # Initialize persistent accumulated distance if necessary:
                    if track_id not in self.accumulated_distance:
                        self.accumulated_distance[track_id] = 0
                    if track_id not in rolling_window:
                        rolling_window[track_id] = deque(maxlen=self.smoothing_window)

                    prev_speed_kph = rolling_window[track_id][-1] if rolling_window[track_id] else 0
                    acceleration_mps2 = (speed_mps - (prev_speed_kph / 3.6)) / time_elapsed

                    if acceleration_mps2 > self.max_acceleration:
                        speed_mps = (prev_speed_kph / 3.6) + (self.max_acceleration * time_elapsed)
                    elif acceleration_mps2 < -self.max_deceleration:
                        speed_mps = (prev_speed_kph / 3.6) - (self.max_deceleration * time_elapsed)

                    speed_kph = round(min(speed_mps * 3.6, self.max_speed), 3)

                    # Append to speed history
                    if track_id not in self.speed_history:
                        self.speed_history[track_id] = []
                    self.speed_history[track_id].append(speed_kph)

                    # Set baseline speed if we have enough history
                    if track_id not in self.player_baseline_speed and len(self.speed_history[track_id]) >= 100:
                        self.player_baseline_speed[track_id] = np.mean(self.speed_history[track_id][:100])
                        print(f"Player {track_id} baseline speed set to {self.player_baseline_speed[track_id]:.2f} km/h")

                    baseline_speed = self.player_baseline_speed.get(track_id, 6.0)

                    # Smoothing: use a rolling median filter
                    rolling_window[track_id].append(speed_kph)
                    median_speed = np.median(rolling_window[track_id])
                    smooth_speed = round((0.7 * median_speed) + (0.3 * speed_kph), 3)

                    # Update the persistent accumulated distance:
                    self.accumulated_distance[track_id] += dist
                    self.accumulated_distance[track_id] = round(self.accumulated_distance[track_id], 3)

                    fatigue_index = self.calculate_fatigue_index(list(rolling_window[track_id]), baseline_speed)

                    # For each frame in the current interval, store data with the frame_offset added.
                    for frame_num in range(start_idx, end_idx):
                        if track_id not in obj_frames[frame_num]:
                            continue
                        # Update track data with computed metrics.
                        obj_frames[frame_num][track_id]['speed'] = smooth_speed
                        obj_frames[frame_num][track_id]['distance'] = self.accumulated_distance[track_id]
                        obj_frames[frame_num][track_id]['fatigue_index'] = fatigue_index

                        speed_distance_data.append({
                            "Frame": frame_num + frame_offset,
                            "Player ID": track_id,
                            "Speed (km/h)": smooth_speed,
                            "Distance Covered (m)": self.accumulated_distance[track_id],
                            "Fatigue Index": fatigue_index
                        })
        # Accumulate data from this segment into a persistent list.
        self.player_data.extend(speed_distance_data)
        return speed_distance_data

    def overlay_speed_and_distance(self, frames, track_data):
        annotated_frames = []
        for idx, frame in enumerate(frames):
            for obj_type, obj_frames in track_data.items():
                if obj_type in ("ball", "referees"):
                    continue
                # Annotate every player on the pitch.
                for track_id, track_info in obj_frames[idx].items():
                    spd = track_info.get('speed')
                    dist = track_info.get('distance')
                    if spd is None or dist is None:
                        continue
                    spd_text = f"{spd:.3f} km/h"
                    dist_text = f"{dist:.3f} m"
                    bbox = track_info['bbox']
                    foot_pos = list(compute_foot_position(bbox))
                    foot_pos[1] += 40
                    foot_pos = tuple(map(int, foot_pos))
                    cv2.putText(frame, spd_text, foot_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, dist_text, (foot_pos[0], foot_pos[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            annotated_frames.append(frame)
        return annotated_frames

    def calculate_fatigue_index(self, speed_data, baseline_speed, window_size=10):
        if len(speed_data) < window_size:
            current_avg_speed = np.mean(speed_data)
        else:
            current_avg_speed = np.mean(speed_data[-window_size:])
        fatigue_index = round(baseline_speed - current_avg_speed, 3)
        return fatigue_index

    def save_final_csv(self, final_csv_path):
        if not self.player_data:
            print("No player data to save.")
            return
        df = pd.DataFrame(self.player_data)
        df.to_csv(final_csv_path, index=False)
        print(f"Final speed, distance, and fatigue CSV saved to: {final_csv_path}")

class SpeedDistanceEstimatorExtended(SpeedDistanceEstimator):
    def __init__(self):
        super().__init__()
        # Dictionaries to hold metrics for all players.
        self.accumulated_distance_all = {}
        self.all_speed_history = {}   # Similar to self.speed_history but for all players.
        self.all_player_data = []     # Will store CSV rows for all players.
    
    def compute_all_players_speed_distance(self, track_data, frame_offset=0):
        # Create a rolling window per player for smoothing.
        rolling_window = {}
        for obj_type, obj_frames in track_data.items():
            if obj_type in ("ball", "referees"):
                continue
            total_frames = len(obj_frames)
            for start_idx in range(0, total_frames, self.interval):
                end_idx = min(start_idx + self.interval, total_frames - 1)
                for track_id, _ in obj_frames[start_idx].items():
                    if track_id not in obj_frames[end_idx]:
                        continue
                    start_pos = obj_frames[start_idx][track_id].get('position_transformed')
                    end_pos = obj_frames[end_idx][track_id].get('position_transformed')
                    if not start_pos or not end_pos:
                        continue

                    dist = compute_euclidean_distance(start_pos, end_pos)
                    time_elapsed = (end_idx - start_idx) / self.fps
                    if time_elapsed == 0:
                        time_elapsed = 1e-6
                    speed_mps = dist / time_elapsed

                    if track_id not in rolling_window:
                        rolling_window[track_id] = deque(maxlen=self.smoothing_window)
                    prev_speed = rolling_window[track_id][-1] if rolling_window[track_id] else 0
                    acceleration = (speed_mps - (prev_speed / 3.6)) / time_elapsed
                    if acceleration > self.max_acceleration:
                        speed_mps = (prev_speed / 3.6) + (self.max_acceleration * time_elapsed)
                    elif acceleration < -self.max_deceleration:
                        speed_mps = (prev_speed / 3.6) - (self.max_deceleration * time_elapsed)
                    speed_kph = round(min(speed_mps * 3.6, self.max_speed), 3)

                    rolling_window[track_id].append(speed_kph)
                    median_speed = np.median(rolling_window[track_id])
                    smooth_speed = round((0.7 * median_speed) + (0.3 * speed_kph), 3)

                    if track_id not in self.accumulated_distance_all:
                        self.accumulated_distance_all[track_id] = 0
                    self.accumulated_distance_all[track_id] += dist
                    self.accumulated_distance_all[track_id] = round(self.accumulated_distance_all[track_id], 3)

                    baseline_speed = self.player_baseline_speed.get(track_id, 6.0)
                    fatigue_index = self.calculate_fatigue_index(list(rolling_window[track_id]), baseline_speed)

                    team_label = obj_frames[end_idx][track_id].get('team', 'Unknown')

                    for frame_num in range(start_idx, end_idx):
                        if track_id not in obj_frames[frame_num]:
                            continue
                        obj_frames[frame_num][track_id]['speed'] = smooth_speed
                        obj_frames[frame_num][track_id]['distance'] = self.accumulated_distance_all[track_id]
                        obj_frames[frame_num][track_id]['fatigue_index'] = fatigue_index
                        obj_frames[frame_num][track_id]['team'] = team_label

                        self.all_player_data.append({
                            "Frame": frame_num + frame_offset,
                            "Player ID": track_id,
                            "Team": team_label,
                            "Speed (km/h)": smooth_speed,
                            "Distance Covered (m)": self.accumulated_distance_all[track_id],
                            "Fatigue Index": fatigue_index
                        })
        return self.all_player_data

    def save_team_csv(self, csv_path):
        if not self.all_player_data:
            print("No team data to save.")
            return
        df = pd.DataFrame(self.all_player_data)
        df.to_csv(csv_path, index=False)
        print(f"Team performance CSV saved to: {csv_path}")

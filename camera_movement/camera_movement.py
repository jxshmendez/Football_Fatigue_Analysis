import pickle
import cv2
import numpy as np
import os
import sys

sys.path.append('../')
from utilities import compute_euclidean_distance, compute_xy_displacement

class CameraMovement:
    def __init__(self, initial_frame):
        # minimum movement threshold
        self.min_dist = 5

        # settings for Lucas-Kanade optical flow
        self.optical_flow_settings = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # prepare first frame for feature detection
        first_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        track_mask = np.zeros_like(first_gray)
        track_mask[:, 0:20] = 1
        track_mask[:, 900:1050] = 1

        # feature parameters for goodFeaturesToTrack
        self.track_features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=track_mask
        )

    def apply_camera_shift(self, track_data, motion_per_frame):
        # adjust object positions based on the camera's motion
        for obj_type, frames_list in track_data.items():
            for idx, frame_tracks in enumerate(frames_list):
                for track_id, track_info in frame_tracks.items():
                    original_pos = track_info['position']
                    shift_vals = motion_per_frame[idx]
                    # subtract camera movement from object position
                    adjusted = (original_pos[0] - shift_vals[0], original_pos[1] - shift_vals[1])
                    track_data[obj_type][idx][track_id]['position_adjusted'] = adjusted

    def calculate_camera_motion(self, input_frames, use_cache=False, cache_path=None):
        # optionally load precomputed motion from file
        if use_cache and cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # initialize motion array
        camera_shifts = [[0, 0]] * len(input_frames)

        # Convert first frame to grayscale and detect features
        prev_gray = cv2.cvtColor(input_frames[0], cv2.COLOR_BGR2GRAY)
        prev_features = cv2.goodFeaturesToTrack(prev_gray, **self.track_features)

        # Ensure correct format
        if prev_features is not None:
            prev_features = np.array(prev_features, dtype=np.float32).reshape(-1, 1, 2)

        # Loop through subsequent frames
        for frame_idx in range(1, len(input_frames)):
            gray_frame = cv2.cvtColor(input_frames[frame_idx], cv2.COLOR_BGR2GRAY)

            if prev_features is None or len(prev_features) == 0:
                print(f"No features found in frame {frame_idx - 1}, skipping.")
                prev_features = cv2.goodFeaturesToTrack(gray_frame, **self.track_features)
                if prev_features is not None:
                    prev_features = np.array(prev_features, dtype=np.float32).reshape(-1, 1, 2)
                prev_gray = gray_frame.copy()
                continue  # Skip this frame if no features are found

            curr_features, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray_frame, prev_features, None, **self.optical_flow_settings
            )

            if curr_features is None or len(curr_features) == 0:
                print(f"Optical flow failed at frame {frame_idx}, skipping.")
                prev_gray = gray_frame.copy()
                continue  # Skip processing if tracking failed

            max_disp = 0
            shift_x, shift_y = 0, 0

            # Find the largest displacement among features
            for i, (old_pt, new_pt) in enumerate(zip(prev_features, curr_features)):
                if status[i] == 1:  # Only consider valid tracked points
                    old_coords = old_pt.ravel()
                    new_coords = new_pt.ravel()
                    dist = compute_euclidean_distance(new_coords, old_coords)
                    if dist > max_disp:
                        max_disp = dist
                        shift_x, shift_y = compute_xy_displacement(old_coords, new_coords)

            if max_disp > self.min_dist:
                camera_shifts[frame_idx] = [shift_x, shift_y]
                prev_features = cv2.goodFeaturesToTrack(gray_frame, **self.track_features)
                if prev_features is not None:
                    prev_features = np.array(prev_features, dtype=np.float32).reshape(-1, 1, 2)

            prev_gray = gray_frame.copy()


        # optionally save motion to cache
        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(camera_shifts, f)

        return camera_shifts

    def overlay_camera_motion(self, frames, motion_per_frame):
        # create annotated frames
        result_frames = []

        for idx, img in enumerate(frames):
            frame_copy = img.copy()

            # draw a rectangle for the motion info
            overlay = frame_copy.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)

            # show camera shift
            shift_x, shift_y = motion_per_frame[idx]
            cv2.putText(frame_copy, f"Camera Movement X: {shift_x:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame_copy, f"Camera Movement Y: {shift_y:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            result_frames.append(frame_copy)

        return result_frames

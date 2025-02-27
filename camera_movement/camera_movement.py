import torch
import numpy as np
import cv2
import os
import pickle

from utilities import compute_euclidean_distance, compute_xy_displacement

class CameraMovement:
    def __init__(self, initial_frame):
        # Convert tensor to NumPy if necessary
        if isinstance(initial_frame, torch.Tensor):
            initial_frame = initial_frame.detach().cpu().numpy()

        if initial_frame is None:
            raise ValueError("Error: initial_frame is None. Check if the video is loading correctly.")

        # Ensure it's a valid NumPy array
        if not isinstance(initial_frame, np.ndarray):
            raise TypeError(f"Expected initial_frame to be a NumPy array, but got {type(initial_frame)} instead.")

        # Ensure correct shape (should be HxWx3)
        if len(initial_frame.shape) == 3 and initial_frame.shape[0] < 10:  # Likely in (C, H, W) format
            print("Warning: Frame is in (C, H, W) format, transposing to (H, W, C)")
            initial_frame = np.transpose(initial_frame, (1, 2, 0))

        # Convert frame to uint8 (if needed)
        if initial_frame.dtype != np.uint8:
            print("Warning: Frame dtype is not uint8, converting")
            initial_frame = (initial_frame * 255).astype(np.uint8)

        print(f"Frame Type: {type(initial_frame)}")
        print(f"Frame Shape: {initial_frame.shape}")

        # Set up optical flow settings for Lucas-Kanade method
        self.optical_flow_settings = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        # Minimum movement threshold
        self.min_dist = 5

        # Prepare first frame for feature detection by converting to grayscale
        first_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        # Create a tracking mask (here using fixed margins)
        track_mask = np.zeros_like(first_gray)
        track_mask[:, :20] = 1
        track_mask[:, -150:] = 1

        # Feature parameters for goodFeaturesToTrack
        self.track_features = {
            "maxCorners": 100,
            "qualityLevel": 0.3,
            "minDistance": 3,
            "blockSize": 7,
            "mask": track_mask
        }

    def apply_camera_shift(self, track_data, motion_per_frame):
        # Adjust object positions based on the camera's motion
        for obj_type, frames_list in track_data.items():
            for idx, frame_tracks in enumerate(frames_list):
                for track_id, track_info in frame_tracks.items():
                    original_pos = track_info['position']
                    shift_vals = motion_per_frame[idx]
                    adjusted = (original_pos[0] - shift_vals[0], original_pos[1] - shift_vals[1])
                    track_data[obj_type][idx][track_id]['position_adjusted'] = adjusted

    def calculate_camera_motion(self, input_frames, use_cache=False, cache_path=None):
        # Optionally load precomputed motion from cache
        if use_cache and cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # Initialize motion array
        camera_shifts = [[0, 0]] * len(input_frames)

        # Process the first frame
        frame0 = input_frames[0]
        if isinstance(frame0, torch.Tensor):
            frame0 = frame0.cpu().numpy()
        frame0 = np.ascontiguousarray(frame0)
        prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        prev_features = cv2.goodFeaturesToTrack(prev_gray, **self.track_features)
        if prev_features is not None:
            prev_features = np.array(prev_features, dtype=np.float32).reshape(-1, 1, 2)

        # Loop through subsequent frames
        for frame_idx in range(1, len(input_frames)):
            frame = input_frames[frame_idx]
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            frame = np.ascontiguousarray(frame)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_features is None or len(prev_features) == 0:
                print(f"No features found in frame {frame_idx - 1}, skipping.")
                prev_features = cv2.goodFeaturesToTrack(gray_frame, **self.track_features)
                if prev_features is not None:
                    prev_features = np.array(prev_features, dtype=np.float32).reshape(-1, 1, 2)
                prev_gray = gray_frame.copy()
                continue

            curr_features, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray_frame, prev_features, None, **self.optical_flow_settings
            )

            if curr_features is None or len(curr_features) == 0:
                print(f"Optical flow failed at frame {frame_idx}, skipping.")
                prev_gray = gray_frame.copy()
                continue

            max_disp = 0
            shift_x, shift_y = 0, 0

            # Determine the largest displacement among tracked features
            for i, (old_pt, new_pt) in enumerate(zip(prev_features, curr_features)):
                if status[i] == 1:
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

        # Optionally save motion data to cache
        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(camera_shifts, f)

        return camera_shifts

    def overlay_camera_motion(self, frames, motion_per_frame):
        # Create annotated frames with overlaid camera motion information
        result_frames = []
        for idx, img in enumerate(frames):
            frame_copy = img.copy()
            overlay = frame_copy.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)
            shift_x, shift_y = motion_per_frame[idx]
            cv2.putText(frame_copy, f"Camera Movement X: {shift_x:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame_copy, f"Camera Movement Y: {shift_y:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            result_frames.append(frame_copy)
        return result_frames

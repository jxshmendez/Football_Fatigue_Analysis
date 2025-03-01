from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
import torch
sys.path.append('../')
from utilities import compute_bbox_center, compute_bbox_width, compute_foot_position

class ObjectTracker:
    def __init__(self, model_file):
        # Initialize YOLO model for object detection
        self.detector = YOLO(model_file)
        # Use ByteTrack for tracking multiple objects
        self.motion_tracker = sv.ByteTrack()
        self.player_id_map = {}  # {tracking_id: custom_player_id}
        self.next_available_id = 1  # Start assigning IDs from 1
        self.tracking_history = {}
        
    def get_persistent_id(self, tracking_id, bbox):
        """
        Assigns a persistent ID to players by checking previous IDs with similar locations.
        """
        if tracking_id in self.player_id_map:
            return self.player_id_map[tracking_id]

        # Try to find an existing player ID with a similar position
        for prev_tracking_id, player_id in self.player_id_map.items():
            prev_bbox = self.tracking_history.get(prev_tracking_id, None)
            if prev_bbox and self.is_similar_bbox(prev_bbox, bbox):
                self.player_id_map[tracking_id] = player_id
                return player_id

        # If no match is found, assign a new ID
        self.player_id_map[tracking_id] = self.next_available_id
        self.tracking_history[tracking_id] = bbox  # Save bounding box for tracking
        self.next_available_id += 1

        return self.player_id_map[tracking_id]

    def is_similar_bbox(self, bbox1, bbox2, threshold=30):
        """
        Check if two bounding boxes are close enough to be the same player.
        """
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2

        # Check center distance
        center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
        center2 = ((x1_ + x2_) // 2, (y1_ + y2_) // 2)
        dist = np.linalg.norm(np.array(center1) - np.array(center2))
        
        return dist < threshold

    def assign_positions(self, tracked_data):
        # Assigns either center or foot position to detected objects
        for category, frame_data in tracked_data.items():
            for frame_idx, objects in enumerate(frame_data):
                for obj_id, obj_details in objects.items():
                    bounding_box = obj_details['bbox']
                    
                    if category == 'ball':
                        position = compute_bbox_center(bounding_box)
                    else:
                        position = compute_foot_position(bounding_box)

                    tracked_data[category][frame_idx][obj_id]['position'] = position

    def smooth_ball_trajectory(self, ball_tracks):
        # Extracts bounding boxes of the ball from frames
        ball_bboxes = [frame.get(1, {}).get('bbox', []) for frame in ball_tracks]
        df_ball_positions = pd.DataFrame(ball_bboxes, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions.interpolate(inplace=True)
        df_ball_positions.bfill(inplace=True)
        return [{1: {"bbox": bbox}} for bbox in df_ball_positions.to_numpy().tolist()]
    
    def batch_inference(self, input_frames):
        # Processes frames in batches using the model
        chunk_size = 20
        predictions = []

        for start_idx in range(0, len(input_frames), chunk_size):
            batch = input_frames[start_idx : start_idx + chunk_size]
            results = self.detector.predict([frame.cpu().numpy() for frame in batch], device="mps", conf=0.05)
            predictions.extend(results)
        return predictions

    def retrieve_object_tracks(self, input_frames, use_stub=False, cache_path=None):
        if use_stub and cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as saved_file:
                preloaded_tracks = pickle.load(saved_file)
            return preloaded_tracks

        detections = self.batch_inference(input_frames)  # Run detection

        object_tracks = {"players": [], "referees": [], "ball": []}

        for idx, detection_result in enumerate(detections):
            class_labels = detection_result.names
            inverse_labels = {label: i for i, label in class_labels.items()}

            sv_detections = sv.Detections.from_ultralytics(detection_result)
            updated_tracks = self.motion_tracker.update_with_detections(sv_detections)

            object_tracks["players"].append({})
            object_tracks["referees"].append({})
            object_tracks["ball"].append({})

            for obj_data in updated_tracks:
                bbox_list = obj_data[0].tolist()
                label_id = obj_data[3]
                tracking_id = obj_data[4]  # Temporary tracking ID
                
                custom_id = self.get_persistent_id(tracking_id, bbox_list)
                self.tracking_history[custom_id] = bbox_list

                if label_id == inverse_labels["player"]:
                    object_tracks["players"][idx][custom_id] = {"bbox": bbox_list}
                elif label_id == inverse_labels["referee"]:
                    object_tracks["referees"][idx][custom_id] = {"bbox": bbox_list}

            # Process ball detection separately
            for detection_item in sv_detections:
                bbox_list = detection_item[0].tolist()
                class_id = detection_item[3]
                if class_id == inverse_labels["ball"]:
                    object_tracks["ball"][idx][1] = {"bbox": bbox_list}
                    
            if cache_path:
                with open(cache_path, 'wb') as out_file:
                    pickle.dump(object_tracks, out_file)

        # Save Player ID Mapping after processing
        id_mapping_df = pd.DataFrame(list(self.player_id_map.items()), columns=["Tracker ID", "Custom Player ID"])
        id_mapping_df.to_csv("outputVid/player_id_mapping.csv", index=False)
        
        return object_tracks

    def draw_ellipse_on_frame(self, frame, box, color, track_id=None):
        # Get the bottom y-coordinate and the midpoint x-coordinate
        y_bottom = int(box[3])
        x_mid, _ = compute_bbox_center(box)
        x_mid = int(x_mid)
        box_w = int(compute_bbox_width(box))

        # Draw the ellipse at the bottom center of the bounding box
        cv2.ellipse(
            frame,
            center=(x_mid, y_bottom),
            axes=(box_w, int(0.35 * box_w)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Optionally, add a label for the track ID
        if track_id is not None:
            rect_w = 40
            rect_h = 20
            rx1 = x_mid - rect_w // 2
            rx2 = x_mid + rect_w // 2
            ry1 = y_bottom - rect_h // 2 + 15
            ry2 = y_bottom + rect_h // 2 + 15

            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, cv2.FILLED)
            text_x = rx1 + 12
            if track_id > 99:
                text_x -= 10
            cv2.putText(
                frame,
                f"{track_id}",
                (text_x, ry1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def annotate_video_frames(self, frames, all_tracks):
        """
        Annotates only player tracks on the given frames.
        Referees and ball annotations are omitted.
        """
        annotated_frames = []

        for idx, original_frame in enumerate(frames):
            if isinstance(original_frame, torch.Tensor):
                original_frame = original_frame.cpu().numpy()
                original_frame = np.ascontiguousarray(original_frame)
            
            frame_copy = original_frame.copy()

            if "players" not in all_tracks or len(all_tracks["players"]) <= idx:
                print(f"WARNING: No player data found for frame {idx}, returning original frames.")
                annotated_frames.append(frame_copy)
                continue  # Skip this frame, but keep original

            players_here = all_tracks["players"][idx]

            for track_id, player_data in players_here.items():
                custom_id = track_id  # Use persistent ID
                team_color = player_data.get("team_color", (0, 0, 255))
                frame_copy = self.draw_ellipse_on_frame(frame_copy, player_data["bbox"], team_color, custom_id)

            annotated_frames.append(frame_copy)

        if not annotated_frames:
            print("ERROR: No frames were annotated. Returning original frames.")
            return frames  # Return original frames instead of None

        return annotated_frames

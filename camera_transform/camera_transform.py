import cv2
import numpy as np

class CameraTransform:
    def __init__(self):
        # set real-world dimensions (in meters) for the court
        pitch_width = 68
        pitch_length = 23.32

        # define source points (pixel coordinates)
        self.src_points = np.array([
            [100, 890],
            [400, 100],
            [1700, 170],
            [1700, 1080]
        ], dtype=np.float32)

        # define target points (real-world coordinates)
        self.dst_points = np.array([
            [0, pitch_width],
            [0, 0],
            [pitch_length, 0],
            [pitch_length, pitch_width]
        ], dtype=np.float32)

        # compute perspective transform matrix
        self.transform_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def warp_point(self, point):
        # checks if the point is inside the polygon
        pt = (int(point[0]), int(point[1]))
        inside_poly = cv2.pointPolygonTest(self.src_points, pt, False) >= 0
        if not inside_poly:
            return None

        # warp the point using the perspective transform
        reshaped = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(reshaped, self.transform_matrix)
        return warped.reshape(-1, 2)

    def apply_transform_to_tracks(self, tracks):
        # loop through each object type
        for obj_type, obj_tracks in tracks.items():
            # loop through frames
            for frame_idx, frame_data in enumerate(obj_tracks):
                # loop through each track in the current frame
                for track_id, info in frame_data.items():
                    adjusted_pos = info['position_adjusted']
                    adjusted_pos = np.array(adjusted_pos)

                    transformed_pt = self.warp_point(adjusted_pos)
                    if transformed_pt is not None:
                        transformed_pt = transformed_pt.squeeze().tolist()

                    # store the transformed position
                    tracks[obj_type][frame_idx][track_id]['position_transformed'] = transformed_pt

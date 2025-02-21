import cv2
import numpy as np

def compute_color_histogram(frame, bbox):
    # Crop the player's region from the frame
    x1, y1, x2, y2 = bbox
    crop = frame[int(y1):int(y2), int(x1):int(x2)]
    # Calculate a normalized histogram in HSV space
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def merge_player_tracks(player_tracks, frames, max_distance=50, hist_threshold=0.5):
    """
    Merge tracks across frames if the new track's position and appearance 
    are similar to a track from a previous frame.
    
    Parameters:
        player_tracks (list): List of dictionaries for each frame.
        frames (list): List of frames from the video (to extract appearance).
        max_distance (float): Maximum distance for spatial merging.
        hist_threshold (float): Minimum histogram correlation for appearance merging.
    """
    for frame_idx in range(1, len(player_tracks)):
        current_frame_tracks = player_tracks[frame_idx]
        previous_frame_tracks = player_tracks[frame_idx - 1]
        
        merge_mapping = {}
        for new_id, new_info in current_frame_tracks.items():
            new_pos = new_info.get('position')
            if new_pos is None:
                continue
            new_pos = np.array(new_pos)
            
            # Optionally compute appearance for new track
            new_bbox = new_info.get('bbox')
            new_hist = None
            if new_bbox is not None:
                new_hist = compute_color_histogram(frames[frame_idx], new_bbox)
            
            for old_id, old_info in previous_frame_tracks.items():
                old_pos = old_info.get('position')
                if old_pos is None:
                    continue
                old_pos = np.array(old_pos)
                dist = np.linalg.norm(new_pos - old_pos)
                if dist < max_distance:
                    # Optionally compare appearance histograms if available
                    if new_hist is not None and 'histogram' in old_info:
                        # Use correlation or another similarity metric
                        correlation = cv2.compareHist(new_hist.astype('float32'), 
                                                    old_info['histogram'].astype('float32'), 
                                                    cv2.HISTCMP_CORREL)
                        if correlation < hist_threshold:
                            continue  # Not similar enough
                    merge_mapping[new_id] = old_id
                    # Merge metadata: speed history, histogram, etc.
                    if 'speed_history' in new_info:
                        if 'speed_history' in old_info:
                            old_info['speed_history'].extend(new_info['speed_history'])
                        else:
                            old_info['speed_history'] = new_info['speed_history']
                    if new_hist is not None:
                        old_info['histogram'] = new_hist
                    break
        
        # Reassign merged tracks
        for new_id, old_id in merge_mapping.items():
            if new_id != old_id:
                current_frame_tracks[old_id] = current_frame_tracks[new_id]
                del current_frame_tracks[new_id]
    return player_tracks

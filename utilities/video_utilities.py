import cv2
import numpy as np

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame is None:
            print("Warning: A frame was not loaded correctly, skipping.")
            continue

        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame is not a valid NumPy array: {type(frame)}")

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len (frame.shape) == 3 and frame.shape[0] < 10:  # If channel-first format (C, H, W)
            frame = np.transpose(frame, (1, 2, 0))  # Convert to (H, W, C)
        frames.append(frame)

    cap.release()
    
    if len(frames) == 0:
        raise ValueError("Error: No frames were loaded from the video.")
    
    return frames

def save_video(outputVidFrame, outputVidPath):
    """Saves a list of frames as a video file."""
    if not outputVidFrame:  # Ensure frames exist
        print("Error: No frames to save.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    frame_size = (outputVidFrame[0].shape[1], outputVidFrame[0].shape[0])

    output = cv2.VideoWriter(outputVidPath, fourcc, 25, frame_size)

    for frame in outputVidFrame:
        output.write(frame)

    output.release()

import cv2

def load_video(vid_path):
    """Reads a video file and returns a list of frames."""
    capture = cv2.VideoCapture(vid_path)
    frames = []

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)

    capture.release()  # Always release the video capture object
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

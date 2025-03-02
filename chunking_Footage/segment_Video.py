import subprocess

def segment_video(input_path, segment_length=30):
    cmd = [
        "ffmpeg", "-i", input_path,
        "-c:v", "libx264", "-preset", "veryfast",
        "-force_key_frames", f"expr:gte(t,n_forced*{segment_length})",
        "-c:a", "copy",
        "-reset_timestamps", "1",
        "-f", "segment",
        "-segment_time", str(segment_length),
        "inputVid/segments/segment_%03d.mp4"
    ]
    subprocess.run(cmd, check=True)

# Example usage:
segment_video("inputVid/1min30.mp4", segment_length=30)

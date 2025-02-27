import glob
import subprocess
import os
import sys

# Get the absolute path to the directory containing main.py
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if main_dir not in sys.path:
    sys.path.append(main_dir)

from main import process_footage

processed_dir = "outputVid/processed"
os.makedirs(processed_dir, exist_ok=True)

segment_files = sorted(glob.glob("outputVid/segment_*.mp4"))
processed_segments = []

for i, seg in enumerate(segment_files):
    processed_path = seg.replace("outputVid", processed_dir)
    processed_segments.append(processed_path)
    
    print(f"Processing segment: {seg}")
    # For example, disable stub caching for all segments:
    process_footage(seg, processed_path, use_stub=False)
    # Or, if you want to use caching only for the first segment:
    # if i == 0:
    #     process_footage(seg, processed_path, use_stub=True)
    # else:
    #     process_footage(seg, processed_path, use_stub=False)
    print(f"Segment processed and saved to: {processed_path}")

print("All segments processed.")

# Create a file list for merging using ffmpeg:
with open("outputVid/filelist.txt", "w") as f:
    for segment in processed_segments:
        abs_path = os.path.abspath(segment)
        f.write(f"file '{abs_path}'\n")

print("File list for merging created.")

subprocess.run([
    "ffmpeg", "-f", "concat", "-safe", "0", "-i", "outputVid/filelist.txt", "-c", "copy", "outputVid/final_output.mp4"
])

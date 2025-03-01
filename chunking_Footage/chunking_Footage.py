import glob
import subprocess
import os
import sys

# Insert the parent directory into sys.path at index 0
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
#print("sys.path:", sys.path)  # Uncomment for debugging

from PlayerPerformanceVisualiser.team_Performance_Visualiser import team_Performance_Visualiser
from PlayerPerformanceVisualiser.playerPerformanceVisualiser import PlayerPerformanceVisualiser
from utilities import load_video, save_video  # make sure load_video is defined there
from speedDistance_estimator import SpeedDistanceEstimator, SpeedDistanceEstimatorExtended
from main import process_footage

def merge_videos(video_files, output_path):
    file_list_path = "temp_filelist.txt"
    with open(file_list_path, "w") as f:
        for video in video_files:
            abs_path = os.path.abspath(video)
            f.write(f"file '{abs_path}'\n")
    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", file_list_path,
        "-c", "copy", output_path
    ], check=True)
    os.remove(file_list_path)

# Get list of segment files (adjust the glob pattern if needed)
segment_files = sorted(glob.glob(os.path.join("inputVid", "segments", "segment_*.mp4")))
print(f"Found {len(segment_files)} segment files.")

# Folder to store processed segments
processed_folder = os.path.join("outputVid", "processed")
os.makedirs(processed_folder, exist_ok=True)

# Create instances for both single-player and team-based calculations
single_player_calc = SpeedDistanceEstimator()
team_calc = SpeedDistanceEstimatorExtended()

cumulative_offset = 0
processed_videos = []

for seg in segment_files:
    seg_name = os.path.basename(seg)
    out_seg = os.path.join(processed_folder, seg_name)
    
    # Get number of frames in this segment
    frames = list(load_video(seg))
    num_frames = len(frames)
    print(f"Segment {seg_name} has {num_frames} frames, cumulative offset: {cumulative_offset}")
    
    # Process the segment with both calculators.
    # (Inside process_footage, make sure team_calc.compute_all_players_speed_distance is called after team classification.)
    process_footage(seg, out_seg, single_player_calc, team_calc, use_stub=False, frame_offset=cumulative_offset)
    
    # Optional: Debug print after processing each segment.
    print(f"After segment {seg_name}, team data count: {len(team_calc.all_player_data)}")
    
    cumulative_offset += num_frames
    processed_videos.append(out_seg)

print("All segments processed.")

# Merge processed segments into one final video.
final_video_path = os.path.join("outputVid", "final_output.mp4")
merge_videos(processed_videos, final_video_path)
print(f"Final video merged and saved to: {final_video_path}")

# (Optional) Re-save team CSV for visualization (if not already updated in process_footage)
team_csv_path = os.path.join("outputVid", "team_performance.csv")
team_calc.save_team_csv(team_csv_path)
print(f"Team CSV saved to: {team_csv_path}")

# Visualize team performance
team_visualizer = team_Performance_Visualiser(team_csv_path)

team_visualizer.plot_fatigue_comparison(save=True, show=True)
team_visualizer.plot_detailed_team_summary(save=True, show=True)
team_visualizer.compare_team_fatigue(save=True, show=True)

team_visualizer.save_team_summary()
team_visualizer.save_detailed_team_summary()
print("Team performance plots and summary generated.")

# Save final CSV for single-player metrics (data accumulated over segments)
final_csv_path = os.path.join("outputVid", "player_5_speed_distance_final.csv")
single_player_calc.save_final_csv(final_csv_path)
print(f"Final CSV saved to: {final_csv_path}")

# Visualize single-player performance
visualizer = PlayerPerformanceVisualiser(final_csv_path)
visualizer.plot_all()
print("Player performance plots generated.")

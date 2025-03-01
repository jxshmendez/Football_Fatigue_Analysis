import os
import pandas as pd
import matplotlib.pyplot as plt

class PlayerPerformanceVisualiser:
    def __init__(self, file_path, output_dir="outputVid"):
        """Initialize with the CSV file path and output directory."""
        self.file_path = file_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure output directory exists

        try:
            self.df = pd.read_csv(file_path)
            if self.df.empty:
                print(f"Warning: The CSV file {file_path} is empty.")
        except Exception as e:
            print(f"Error reading CSV file {file_path}: {e}")
            self.df = pd.DataFrame()

    def plot_speed(self, save=True):
        """Plot Speed Over Time and Save if Required."""
        if self.df.empty:
            print("No data to plot for speed.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.df["Frame"], self.df["Speed (km/h)"], label="Speed (km/h)", color="blue")
        plt.xlabel("Frame")
        plt.ylabel("Speed (km/h)")
        plt.title("Player 5 Speed Over Time")
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(os.path.join(self.output_dir, "speed_plot.png"))
        plt.close()

    def plot_distance(self, save=True):
        """Plot Distance Covered Over Time and Save if Required."""
        if self.df.empty:
            print("No data to plot for distance.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.df["Frame"], self.df["Distance Covered (m)"], label="Distance Covered (m)", color="green")
        plt.xlabel("Frame")
        plt.ylabel("Distance Covered (m)")
        plt.title("Player 5 Distance Covered Over Time")
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(os.path.join(self.output_dir, "distance_plot.png"))
        plt.close()

    def plot_fatigue_index(self, save=True):
        """Plot Fatigue Index Over Time and Save if Required."""
        if self.df.empty:
            print("No data to plot for fatigue index.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.df["Frame"], self.df["Fatigue Index"], label="Fatigue Index", color="red")
        plt.xlabel("Frame")
        plt.ylabel("Fatigue Index")
        plt.title("Player 5 Fatigue Index Over Time")
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(os.path.join(self.output_dir, "fatigue_plot.png"))
        plt.close()

    def plot_all(self, save=True):
        """Plot all metrics in one figure and Save if Required."""
        if self.df.empty:
            print("No data to plot for all metrics.")
            return

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # Speed plot
        axs[0].plot(self.df["Frame"], self.df["Speed (km/h)"], label="Speed (km/h)", color="blue")
        axs[0].set_xlabel("Frame")
        axs[0].set_ylabel("Speed (km/h)")
        axs[0].set_title("Player 5 Speed Over Time")
        axs[0].legend()
        axs[0].grid(True)

        # Distance plot
        axs[1].plot(self.df["Frame"], self.df["Distance Covered (m)"], label="Distance Covered (m)", color="green")
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Distance Covered (m)")
        axs[1].set_title("Player 5 Distance Covered Over Time")
        axs[1].legend()
        axs[1].grid(True)

        # Fatigue Index plot
        axs[2].plot(self.df["Frame"], self.df["Fatigue Index"], label="Fatigue Index", color="red")
        axs[2].set_xlabel("Frame")
        axs[2].set_ylabel("Fatigue Index")
        axs[2].set_title("Player 5 Fatigue Index Over Time")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.output_dir, "all_plots.png"))
        plt.close()

    def save_summary(self):
        """Save key metrics (Top Speed, Total Distance, Avg Fatigue) to a CSV file."""
        if self.df.empty:
            print("No data to save summary.")
            return

        top_speed = self.df["Speed (km/h)"].max()
        total_distance = self.df["Distance Covered (m)"].max()
        avg_fatigue = self.df["Fatigue Index"].mean()

        summary_data = {
            "Top Speed (km/h)": [round(top_speed, 2)],
            "Total Distance (m)": [round(total_distance, 2)],
            "Average Fatigue Index": [round(avg_fatigue, 2)]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(self.output_dir, "player_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Player summary saved to: {summary_csv_path}")

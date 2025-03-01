import os
import pandas as pd
import matplotlib.pyplot as plt

class team_Performance_Visualiser:
    def __init__(self, file_path, output_dir="outputVid"):
        self.file_path = file_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            self.df = pd.read_csv(file_path)
            if self.df.empty:
                print(f"Warning: The CSV file {file_path} is empty.")
        except Exception as e:
            print(f"Error reading CSV file {file_path}: {e}")
            self.df = pd.DataFrame()

    def plot_fatigue_comparison(self, save=True, show=True):
        if self.df.empty:
            print("No team data to plot for fatigue index.")
            return

        # Group by frame and team to compute average fatigue index per team per frame.
        grouped = self.df.groupby(["Frame", "Team"]).mean().reset_index()
        teams = grouped["Team"].unique()

        plt.figure(figsize=(10, 5))
        for team in teams:
            team_data = grouped[grouped["Team"] == team]
            plt.plot(team_data["Frame"], team_data["Fatigue Index"], label=f"Team {team}")
        
        plt.xlabel("Frame")
        plt.ylabel("Average Fatigue Index")
        plt.title("Team Fatigue Index Comparison Over Time")
        plt.legend()
        plt.grid(True)

        # Define fatigue zones (example thresholds; adjust as needed)
        plt.axhspan(0, 1, color='green', alpha=0.2, label='Low Fatigue Zone')
        plt.axhspan(1, 3, color='yellow', alpha=0.2, label='Moderate Fatigue Zone')
        ymax = self.df["Fatigue Index"].max() + 1
        plt.axhspan(3, ymax, color='red', alpha=0.2, label='High Fatigue Zone')
        
        if save:
            save_path = os.path.join(self.output_dir, "team_fatigue_comparison.png")
            plt.savefig(save_path)
            print(f"Team fatigue comparison chart saved to: {save_path}")
        if show:
            plt.show()
        plt.close()

    def save_team_summary(self):
        if self.df.empty:
            print("No team data to save summary.")
            return
        summary = self.df.groupby("Team").agg({
            "Speed (km/h)": "mean",
            "Distance Covered (m)": "max",
            "Fatigue Index": "mean"
        }).reset_index()
        summary_csv_path = os.path.join(self.output_dir, "team_summary.csv")
        summary.to_csv(summary_csv_path, index=False)
        print(f"Team summary saved to: {summary_csv_path}")
        
    def save_detailed_team_summary(self):
        """
        Save a detailed summary per team that includes the minimum and maximum values for
        Speed (km/h), Distance Covered (m), and Fatigue Index along with the frame numbers
        at which these values occurred.
        """
        if self.df.empty:
            print("No team data to save detailed summary.")
            return
        teams = self.df["Team"].unique()
        summary_list = []
        for team in teams:
            team_df = self.df[self.df["Team"] == team]
            
            # For Speed
            min_speed = team_df["Speed (km/h)"].min()
            max_speed = team_df["Speed (km/h)"].max()
            min_speed_frame = team_df.loc[team_df["Speed (km/h)"].idxmin(), "Frame"]
            max_speed_frame = team_df.loc[team_df["Speed (km/h)"].idxmax(), "Frame"]
            
            # For Distance Covered
            min_distance = team_df["Distance Covered (m)"].min()
            max_distance = team_df["Distance Covered (m)"].max()
            min_distance_frame = team_df.loc[team_df["Distance Covered (m)"].idxmin(), "Frame"]
            max_distance_frame = team_df.loc[team_df["Distance Covered (m)"].idxmax(), "Frame"]
            
            # For Fatigue Index
            min_fatigue = team_df["Fatigue Index"].min()
            max_fatigue = team_df["Fatigue Index"].max()
            min_fatigue_frame = team_df.loc[team_df["Fatigue Index"].idxmin(), "Frame"]
            max_fatigue_frame = team_df.loc[team_df["Fatigue Index"].idxmax(), "Frame"]
            
            summary_list.append({
                "Team": team,
                "Min Speed (km/h)": min_speed,
                "Min Speed Frame": min_speed_frame,
                "Max Speed (km/h)": max_speed,
                "Max Speed Frame": max_speed_frame,
                "Min Distance (m)": min_distance,
                "Min Distance Frame": min_distance_frame,
                "Max Distance (m)": max_distance,
                "Max Distance Frame": max_distance_frame,
                "Min Fatigue": min_fatigue,
                "Min Fatigue Frame": min_fatigue_frame,
                "Max Fatigue": max_fatigue,
                "Max Fatigue Frame": max_fatigue_frame
            })
        
        detailed_summary = pd.DataFrame(summary_list)
        detailed_csv_path = os.path.join(self.output_dir, "team_detailed_summary.csv")
        detailed_summary.to_csv(detailed_csv_path, index=False)
        print(f"Team detailed summary saved to: {detailed_csv_path}")

    def plot_detailed_team_summary(self, save=True, show=True):
        """
        Plot a chart displaying min and max values for Speed, Distance, and Fatigue per team.
        This method reads the detailed summary CSV (generated by save_detailed_team_summary).
        """
        detailed_csv_path = os.path.join(self.output_dir, "team_detailed_summary.csv")
        if not os.path.exists(detailed_csv_path):
            print("Detailed summary CSV not found. Run save_detailed_team_summary() first.")
            return
        
        detailed_df = pd.read_csv(detailed_csv_path)
        teams = detailed_df["Team"].tolist()
        x = range(len(teams))
        width = 0.35

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Speed subplot
        axs[0].bar(x, detailed_df["Min Speed (km/h)"], width, label="Min Speed", color="blue")
        axs[0].bar([i + width for i in x], detailed_df["Max Speed (km/h)"], width, label="Max Speed", color="orange")
        axs[0].set_xticks([i + width/2 for i in x])
        axs[0].set_xticklabels(teams)
        axs[0].set_ylabel("Speed (km/h)")
        axs[0].set_title("Min and Max Speed per Team")
        axs[0].legend()
        
        # Distance subplot
        axs[1].bar(x, detailed_df["Min Distance (m)"], width, label="Min Distance", color="green")
        axs[1].bar([i + width for i in x], detailed_df["Max Distance (m)"], width, label="Max Distance", color="red")
        axs[1].set_xticks([i + width/2 for i in x])
        axs[1].set_xticklabels(teams)
        axs[1].set_ylabel("Distance (m)")
        axs[1].set_title("Min and Max Distance Covered per Team")
        axs[1].legend()
        
        # Fatigue subplot
        axs[2].bar(x, detailed_df["Min Fatigue"], width, label="Min Fatigue", color="purple")
        axs[2].bar([i + width for i in x], detailed_df["Max Fatigue"], width, label="Max Fatigue", color="brown")
        axs[2].set_xticks([i + width/2 for i in x])
        axs[2].set_xticklabels(teams)
        axs[2].set_ylabel("Fatigue Index")
        axs[2].set_title("Min and Max Fatigue per Team")
        axs[2].legend()

        plt.tight_layout()
        if save:
            plot_path = os.path.join(self.output_dir, "team_detailed_summary_chart.png")
            plt.savefig(plot_path)
            print(f"Team detailed summary chart saved to: {plot_path}")
        if show:
            plt.show()
        plt.close()

    def compare_team_fatigue(self, save=True, show=True):
        """
        Compute the average fatigue per team and display which team is more fatigued.
        Also generate a bar chart comparing average fatigue indices across teams,
        using blue for the first team and orange for the second.
        """
        if self.df.empty:
            print("No team data to compare fatigue.")
            return

        avg_fatigue = self.df.groupby("Team")["Fatigue Index"].mean().reset_index()
        # Identify the team with the highest average fatigue
        max_row = avg_fatigue.loc[avg_fatigue["Fatigue Index"].idxmax()]
        more_fatigued_team = max_row["Team"]
        print(f"Team {more_fatigued_team} is more fatigued on average with a fatigue index of {max_row['Fatigue Index']:.2f}.")

        fig, ax = plt.subplots(figsize=(8, 4))
        
        colors = ['blue', 'orange']
        for i, row in avg_fatigue.iterrows():
            ax.bar(row["Team"], row["Fatigue Index"], color=colors[i], label=row["Team"])
        
        ax.set_xlabel("Team")
        ax.set_ylabel("Average Fatigue Index")
        ax.set_title("Average Fatigue Index per Team")
        ax.legend()
        
        if save:
            avg_plot_path = os.path.join(self.output_dir, "team_avg_fatigue_comparison.png")
            plt.savefig(avg_plot_path)
            print(f"Team average fatigue comparison chart saved to: {avg_plot_path}")
        if show:
            plt.show()
        plt.close()

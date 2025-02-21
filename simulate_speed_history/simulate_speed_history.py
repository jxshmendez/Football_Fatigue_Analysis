import numpy as np

def simulate_speed_history(num_frames, initial_speed=6.0, fatigue_rate=0.005, noise_std=0.1):
    """
    Simulate a synthetic speed history over a number of frames.
    
    Parameters:
        num_frames (int): Number of speed data points (frames) to simulate.
        initial_speed (float): Starting speed in km/h.
        fatigue_rate (float): Amount (in km/h) by which the speed decreases per frame.
        noise_std (float): Standard deviation of random noise added to each frame.
        
    Returns:
        list: A list of simulated speeds over time.
    """
    speeds = []
    for i in range(num_frames):
        speed = initial_speed - i * fatigue_rate
        noise = np.random.normal(0, noise_std)
        speeds.append(speed + noise)
    return speeds



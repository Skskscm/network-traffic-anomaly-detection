import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_ddos_data(num_points=10000, attack_periods=8):
    """
    Generate realistic network traffic data with DDoS attacks
    Returns: DataFrame with timestamp, packets_per_second, is_anomaly
    """
    print("Generating synthetic network traffic data...")
    
    # Create time index
    timestamps = pd.date_range('2024-01-01', periods=num_points, freq='S')
    
    # Base normal traffic patterns
    t = np.arange(num_points)
    
    # Daily pattern (24-hour cycle)
    daily_pattern = 1000 + 500 * np.sin(2 * np.pi * t / (24 * 3600))
    
    # Weekly pattern (7-day cycle)
    weekly_pattern = 200 * np.sin(2 * np.pi * t / (7 * 24 * 3600))
    
    # Random noise
    noise = np.random.normal(0, 50, num_points)
    
    # Combine to create normal traffic
    normal_traffic = daily_pattern + weekly_pattern + noise
    normal_traffic = np.maximum(normal_traffic, 100)  # Minimum traffic
    
    # Create labels (all normal initially)
    labels = np.zeros(num_points)
    
    # Add DDoS attack periods - fix the range issue
    min_start = 50
    max_start = num_points - 150
    
    if max_start > min_start:
        attack_starts = np.random.choice(range(min_start, max_start), min(attack_periods, max_start-min_start), replace=False)
    else:
        attack_starts = [num_points // 4]  # Fallback to a single attack in the middle
    
    attack_durations = np.random.randint(20, min(100, num_points//5), len(attack_starts))
    
    for start, duration in zip(attack_starts, attack_durations):
        end = min(start + duration, num_points)
        attack_intensity = np.random.uniform(3, 10)  # Reduced intensity for smaller datasets
        
        # Mark attack period
        for i in range(start, end):
            if i < num_points:
                normal_traffic[i] *= attack_intensity
                labels[i] = 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'packets_per_second': normal_traffic,
        'is_anomaly': labels.astype(int)
    })
    
    print(f"Generated {len(df)} data points with {labels.sum()} attack points")
    return df
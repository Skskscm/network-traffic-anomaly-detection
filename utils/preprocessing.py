import numpy as np
import pandas as pd
import torch  # Add this import
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

def create_sequences(data, sequence_length):
    """
    Create sequences from time series data
    """
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

def prepare_data(df, sequence_length=60, train_ratio=0.7):
    """
    Prepare data for Donut model training and testing
    """
    print("Preparing data for training...")
    
    # Normalize the data
    scaler = MinMaxScaler()
    traffic_values = df['packets_per_second'].values.reshape(-1, 1)
    scaled_traffic = scaler.fit_transform(traffic_values).flatten()
    
    # Split data into training (normal only) and testing (mixed)
    # Ensure we have attacks in both sets for proper evaluation
    train_size = int(len(df) * train_ratio)
    
    # Use first portion for training (normal only)
    train_data = scaled_traffic[:train_size]
    
    # Use second portion for testing (mixed normal and attacks)
    test_data = scaled_traffic[train_size:]
    test_labels = df['is_anomaly'].values[train_size:]
    
    print(f"Training data points: {len(train_data)}")
    print(f"Testing data points: {len(test_data)}")
    print(f"Attacks in test set: {test_labels.sum()}")
    
    # Create sequences
    X_train = create_sequences(train_data, sequence_length)
    X_test = create_sequences(test_data, sequence_length)
    y_test = test_labels[sequence_length:]  # Align labels with sequences
    
    print(f"Training sequences: {X_train.shape}")
    print(f"Testing sequences: {X_test.shape}")
    
    return X_train, X_test, y_test, scaler
class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data"""
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

print("Preprocessing utilities defined successfully!")
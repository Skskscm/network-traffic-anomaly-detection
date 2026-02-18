"""
Configuration file for DDoS Detection using Donut VAE
"""

class Config:
    # Data parameters
    SEQUENCE_LENGTH = 60  # Look-back window in seconds
    TRAIN_RATIO = 0.7     # Ratio of data to use for training
    BATCH_SIZE = 64
    
    # Model architecture
    HIDDEN_DIM = 100
    LATENT_DIM = 20
    
    # Training parameters
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    BETA = 1.0  # Weight for KL divergence in VAE loss
    
    # Anomaly detection
    THRESHOLD_QUANTILE = 0.95  # Quantile for anomaly threshold
    
    # Paths
    DATA_PATH = "data/"
    MODEL_PATH = "models/donut_model.pth"
    RESULTS_PATH = "results/"
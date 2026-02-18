# detect.py - Real-time DDoS detection
import torch
import numpy as np
import time
from models.donut_model import DonutVAE
from config import Config

def real_time_detection():
    """Real-time DDoS detection using the trained model"""
    print("Initializing real-time DDoS detection system...")
    
    # Load model
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DonutVAE(
        sequence_length=cfg.SEQUENCE_LENGTH,
        hidden_dim=cfg.HIDDEN_DIM,
        latent_dim=cfg.LATENT_DIM
    ).to(device)
    
    model.load_state_dict(torch.load(cfg.MODEL_PATH))
    model.eval()
    
    # Use the optimal threshold we validated
    detection_threshold = 12208.113270
    
    print(f"Model: Donut VAE")
    print(f"Detection threshold: {detection_threshold:.2f}")
    print(f"Device: {device}")
    print("Real-time monitoring started...")
    print("-" * 50)
    
    # Real-time monitoring simulation
    monitoring_duration = 60  # Monitor for 60 time points
    attack_count = 0
    normal_count = 0
    
    for i in range(monitoring_duration):
        # Simulate incoming network traffic (replace with actual data source)
        if i % 15 == 0 and i > 0:  # Simulate periodic attacks
            traffic_data = np.random.normal(800, 200, cfg.SEQUENCE_LENGTH)  # DDoS pattern
            actual_status = "DDoS_ATTACK"
        else:
            traffic_data = np.random.normal(100, 25, cfg.SEQUENCE_LENGTH)   # Normal pattern
            actual_status = "NORMAL"
        
        # Detect anomalies
        with torch.no_grad():
            input_data = torch.FloatTensor(traffic_data).view(1, -1).to(device)
            reconstructed, _, _ = model(input_data)
            reconstruction_error = torch.nn.MSELoss()(reconstructed, input_data)
            
            is_anomaly = reconstruction_error.item() > detection_threshold
            detected_status = "DETECTED" if is_anomaly else "NORMAL"
            
        # Update counters
        if actual_status == "DDoS_ATTACK":
            if is_anomaly:
                attack_count += 1
                alert_level = "CRITICAL"
            else:
                alert_level = "MISSED"
        else:
            if is_anomaly:
                alert_level = "FALSE_ALARM"
            else:
                normal_count += 1
                alert_level = "NORMAL"
        
        # Display results
        if alert_level in ["CRITICAL", "MISSED", "FALSE_ALARM"]:
            print(f"Time {i:02d}: [{alert_level:<12}] {actual_status} - Error: {reconstruction_error.item():.2f}")
        
        time.sleep(0.1)  # Simulate real-time interval
    
    # Summary
    print("-" * 50)
    print("REAL-TIME DETECTION SUMMARY")
    print(f"Monitoring Duration: {monitoring_duration} time points")
    print(f"DDoS Attacks Detected: {attack_count}")
    print(f"Normal Traffic Periods: {normal_count}")
    print(f"Detection System: ACTIVE")

if __name__ == "__main__":
    real_time_detection()
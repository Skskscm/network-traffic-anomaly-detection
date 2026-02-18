# final_demo.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.donut_model import DonutVAE
from config import Config

def live_ddos_detection_demo():
    """Final demonstration of your DDoS detection system"""
    print("=== DDoS Detection System - Final Demo ===")
    
    # Load trained model
    model = DonutVAE(sequence_length=60, hidden_dim=100, latent_dim=20)
    model.load_state_dict(torch.load('models/donut_model.pth'))
    model.eval()
    
    # Simulate real-time traffic monitoring
    print("Starting real-time network monitoring...")
    
    # Generate simulated traffic with occasional attacks
    time_points = 500
    traffic = []
    detections = []
    reconstruction_errors = []
    
    # Collect baseline errors first to set better threshold
    print("Calibrating detection threshold...")
    baseline_errors = []
    for i in range(100):  # Collect 100 normal samples
        normal_traffic = np.random.normal(100, 20, 60)
        with torch.no_grad():
            input_data = torch.FloatTensor(normal_traffic).view(1, -1)
            reconstructed, _, _ = model(input_data)
            error = torch.nn.MSELoss()(reconstructed, input_data)
            baseline_errors.append(error.item())
    
    # Calculate adaptive threshold based on normal traffic
    baseline_mean = np.mean(baseline_errors)
    baseline_std = np.std(baseline_errors)
    adaptive_threshold = baseline_mean + 3 * baseline_std  # 3-sigma rule
    
    print(f"Calibrated Threshold: {adaptive_threshold:.6f}")
    print(f"Baseline Normal Error: {baseline_mean:.6f} ± {baseline_std:.6f}")
    
    # Main monitoring loop
    for i in range(time_points):
        # Simulate normal traffic with occasional DDoS
        if i % 100 == 0 and i > 0:  # Attack every 100 time points
            # More realistic DDoS pattern - sustained high traffic
            attack_duration = 10
            if i % 100 < attack_duration:
                current_traffic = np.random.normal(800, 150, 60)  # DDoS pattern
                attack = True
            else:
                current_traffic = np.random.normal(100, 20, 60)   # Normal pattern
                attack = False
        else:
            current_traffic = np.random.normal(100, 20, 60)   # Normal pattern  
            attack = False
        
        # Detect using your model
        with torch.no_grad():
            input_data = torch.FloatTensor(current_traffic).view(1, -1)
            reconstructed, _, _ = model(input_data)
            error = torch.nn.MSELoss()(reconstructed, input_data)
            
            # Use adaptive threshold
            is_anomaly = error.item() > adaptive_threshold
            
        traffic.append(current_traffic.mean())
        detections.append(is_anomaly)
        reconstruction_errors.append(error.item())
        
        # Log only significant events
        if is_anomaly:
            if attack:
                print(f"Time {i}: [DETECTED] DDoS Attack - Error: {error.item():.6f}")
            else:
                if error.item() > adaptive_threshold * 1.5:  # Only log strong false positives
                    print(f"Time {i}: [FALSE POSITIVE] Error: {error.item():.6f}")
    
    # Plot final results
    plt.figure(figsize=(14, 10))
    
    # Main traffic plot
    plt.subplot(3, 1, 1)
    plt.plot(traffic, 'b-', label='Network Traffic', alpha=0.7, linewidth=1)
    
    # Mark true detections (attacks that were correctly detected)
    true_detection_times = [i for i in range(time_points) 
                           if detections[i] and (i % 100 == 0 and i > 0 and i % 100 < 10)]
    true_detection_values = [traffic[i] for i in true_detection_times]
    
    # Mark false positives
    false_positive_times = [i for i in range(time_points) 
                           if detections[i] and not (i % 100 == 0 and i > 0 and i % 100 < 10)]
    false_positive_values = [traffic[i] for i in false_positive_times]
    
    plt.scatter(true_detection_times, true_detection_values, color='red', s=80, 
                label='True DDoS Detections', zorder=5, marker='X')
    plt.scatter(false_positive_times, false_positive_values, color='orange', s=40,
                label='False Positives', zorder=4, marker='o', alpha=0.6)
    
    plt.axhline(y=400, color='orange', linestyle='--', label='High Traffic Threshold')
    plt.ylabel('Packets per Second')
    plt.title('DDoS Detection System - Real-time Network Monitoring')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Reconstruction errors plot
    plt.subplot(3, 1, 2)
    plt.plot(reconstruction_errors, 'g-', alpha=0.7, label='Reconstruction Error')
    plt.axhline(y=adaptive_threshold, color='red', linestyle='--', 
                label=f'Detection Threshold: {adaptive_threshold:.4f}')
    
    # Mark regions with actual attacks
    for i in range(time_points):
        if i % 100 == 0 and i > 0 and i % 100 < 10:
            plt.axvspan(i-0.5, i+9.5, alpha=0.2, color='red', label='Actual DDoS Period' if i == 100 else "")
    
    plt.ylabel('Reconstruction Error')
    plt.xlabel('Time (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Model Reconstruction Errors')
    
    # Detection status plot
    plt.subplot(3, 1, 3)
    status = []
    for i in range(time_points):
        if detections[i] and (i % 100 == 0 and i > 0 and i % 100 < 10):
            status.append(2)  # True positive
        elif detections[i]:
            status.append(1)  # False positive
        else:
            status.append(0)  # Normal
    
    colors = ['green', 'orange', 'red']
    labels = ['Normal', 'False Positive', 'True Positive']
    
    for i in range(time_points):
        plt.scatter(i, status[i], color=colors[status[i]], s=20, alpha=0.7)
    
    plt.yticks([0, 1, 2], labels)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Detection Status')
    plt.title('Detection Accuracy Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/final_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("DETECTION SUMMARY")
    print("="*50)
    
    true_positives = sum(1 for i in range(time_points) 
                        if detections[i] and (i % 100 == 0 and i > 0 and i % 100 < 10))
    false_positives = sum(1 for i in range(time_points) 
                         if detections[i] and not (i % 100 == 0 and i > 0 and i % 100 < 10))
    attacks_occurred = sum(1 for i in range(time_points) 
                          if (i % 100 == 0 and i > 0 and i % 100 < 10))
    total_normal = time_points - attacks_occurred
    
    print(f"Total Time Points: {time_points}")
    print(f"Actual DDoS Attacks: {attacks_occurred}")
    print(f"True Positives: {true_positives}/{attacks_occurred}")
    print(f"False Positives: {false_positives}/{total_normal}")
    print(f"Detection Rate: {true_positives/attacks_occurred*100:.1f}%")
    print(f"False Positive Rate: {false_positives/total_normal*100:.1f}%")
    print(f"Adaptive Threshold: {adaptive_threshold:.6f}")
    print("\nDemo Complete - Check 'results/plots/final_demo.png'")

if __name__ == "__main__":
    live_ddos_detection_demo()
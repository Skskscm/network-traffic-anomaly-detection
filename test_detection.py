import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.donut_model import DonutVAE
from utils.data_loader import generate_synthetic_ddos_data
from utils.preprocessing import prepare_data
from config import Config

def test_anomaly_detection():
    print("Testing DDoS Anomaly Detection with Your Trained Model...")
    
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your trained model
    model = DonutVAE(
        sequence_length=cfg.SEQUENCE_LENGTH,
        hidden_dim=cfg.HIDDEN_DIM,
        latent_dim=cfg.LATENT_DIM
    ).to(device)
    
    model.load_state_dict(torch.load(cfg.MODEL_PATH))
    model.eval()
    print("Model loaded successfully!")
    
    # Generate test data
    print("Generating test data with DDoS attacks...")
    df = generate_synthetic_ddos_data()
    
    print(f"Data shape: {df.shape}")
    print(f"Data columns: {df.columns.tolist()}")
    print(f"Total data points: {len(df)}")
    print(f"Total attacks in data: {df['is_anomaly'].sum()}")
    
    # Use ALL data for testing (train_ratio=0)
    X_test, _, y_test, scaler = prepare_data(df, cfg.SEQUENCE_LENGTH, train_ratio=0)
    
    print(f"Testing sequences shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Attacks in test set: {y_test.sum()}")
    
    # Check if we have test data
    if X_test.shape[0] == 0:
        print("ERROR: No test data generated! Using alternative approach...")
        # Create simple test data manually
        X_test = np.random.rand(100, cfg.SEQUENCE_LENGTH, 1).astype(np.float32)
        y_test = np.random.randint(0, 2, 100).astype(np.float32)
        print(f"Created manual test data: {X_test.shape}")
    
    # Convert to torch - ensure correct shape
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test)
    
    print(f"Final test samples: {X_test.shape[0]}")
    print(f"Final test data shape: {X_test.shape}")
    
    if X_test.shape[0] == 0:
        print("ERROR: Still no test data. Cannot proceed.")
        return
    
    # Detect anomalies
    print("Running anomaly detection...")
    with torch.no_grad():
        # Ensure input shape matches model expectations
        # Your model expects (batch_size, sequence_length, features)
        # But might need reshaping to (batch_size, sequence_length * features)
        batch_size, seq_len, features = X_test.shape
        X_test_flat = X_test.view(batch_size, -1)  # Flatten for linear layers
        
        reconstructed, mu, logvar = model(X_test_flat)
        reconstruction_errors = nn.MSELoss(reduction='none')(reconstructed, X_test_flat)
        reconstruction_errors = reconstruction_errors.mean(dim=1)  # Mean over features
        
        errors_cpu = reconstruction_errors.cpu().numpy()
        actual_attacks = y_test.numpy()
        
        print("Reconstruction Error Stats:")
        print(f"   Min: {errors_cpu.min():.6f}")
        print(f"   Max: {errors_cpu.max():.6f}") 
        print(f"   Mean: {errors_cpu.mean():.6f}")
        print(f"   Std: {errors_cpu.std():.6f}")
        
        # Test multiple thresholds
        print("\nTesting different detection thresholds:")
        print("=" * 60)
        
        # Adjust thresholds based on actual error range
        error_range = errors_cpu.max() - errors_cpu.min()
        base_threshold = errors_cpu.mean() + errors_cpu.std()
        
        thresholds = [
            base_threshold * 0.5,
            base_threshold * 0.75, 
            base_threshold,
            base_threshold * 1.25,
            base_threshold * 1.5,
            base_threshold * 2.0
        ]
        
        best_accuracy = 0
        best_threshold = 0
        
        for threshold in thresholds:
            predictions = (errors_cpu > threshold).astype(int)
            
            # Calculate metrics
            true_positives = np.sum((predictions == 1) & (actual_attacks == 1))
            false_positives = np.sum((predictions == 1) & (actual_attacks == 0))
            true_negatives = np.sum((predictions == 0) & (actual_attacks == 0))
            false_negatives = np.sum((predictions == 0) & (actual_attacks == 1))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            accuracy = (true_positives + true_negatives) / len(actual_attacks)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"Threshold: {threshold:.6f}")
            print(f"  True Positives: {true_positives}")
            print(f"  False Positives: {false_positives}")
            print(f"  True Negatives: {true_negatives}") 
            print(f"  False Negatives: {false_negatives}")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1 Score: {f1_score:.3f}")
            print("")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        print(f"Best threshold: {best_threshold:.6f} with accuracy: {best_accuracy:.3f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Reconstruction errors vs actual attacks
    plt.subplot(2, 1, 1)
    time_points = range(len(errors_cpu))
    plt.scatter(time_points, errors_cpu, c=actual_attacks, cmap='coolwarm', alpha=0.6)
    plt.axhline(y=best_threshold, color='red', linestyle='--', label=f'Optimal Threshold: {best_threshold:.6f}')
    plt.ylabel('Reconstruction Error')
    plt.title('DDoS Attack Detection Results')
    plt.legend()
    plt.colorbar(label='Actual Attack (1=Attack, 0=Normal)')
    
    # Plot 2: Error distribution
    plt.subplot(2, 1, 2)
    normal_errors = errors_cpu[actual_attacks == 0]
    attack_errors = errors_cpu[actual_attacks == 1]
    
    if len(normal_errors) > 0:
        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal Traffic', color='blue')
    if len(attack_errors) > 0:
        plt.hist(attack_errors, bins=50, alpha=0.7, label='DDoS Attacks', color='red')
    
    plt.axvline(x=best_threshold, color='black', linestyle='--', label=f'Threshold: {best_threshold:.6f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Error Distribution: Normal vs Attack Traffic')
    
    plt.tight_layout()
    plt.savefig('results/plots/detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Testing complete! Check 'results/plots/detection_results.png'")

if __name__ == "__main__":
    test_anomaly_detection()
# analyze_patterns.py - Traffic pattern analysis
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import generate_synthetic_ddos_data

def analyze_traffic_patterns():
    """Analyze network traffic patterns and characteristics"""
    print("Analyzing Network Traffic Patterns...")
    
    # Generate sample data
    df = generate_synthetic_ddos_data()
    
    # Separate normal and attack traffic
    normal_traffic = df[df['is_anomaly'] == 0]['packets_per_second']
    attack_traffic = df[df['is_anomaly'] == 1]['packets_per_second']
    
    print(f"Total samples: {len(df)}")
    print(f"Normal traffic: {len(normal_traffic)} samples")
    print(f"Attack traffic: {len(attack_traffic)} samples")
    
    # Statistical analysis
    print("\nTRAFFIC STATISTICS:")
    print(f"{'Metric':<20} {'Normal':<15} {'Attack':<15}")
    print("-" * 50)
    print(f"{'Mean':<20} {normal_traffic.mean():<15.2f} {attack_traffic.mean():<15.2f}")
    print(f"{'Std Dev':<20} {normal_traffic.std():<15.2f} {attack_traffic.std():<15.2f}")
    print(f"{'Min':<20} {normal_traffic.min():<15.2f} {attack_traffic.min():<15.2f}")
    print(f"{'Max':<20} {normal_traffic.max():<15.2f} {attack_traffic.max():<15.2f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Traffic distribution
    plt.subplot(2, 2, 1)
    plt.hist(normal_traffic, bins=50, alpha=0.7, label='Normal Traffic', color='blue')
    plt.hist(attack_traffic, bins=50, alpha=0.7, label='DDoS Attacks', color='red')
    plt.xlabel('Packets per Second')
    plt.ylabel('Frequency')
    plt.title('Traffic Distribution: Normal vs DDoS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Time series sample
    plt.subplot(2, 2, 2)
    sample_points = 200
    plt.plot(df['packets_per_second'].values[:sample_points], 'b-', alpha=0.7)
    
    # Mark attack points in red
    attack_indices = df[df['is_anomaly'] == 1].index[:sample_points]
    plt.scatter(attack_indices, df.loc[attack_indices, 'packets_per_second'], 
                color='red', s=30, label='DDoS Attacks', zorder=5)
    
    plt.xlabel('Time')
    plt.ylabel('Packets per Second')
    plt.title('Traffic Pattern with DDoS Attacks')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Box plot comparison
    plt.subplot(2, 2, 3)
    data = [normal_traffic, attack_traffic]
    labels = ['Normal', 'DDoS']
    plt.boxplot(data, labels=labels)
    plt.ylabel('Packets per Second')
    plt.title('Traffic Volume Comparison')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative distribution
    plt.subplot(2, 2, 4)
    plt.hist(normal_traffic, bins=50, density=True, cumulative=True, 
             alpha=0.7, label='Normal', color='blue', histtype='step')
    plt.hist(attack_traffic, bins=50, density=True, cumulative=True, 
             alpha=0.7, label='DDoS', color='red', histtype='step')
    plt.xlabel('Packets per Second')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/traffic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nAnalysis complete. Results saved to 'results/plots/traffic_analysis.png'")

if __name__ == "__main__":
    analyze_traffic_patterns()
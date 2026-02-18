import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import get_ddos_dataloader  # Your existing function
from models.donut_model import YourModel  # Your existing model
from dashboard.live_dashboard import LiveTrainingDashboard
import torch

def main():
    print("Starting Live DDoS Detection Dashboard...")
    
    # Use your existing data loader
    train_loader, test_loader = get_ddos_dataloader()
    
    # Use your existing model (donut or VAE - whatever you have)
    model = YourModel()  # Replace with your actual model loading
    
    # Initialize dashboard
    dashboard = LiveTrainingDashboard()
    
    # Simulate training updates (replace with your actual training)
    for epoch in range(100):
        # Simulate your training - replace with actual training loop
        simulated_loss = 0.1 / (1 + epoch * 0.1)  # Example loss decrease
        
        dashboard.update_training_plot(epoch, simulated_loss)
        dashboard.update_live_traffic(epoch)
        dashboard.update_stats(epoch, simulated_loss, epoch % 25)
        
        plt.pause(0.1)  # Update every 0.1 seconds
    
    print("Dashboard demo complete!")
    plt.show()

if __name__ == "__main__":
    main()
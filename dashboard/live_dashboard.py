import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
from collections import deque

class LiveTrainingDashboard:
    def __init__(self):
        # Use your existing model and data loader
        self.fig = plt.figure(figsize=(15, 10))
        self.setup_plots()
        
        # Data storage for live updates
        self.epochs = []
        self.losses = []
        self.reconstruction_errors = deque(maxlen=100)  # Store last 100 errors
        
    def setup_plots(self):
        # Plot 1: Training Progress (using YOUR existing loss)
        self.ax1 = plt.subplot(2, 2, 1)
        self.ax1.set_title('Your Model Training Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.loss_line, = self.ax1.plot([], [], 'b-', label='Your Loss')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Plot 2: Reconstruction Errors (for anomaly detection)
        self.ax2 = plt.subplot(2, 2, 2)
        self.ax2.set_title('Reconstruction Errors - Live')
        self.ax2.set_xlabel('Samples')
        self.ax2.set_ylabel('Error')
        self.error_line, = self.ax2.plot([], [], 'r-', alpha=0.7)
        self.threshold_line = self.ax2.axhline(y=0.1, color='g', linestyle='--', label='Threshold')
        self.ax2.legend()
        self.ax2.grid(True)
        
        # Plot 3: Live Traffic Simulation
        self.ax3 = plt.subplot(2, 2, 3)
        self.ax3.set_title('Network Traffic - Live')
        self.ax3.set_xlabel('Time')
        self.ax3.set_ylabel('Traffic Volume')
        self.traffic_line, = self.ax3.plot([], [], 'b-', label='Traffic')
        self.anomaly_scatter = self.ax3.scatter([], [], c='red', s=50, label='Anomalies')
        self.ax3.legend()
        self.ax3.grid(True)
        
        # Plot 4: Detection Statistics
        self.ax4 = plt.subplot(2, 2, 4)
        self.ax4.set_title('Detection Performance')
        self.ax4.axis('off')  # We'll use text for stats
        self.stats_text = self.ax4.text(0.1, 0.9, 'Initializing...', transform=self.ax4.transAxes, 
                                       fontsize=12, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
    
    def update_training_plot(self, epoch, loss):
        """Update training loss plot with your actual training data"""
        self.epochs.append(epoch)
        self.losses.append(loss)
        
        self.loss_line.set_data(self.epochs, self.losses)
        self.ax1.relim()
        self.ax1.autoscale_view()
    
    def update_anomaly_detection(self, reconstruction_error):
        """Update reconstruction errors with actual error values - NEW FUNCTION"""
        self.reconstruction_errors.append(reconstruction_error)
        
        # Update the plot
        x_data = range(len(self.reconstruction_errors))
        self.error_line.set_data(x_data, self.reconstruction_errors)
        
        # Auto-adjust y-axis
        if len(self.reconstruction_errors) > 0:
            current_errors = list(self.reconstruction_errors)
            self.ax2.set_ylim(min(current_errors) * 0.9, max(current_errors) * 1.1)
        
        self.ax2.relim()
        self.ax2.autoscale_view()
    
    def update_live_traffic(self, current_step):
        """Simulate live network traffic"""
        time_points = np.arange(max(0, current_step - 50), current_step)
        
        # Simulate normal traffic with occasional spikes
        traffic = 100 + 30 * np.sin(time_points * 0.1) + np.random.normal(0, 10, len(time_points))
        
        # Add occasional DDoS attacks (every 100 steps)
        if current_step % 100 < 10:
            traffic[-10:] += 200  # Attack spike
        
        self.traffic_line.set_data(time_points, traffic)
        self.ax3.relim()
        self.ax3.autoscale_view()
    
    def update_stats(self, epoch, loss, detected_anomalies, live_samples):
        """Update statistics text - FIXED VERSION"""
        stats = f"""
        Epoch: {epoch}
        Loss: {loss:.6f}
        Anomalies Detected: {detected_anomalies}
        Live Samples: {live_samples}
        Status: {'Training' if epoch < 100 else 'Monitoring'}
        Reconstruction Errors: {len(self.reconstruction_errors)}
        """
        self.stats_text.set_text(stats)
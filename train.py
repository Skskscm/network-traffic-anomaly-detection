import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.donut_model import DonutVAE
from utils.data_loader import generate_synthetic_ddos_data
from utils.preprocessing import prepare_data, TimeSeriesDataset
from config import Config
from dashboard.live_dashboard import LiveTrainingDashboard  # Import the dashboard

def vae_loss(reconstructed_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function with beta parameter for anomaly detection
    """
    # Reconstruction loss (MSE)
    reconstruction_loss = nn.MSELoss()(reconstructed_x, x)
    
    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_divergence /= x.size(0)  # Normalize by batch size
    
    # Total loss
    total_loss = reconstruction_loss + beta * kl_divergence
    
    return total_loss, reconstruction_loss, kl_divergence

def train_model_with_live_dashboard():
    """Main training function WITH live dashboard"""
    print("Starting Donut VAE Training for DDoS Detection with Live Dashboard...")
    
    # Configuration
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate and prepare data
    df = generate_synthetic_ddos_data()
    X_train, X_test, y_test, scaler = prepare_data(df, cfg.SEQUENCE_LENGTH, cfg.TRAIN_RATIO)
    
    # Create data loaders
    train_dataset = TimeSeriesDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = DonutVAE(
        sequence_length=cfg.SEQUENCE_LENGTH,
        hidden_dim=cfg.HIDDEN_DIM,
        latent_dim=cfg.LATENT_DIM
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # Initialize LIVE DASHBOARD
    dashboard = LiveTrainingDashboard()
    
    # Training loop
    model.train()
    train_losses = []
    
    print("Training started with Live Dashboard...")
    for epoch in range(cfg.EPOCHS):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, mu, logvar = model(batch)
            
            # Calculate loss
            loss, recon_loss, kl_loss = vae_loss(reconstructed, batch, mu, logvar, cfg.BETA)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            #  LIVE DASHBOARD UPDATE - FIXED VERSION
            if batch_idx % 10 == 0:  # Update every 10 batches
                current_progress = epoch + (batch_idx / len(train_loader))
                
                # Calculate reconstruction error for THIS BATCH (FAST)
                with torch.no_grad():
                    reconstruction_error = nn.MSELoss()(reconstructed, batch)
                    detected_anomalies = (reconstruction_error > 0.05).sum().item()
                    live_samples = (epoch * len(train_loader) + batch_idx) * cfg.BATCH_SIZE
                
                # Update ALL dashboard components
                dashboard.update_training_plot(current_progress, loss.item())
                dashboard.update_live_traffic(epoch * len(train_loader) + batch_idx)
                dashboard.update_anomaly_detection(reconstruction_error.mean())  # NEW LINE
                dashboard.update_stats(epoch, loss.item(), detected_anomalies, live_samples)  # UPDATED
                
                plt.pause(0.01)  # Allow plot to update
        
        # Average losses for the epoch
        avg_loss = epoch_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_kl_loss = epoch_kl_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{cfg.EPOCHS}], '
                  f'Loss: {avg_loss:.4f}, '
                  f'Recon: {avg_recon_loss:.4f}, '
                  f'KL: {avg_kl_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), cfg.MODEL_PATH)
    print(f"Model saved to {cfg.MODEL_PATH}")
    
    # Plot final training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Donut VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/plots/training_loss.png')
    plt.show()  # Keep the dashboard open
    
    # Save training metrics
    from utils.results_manager import ResultsManager
    
    results_manager = ResultsManager()
    
    training_metrics = {
        'final_loss': avg_loss,
        'final_reconstruction_loss': avg_recon_loss,
        'final_kl_loss': avg_kl_loss,
        'training_time': f"{cfg.EPOCHS} epochs",
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'training_sequences': X_train.shape[0],
        'training_data_points': X_train.shape[0]
    }
    
    results_manager.save_training_metrics(training_metrics)
    
    print("Training with Live Dashboard completed successfully!")
    return model, X_test, y_test, scaler


if __name__ == "__main__":
    train_model_with_live_dashboard()
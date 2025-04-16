#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import csv

def parse_training_log(log_file='training_log.txt'):
    """Parse training_log.txt in CSV format with columns Epoch,Train,Valid,Test."""
    train_losses = []
    valid_losses = []
    test_losses = []
    epochs = []
    
    try:
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            # Skip header row
            next(reader)
            
            for row in reader:
                if len(row) >= 4:  # Ensure row has enough columns
                    epochs.append(int(row[0]))
                    train_losses.append(float(row[1]))
                    valid_losses.append(float(row[2]))
                    test_losses.append(float(row[3]))
    
    except Exception as e:
        print(f"Error parsing log file: {e}")
        sys.exit(1)
    
    # Convert to numpy arrays
    return np.array(train_losses), np.array(valid_losses), np.array(test_losses)

def main():
    parser = argparse.ArgumentParser(description='Plot training, validation, and test losses on log-log scale')
    parser.add_argument('--log_file', type=str, default='training_log.txt',
                        help='Path to the training log file (default: training_log.txt)')
    args = parser.parse_args()
    
    # Check if log file exists
    log_file = args.log_file
    if not os.path.exists(log_file):
        print(f"Error: Training log file '{log_file}' not found.")
        sys.exit(1)
    
    print(f"Parsing losses from: {log_file}")
    
    # Parse the training log to get losses
    train_losses, valid_losses, test_losses = parse_training_log(log_file)
    
    # Check if we extracted any data
    if len(train_losses) == 0 or len(valid_losses) == 0 or len(test_losses) == 0:
        print("Error: Could not extract losses from the log file. Check the format.")
        sys.exit(1)
    
    # Calculate minimum validation loss
    min_valid_loss = np.min(valid_losses)
    min_valid_epoch = np.argmin(valid_losses)
    
    # Plot with LaTeX rendering
    try:
        # Enable LaTeX rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif"
        })
        
        # Create figure with log-log scale
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create x-axis for epochs
        epochs = np.arange(1, len(valid_losses) + 1)
        
        # Plot with log scale on both axes
        ax.loglog(epochs, train_losses, label=r'Training Loss', alpha=0.7)
        ax.loglog(epochs, valid_losses, label=r'Validation Loss', marker='o', linestyle='-')
        ax.loglog(epochs, test_losses, label=r'Test Loss', marker='s', linestyle='-')
        
        # Mark minimum validation loss
        ax.scatter(min_valid_epoch + 1, min_valid_loss, color='red', s=100, 
                   marker='*', zorder=10, label=f'Min Validation Loss: {min_valid_loss:.6f}')
        
        # Set labels and title
        ax.set_xlabel(r'Epoch')
        ax.set_ylabel(r'Loss (log scale)')
        ax.set_title(r'Training, Validation and Test Losses (Log-Log Scale)')
        
        # Add grid and legend
        ax.grid(True, which="both", ls="--", alpha=0.7)
        ax.legend(loc='best')
        
        # Ensure entire plot is visible
        plt.tight_layout()
        
        # Save with LaTeX rendering
        plt.savefig('loss_curves_loglog_latex.png', dpi=300, bbox_inches='tight')
        print("Loss curves with LaTeX rendering saved as loss_curves_loglog_latex.png")
        
    except Exception as e:
        print(f"Error with LaTeX rendering: {e}")
        print("Falling back to standard rendering...")
        
        # Reset matplotlib parameters
        plt.rcParams.update({"text.usetex": False})
        plt.figure(figsize=(10, 6))
        
        # Create x-axis for epochs
        epochs = np.arange(1, len(valid_losses) + 1)
        
        # Plot with log scale on both axes
        plt.loglog(epochs, train_losses, label='Train Loss', marker='*', linestyle='-')
        plt.loglog(epochs, valid_losses, label='Validation Loss', marker='o', linestyle='-')
        plt.loglog(epochs, test_losses, label='Test Loss', marker='s', linestyle='-')
        
        # Mark minimum validation loss
        plt.scatter(min_valid_epoch + 1, min_valid_loss, color='red', s=100, 
                   marker='*', zorder=10, label=f'Min Validation Loss: {min_valid_loss:.6f}')
        
        # Set labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training, Validation and Test Losses (Log-Log Scale)')
        
        # Add grid and legend
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.legend(loc='best')
        
        # Save with standard rendering
        plt.tight_layout()
        plt.savefig('loss_curves_loglog.png', dpi=300, bbox_inches='tight')
        print("Loss curves saved as loss_curves_loglog.png")

if __name__ == "__main__":
    main() 
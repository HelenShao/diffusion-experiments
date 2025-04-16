#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description='Plot training, validation, and test losses on log-log scale')
    parser.add_argument('--model_name', type=str, default=None, 
                        help='Base model name (without extension) to load losses. If None, will use the latest available.')
    args = parser.parse_args()
    
    # Find the model name if not specified
    if args.model_name is None:
        # Try to find any valid loss files
        train_files = glob.glob('*_train_losses.npy')
        if train_files:
            # Extract base model name from first found file
            args.model_name = train_files[0].replace('_train_losses.npy', '')
            print(f"Using model: {args.model_name}")
        else:
            print("Error: No loss files found. Specify --model_name or ensure loss files exist.")
            sys.exit(1)
    
    # File paths for losses
    train_loss_file = f"{args.model_name}_train_losses.npy"
    valid_loss_file = f"{args.model_name}_valid_losses.npy"
    test_loss_file = f"{args.model_name}_test_losses.npy"
    
    # Check if files exist
    for file_path in [train_loss_file, valid_loss_file, test_loss_file]:
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.")
            sys.exit(1)
    
    # Load the losses
    train_losses = np.load(train_loss_file)
    valid_losses = np.load(valid_loss_file)
    test_losses = np.load(test_loss_file)
    
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
        
        # For train losses which has more points (multiple per epoch)
        train_steps = np.linspace(1, len(valid_losses), len(train_losses))
        
        # Plot with log scale on both axes
        ax.loglog(train_steps, train_losses, label=r'Training Loss', alpha=0.7)
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
        
        # For train losses which has more points (multiple per epoch)
        train_steps = np.linspace(1, len(valid_losses), len(train_losses))
        
        # Plot with log scale on both axes
        plt.loglog(train_steps, train_losses, label='Training Loss', alpha=0.7)
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
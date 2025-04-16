#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description='Compare validation losses for different model architectures')
    parser.add_argument('--shallow_model', type=str, default=None, 
                        help='Base model name for shallow model (without extension). If None, will use the first available.')
    parser.add_argument('--medium_model', type=str, default=None, 
                        help='Base model name for medium model (without extension). If None, will use the first available.')
    parser.add_argument('--deep_model', type=str, default=None, 
                        help='Base model name for deep model (without extension). If None, will use the first available.')
    args = parser.parse_args()
    
    # Directory paths
    shallow_dir = "unet_shallow"
    medium_dir = "unet_medium"
    deep_dir = "unet_deep"
    
    # Dictionary to store model names and losses
    models = {
        "shallow": {"dir": shallow_dir, "name": args.shallow_model},
        "medium": {"dir": medium_dir, "name": args.medium_model},
        "deep": {"dir": deep_dir, "name": args.deep_model}
    }
    
    # Find model files and load validation losses
    for model_type, model_info in models.items():
        # Get directory
        model_dir = model_info["dir"]
        
        # Find the model name if not specified
        if model_info["name"] is None:
            # Try to find any valid loss files
            train_files = glob.glob(f"{model_dir}/*_valid_losses.npy")
            if train_files:
                # Extract base model name from first found file
                file_name = os.path.basename(train_files[0])
                model_name = file_name.replace('_valid_losses.npy', '')
                model_info["name"] = model_name
                print(f"Using {model_type} model: {model_name}")
            else:
                print(f"Error: No loss files found for {model_type} model in {model_dir}.")
                sys.exit(1)
        
        # Path to validation loss file
        valid_loss_file = os.path.join(model_dir, f"{model_info['name']}_valid_losses.npy")
        
        # Check if file exists
        if not os.path.exists(valid_loss_file):
            print(f"Error: {valid_loss_file} not found.")
            sys.exit(1)
        
        # Load validation losses
        valid_losses = np.load(valid_loss_file)
        model_info["valid_losses"] = valid_losses
        
        # Calculate minimum validation loss
        min_valid_loss = np.min(valid_losses)
        min_valid_epoch = np.argmin(valid_losses)
        model_info["min_loss"] = min_valid_loss
        model_info["min_epoch"] = min_valid_epoch
    
    # Plot with LaTeX rendering
    try:
        # Enable LaTeX rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif"
        })
        
        # Create figure with log scale on y-axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot validation losses
        for model_type, model_info in models.items():
            valid_losses = model_info["valid_losses"]
            epochs = np.arange(1, len(valid_losses) + 1)
            
            if model_type == "shallow":
                label = r"Shallow UNet (2 levels, 2 layers/block)"
                marker = 'o'
                color = 'blue'
            elif model_type == "medium":
                label = r"Medium UNet (3 levels, 3 layers/block)"
                marker = 's'
                color = 'green'
            else:  # deep
                label = r"Deep UNet (4 levels, 4 layers/block)"
                marker = '^'
                color = 'red'
            
            # Plot with log scale on y-axis
            ax.semilogy(epochs, valid_losses, label=label, marker=marker, 
                       linestyle='-', color=color, markersize=8)
            
            # Mark minimum validation loss
            min_loss = model_info["min_loss"]
            min_epoch = model_info["min_epoch"]
            ax.scatter(min_epoch + 1, min_loss, color=color, s=150, 
                      marker='*', zorder=10, 
                      label=f"{model_type.capitalize()} Min: {min_loss:.6f} (Epoch {min_epoch+1})")
        
        # Set labels and title
        ax.set_xlabel(r'Epoch')
        ax.set_ylabel(r'Validation Loss (log scale)')
        ax.set_title(r'Validation Loss Comparison Across Model Architectures')
        
        # Add grid and legend
        ax.grid(True, which="both", ls="--", alpha=0.7)
        ax.legend(loc='best')
        
        # Ensure entire plot is visible
        plt.tight_layout()
        
        # Save with LaTeX rendering
        plt.savefig('model_comparison_latex.png', dpi=300, bbox_inches='tight')
        print("Model comparison with LaTeX rendering saved as model_comparison_latex.png")
        
    except Exception as e:
        print(f"Error with LaTeX rendering: {e}")
        print("Falling back to standard rendering...")
        
        # Reset matplotlib parameters
        plt.rcParams.update({"text.usetex": False})
        plt.figure(figsize=(10, 6))
        
        # Plot validation losses
        for model_type, model_info in models.items():
            valid_losses = model_info["valid_losses"]
            epochs = np.arange(1, len(valid_losses) + 1)
            
            if model_type == "shallow":
                label = "Shallow UNet (2 levels, 2 layers/block)"
                marker = 'o'
                color = 'blue'
            elif model_type == "medium":
                label = "Medium UNet (3 levels, 3 layers/block)"
                marker = 's'
                color = 'green'
            else:  # deep
                label = "Deep UNet (4 levels, 4 layers/block)"
                marker = '^'
                color = 'red'
            
            # Plot with log scale on y-axis
            plt.semilogy(epochs, valid_losses, label=label, marker=marker, 
                        linestyle='-', color=color, markersize=8)
            
            # Mark minimum validation loss
            min_loss = model_info["min_loss"]
            min_epoch = model_info["min_epoch"]
            plt.scatter(min_epoch + 1, min_loss, color=color, s=150, 
                       marker='*', zorder=10, 
                       label=f"{model_type.capitalize()} Min: {min_loss:.6f} (Epoch {min_epoch+1})")
        
        # Set labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss (log scale)')
        plt.title('Validation Loss Comparison Across Model Architectures')
        
        # Add grid and legend
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.legend(loc='best')
        
        # Save with standard rendering
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("Model comparison saved as model_comparison.png")

if __name__ == "__main__":
    main() 
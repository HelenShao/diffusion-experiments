import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json

def plot_losses(save_directory):
    """Loads loss history arrays from a directory and plots them."""
    epochs_path = os.path.join(save_directory, "epochs.npy")
    train_losses_path = os.path.join(save_directory, "train_losses.npy")
    valid_losses_path = os.path.join(save_directory, "valid_losses.npy")
    test_losses_path = os.path.join(save_directory, "test_losses.npy")
    config_path = os.path.join(save_directory, "config.json")

    if not all(os.path.exists(p) for p in [epochs_path, train_losses_path, valid_losses_path, test_losses_path, config_path]):
        print("Error: One or more required files (epochs.npy, train_losses.npy, valid_losses.npy, test_losses.npy, config.json) not found in the specified directory.")
        return

    epochs = np.load(epochs_path)
    train_losses = np.load(train_losses_path)
    valid_losses = np.load(valid_losses_path)
    test_losses = np.load(test_losses_path)

    # Load config to get min validation loss and debug factor
    with open(config_path, 'r') as f:
        config = json.load(f)
    min_valid_loss = config.get('min_valid_loss', 'N/A')
    # Assuming debug factor might be relevant, load it if present
    # debug_factor = config.get('debug_factor', 1.0) # Get debug factor if needed, default to 1.0

    plt.figure(figsize=(12, 6))
    plt.plot(epochs + 1, train_losses, label='Training Loss') # Add 1 to epochs for 1-based plotting
    plt.plot(epochs + 1, valid_losses, label=f'Validation Loss (Min: {min_valid_loss:.3e})')
    plt.plot(epochs + 1, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training, Validation, and Test Loss') # Updated title
    plt.grid(True)
    plt.legend()
    plt.yscale('log') # Use log scale for potentially large loss ranges
    
    plot_filename = os.path.join(save_directory, "loss_history_plot.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training, validation, and test loss from saved numpy arrays.")
    parser.add_argument("save_directory", type=str, help="Directory containing the saved loss history .npy files and config.json.")
    args = parser.parse_args()

    plot_losses(args.save_directory) 
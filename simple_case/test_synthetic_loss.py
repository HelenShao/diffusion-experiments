import numpy as np
import matplotlib.pyplot as plt
import os

# Generate synthetic loss data that mimics training logs
n_epochs = 30
epochs = np.arange(n_epochs)

# Generate synthetic loss values that start high and decrease
np.random.seed(42)  # For reproducibility
initial_loss = 0.8

# Training loss: starts at initial_loss and decreases with noise
train_loss = initial_loss * np.exp(-0.1 * epochs) + 0.05 * np.random.rand(n_epochs)

# Validation loss: similar pattern but slightly higher with different noise
valid_loss = initial_loss * np.exp(-0.08 * epochs) + 0.08 * np.random.rand(n_epochs)

# Test loss: similar to validation
test_loss = valid_loss + 0.02 * np.random.rand(n_epochs)

# Combine into a single array
synthetic_data = np.column_stack((epochs, train_loss, valid_loss, test_loss))

# Save to file with header (like the actual training_log.txt)
header = "Epoch,Train,Valid,Test"
np.savetxt('synthetic_loss.txt', synthetic_data, delimiter=',', header=header, comments='')
print(f"Synthetic loss data saved to synthetic_loss.txt")

# Display the first few rows of synthetic data
print("\nFirst 5 rows of synthetic data:")
with open('synthetic_loss.txt', 'r') as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 5:
            break

# Now try to plot the synthetic data with LaTeX rendering
print("\nAttempting to plot with LaTeX rendering...")

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

try:
    # Create a plot
    plt.figure(figsize=(10, 6))
    
    # Plot all three loss curves
    plt.plot(synthetic_data[:, 0], synthetic_data[:, 1], label=r'Training Loss')
    plt.plot(synthetic_data[:, 0], synthetic_data[:, 2], label=r'Validation Loss')
    plt.plot(synthetic_data[:, 0], synthetic_data[:, 3], label=r'Test Loss')
    
    plt.title(r'Synthetic Training Losses')
    plt.xlabel(r'Epoch')
    plt.ylabel(r'Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save and show the plot
    plt.savefig('synthetic_loss_plot.png')
    print("Plotting successful! Saved as synthetic_loss_plot.png")
    plt.show()
    
except Exception as e:
    print(f"Error plotting with LaTeX: {e}")
    
    # Try without LaTeX as a fallback
    print("\nTrying again without LaTeX rendering...")
    plt.rcParams.update({"text.usetex": False})
    
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(synthetic_data[:, 0], synthetic_data[:, 1], label='Training Loss')
        plt.plot(synthetic_data[:, 0], synthetic_data[:, 2], label='Validation Loss')
        plt.plot(synthetic_data[:, 0], synthetic_data[:, 3], label='Test Loss')
        
        plt.title('Synthetic Training Losses (without LaTeX)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('synthetic_loss_plot_no_latex.png')
        print("Non-LaTeX plotting successful! Saved as synthetic_loss_plot_no_latex.png")
        plt.show()
        
    except Exception as e2:
        print(f"Error plotting without LaTeX: {e2}") 
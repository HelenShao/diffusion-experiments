import os
import numpy as np
import matplotlib
# Force non-interactive backend
matplotlib.use('Agg')

# Try to set font to Times New Roman with fallback
try:
    # Check if Times New Roman is available
    import matplotlib.font_manager as fm
    font_names = [f.name for f in fm.fontManager.ttflist]
    if 'Times New Roman' in font_names:
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['Times New Roman'] + matplotlib.rcParams['font.serif']
        print("Successfully set Times New Roman font.")
    else:
        print("Times New Roman font not found, using default serif font.")
        matplotlib.rcParams['font.family'] = 'serif'
except Exception as e:
    print(f"Error setting font: {e}. Using default.")
    
# Disable LaTeX
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Path to generated samples
samples_dir = '/scratch/gpfs/hshao/ILC_ML/1_freq_scale/220_dust_only/d9/d9_ell_200_no_norm/diffusion/d12_rm_UNet2DModel/d12_rm_normalize_ddpm_wg_UNet2DModel/generated_samples'
output_dir = os.path.join(samples_dir, 'plots')
os.makedirs(output_dir, exist_ok=True)

# Number of samples to plot
num_samples = 10  # Adjust as needed

print(f"Plotting {num_samples} samples...")

# Plot each sample
for i in range(num_samples):
    try:
        # Load the numpy arrays
        small_scale = np.load(f"{samples_dir}/sample_{i}_small_scale.npy")
        ground_truth = np.load(f"{samples_dir}/sample_{i}_ground_truth.npy")
        generated = np.load(f"{samples_dir}/sample_{i}_generated.npy")
        
        # Squeeze arrays to remove singleton dimensions if necessary
        small_scale = small_scale.squeeze()
        ground_truth = ground_truth.squeeze()
        generated = generated.squeeze()
        
        # Create a figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(wspace=0.3)  # Add space between subplots for colorbars
        
        # Plot small scale conditioning
        im0 = axes[0].imshow(small_scale, cmap='RdYlBu')
        axes[0].set_title("Small Scale")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        divider = make_axes_locatable(axes[0])
        cax0 = divider.append_axes("right", size="5%", pad=0.05)
        cbar0 = fig.colorbar(im0, cax=cax0)
        
        # Plot generated large scale
        mse = mean_squared_error(ground_truth.flatten(), generated.flatten())
        corr = np.corrcoef(ground_truth.flatten(), generated.flatten())[0, 1]
        im1 = axes[1].imshow(generated, cmap='RdYlBu')
        axes[1].set_title(f"Generated\nMSE: {mse:.2e}\nCorr: {corr:.3f}")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        divider = make_axes_locatable(axes[1])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = fig.colorbar(im1, cax=cax1)
        
        # Plot ground truth
        im2 = axes[2].imshow(ground_truth, cmap='RdYlBu')
        axes[2].set_title("Ground Truth")
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        divider = make_axes_locatable(axes[2])
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar2 = fig.colorbar(im2, cax=cax2)
        
        # Save the figure
        plt.savefig(f"{output_dir}/sample_{i}_comparison.png")
        plt.close(fig)
        
        print(f"Plotted sample {i}")
        
    except FileNotFoundError as e:
        print(f"Could not find files for sample {i}: {e}")
        break
    except Exception as e:
        print(f"Error plotting sample {i}: {e}")
        continue

# Also plot the denoising steps if available
denoising_dir = os.path.join(samples_dir, "denoising_steps")
if os.path.exists(denoising_dir):
    print("Plotting denoising steps...")
    
    # Get all denoising step files
    step_files = [f for f in os.listdir(denoising_dir) if f.endswith('.npy')]
    
    # Sort by step number
    step_files.sort(key=lambda x: int(x.split('_')[1]))  # Sort by step number
    
    if step_files:
        print(f"Found {len(step_files)} denoising steps files")
        
        # Get the number of steps
        num_steps = len(step_files)
        
        # Layout: Main grid for steps with special placement for reference images
        # We'll create custom figure layout
        fig = plt.figure(figsize=(3 * (num_steps + 1), 8))
        
        # Add a suptitle
        fig.suptitle(f"Denoising Process", fontsize=16)
        
        # Try to load a sample for reference images
        try:
            sample_id = 0  # Use the first sample for reference
            ground_truth = np.load(f"{samples_dir}/sample_{sample_id}_ground_truth.npy").squeeze()
            small_scale = np.load(f"{samples_dir}/sample_{sample_id}_small_scale.npy").squeeze()
            
            # Create grid with custom layout
            # Main area for denoising steps
            gs_main = fig.add_gridspec(1, num_steps, left=0.05, right=0.75, top=0.85, bottom=0.4)
            
            # Area for ground truth (to right of step 9)
            gs_truth = fig.add_gridspec(1, 1, left=0.8, right=0.95, top=0.85, bottom=0.4)
            
            # Area for small scale (under ground truth)
            gs_small = fig.add_gridspec(1, 1, left=0.8, right=0.95, top=0.3, bottom=0.05)
            
            # Plot each denoising step in the main area
            for i, file in enumerate(step_files):
                # Load the step image
                step_img = np.load(os.path.join(denoising_dir, file))
                step_img = step_img.squeeze()
                
                # Extract step and timestep info from filename (format: step_X_t_Y.npy)
                parts = file.split('_')
                step_num = parts[1]
                timestep = parts[3].split('.')[0]
                
                # Plot the step
                ax = fig.add_subplot(gs_main[0, i])
                im = ax.imshow(step_img, cmap='RdYlBu')
                ax.set_title(f"Step {step_num} (t={timestep})")
                ax.set_xticks([])
                ax.set_yticks([])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)
            
            # Plot ground truth to the right of steps
            ax_truth = fig.add_subplot(gs_truth[0, 0])
            im_truth = ax_truth.imshow(ground_truth, cmap='RdYlBu')
            ax_truth.set_title("True Large Scales")
            ax_truth.set_xticks([])
            ax_truth.set_yticks([])
            divider = make_axes_locatable(ax_truth)
            cax_truth = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_truth, cax=cax_truth)
            
            # Plot small scale underneath the ground truth
            ax_small = fig.add_subplot(gs_small[0, 0])
            im_small = ax_small.imshow(small_scale, cmap='RdYlBu')
            ax_small.set_title("Small Scale")
            ax_small.set_xticks([])
            ax_small.set_yticks([])
            divider = make_axes_locatable(ax_small)
            cax_small = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_small, cax=cax_small)
            
        except FileNotFoundError as e:
            print(f"Could not find reference files: {e}")
        
        # Save the figure
        plt.savefig(f"{output_dir}/denoising_process.png", dpi=150)
        plt.close(fig)
        
        print(f"Plotted all denoising steps in one figure")

print(f"All plots saved to {output_dir}")

# Create a comparison grid of multiple samples
print("Creating comparison grid...")

# Number of samples to include in grid
grid_samples = min(5, num_samples)
fig, axes = plt.subplots(grid_samples, 3, figsize=(15, grid_samples*4))
plt.subplots_adjust(wspace=0.3)  # Add space between subplots for colorbars

for i in range(grid_samples):
    try:
        # Load the numpy arrays
        small_scale = np.load(f"{samples_dir}/sample_{i}_small_scale.npy")
        ground_truth = np.load(f"{samples_dir}/sample_{i}_ground_truth.npy")
        generated = np.load(f"{samples_dir}/sample_{i}_generated.npy")
        
        # Squeeze arrays
        small_scale = small_scale.squeeze()
        ground_truth = ground_truth.squeeze()
        generated = generated.squeeze()
        
        # Plot in grid
        im0 = axes[i, 0].imshow(small_scale, cmap='RdYlBu')
        axes[i, 0].set_title(f"Sample {i}: Small Scale")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        divider = make_axes_locatable(axes[i, 0])
        cax0 = divider.append_axes("right", size="5%", pad=0.05)
        cbar0 = fig.colorbar(im0, cax=cax0)
        
        mse = mean_squared_error(ground_truth.flatten(), generated.flatten())
        corr = np.corrcoef(ground_truth.flatten(), generated.flatten())[0, 1]
        im1 = axes[i, 1].imshow(generated, cmap='RdYlBu')
        axes[i, 1].set_title(f"Sample {i}: Generated\nMSE: {mse:.2e}\nCorr: {corr:.3f}")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        divider = make_axes_locatable(axes[i, 1])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = fig.colorbar(im1, cax=cax1)
        
        im2 = axes[i, 2].imshow(ground_truth, cmap='RdYlBu')
        axes[i, 2].set_title(f"Sample {i}: Ground Truth")
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        divider = make_axes_locatable(axes[i, 2])
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar2 = fig.colorbar(im2, cax=cax2)
        
    except FileNotFoundError:
        continue

# Set column titles
plt.tight_layout()
plt.savefig(f"{output_dir}/comparison_grid.png", dpi=150)
plt.close(fig)
print(f"Saved comparison grid to {output_dir}/comparison_grid.png")

print("Plotting complete!") 
import os
import numpy as np
import torch
import matplotlib
# Force Agg backend which doesn't require GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import sys
# Global configurations
import warnings

# Completely disable LaTeX everywhere
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
warnings.filterwarnings("ignore")

# Add paths to import from parent modules
sys.path.extend([
    os.path.abspath("/scratch/gpfs/hshao/PartIII-Project"),
    "/scratch/gpfs/hshao/projectron/",
    "/scratch/gpfs/hshao/ILC_ML/reproducing/unet_vanilla_4_freq"
])

# Set output directory for plots
output_dir = "data_analysis"
os.makedirs(output_dir, exist_ok=True)

def analyze_dataset(loader, dataset_name="train"):
    """Analyze the small and large scale data distributions."""
    print(f"Analyzing {dataset_name} dataset...")
    
    # Access global variables from load_data.py
    global dust_model, gal_inclusion
    
    # Collect statistics across batches
    small_scale_pixels = []
    large_scale_pixels = []
    
    # Per-image statistics
    small_scale_stats = {'min': [], 'max': [], 'mean': [], 'std': [], 'dynamic_range': []}
    large_scale_stats = {'min': [], 'max': [], 'mean': [], 'std': [], 'dynamic_range': []}
    
    # Sample patches for visualization
    sample_small_scales = []
    sample_large_scales = []
    num_samples = 10
    
    for i, (small_scale, large_scale) in enumerate(tqdm(loader)):
        # Convert to numpy for easier handling
        small_scale_np = small_scale.detach().cpu().numpy()
        large_scale_np = large_scale.detach().cpu().numpy()
        
        # Collect all pixel values for overall distribution
        small_scale_pixels.append(small_scale_np.flatten())
        large_scale_pixels.append(large_scale_np.flatten())
        
        # Collect per-image statistics
        for j in range(small_scale_np.shape[0]):
            # Small scale stats
            small_img = small_scale_np[j]
            small_scale_stats['min'].append(small_img.min())
            small_scale_stats['max'].append(small_img.max())
            small_scale_stats['mean'].append(small_img.mean())
            small_scale_stats['std'].append(small_img.std())
            small_scale_stats['dynamic_range'].append(small_img.max() - small_img.min())
            
            # Large scale stats
            large_img = large_scale_np[j]
            large_scale_stats['min'].append(large_img.min())
            large_scale_stats['max'].append(large_img.max())
            large_scale_stats['mean'].append(large_img.mean())
            large_scale_stats['std'].append(large_img.std())
            large_scale_stats['dynamic_range'].append(large_img.max() - large_img.min())
        
        # Sample some images for visualization
        if i == 0:
            max_samples = min(num_samples, small_scale_np.shape[0])
            sample_small_scales = small_scale_np[:max_samples]
            sample_large_scales = large_scale_np[:max_samples]
        
        # Limit the number of batches to process to avoid memory issues
        # if i >= 20:  # Process 20 batches at most
        #     break
    
    # Combine all pixel values
    small_scale_pixels = np.concatenate(small_scale_pixels)
    large_scale_pixels = np.concatenate(large_scale_pixels)
    
    # General statistics
    print("\nOverall Statistics:")
    print("Small Scale (Conditioning):")
    print(f"  Min: {small_scale_pixels.min():.6f}, Max: {small_scale_pixels.max():.6f}")
    print(f"  Mean: {small_scale_pixels.mean():.6f}, Std: {small_scale_pixels.std():.6f}")
    print(f"  Dynamic Range: {small_scale_pixels.max() - small_scale_pixels.min():.6f}")
    print(f"  5th/95th percentiles: {np.percentile(small_scale_pixels, 5):.6f} / {np.percentile(small_scale_pixels, 95):.6f}")
    
    print("\nLarge Scale (Target):")
    print(f"  Min: {large_scale_pixels.min():.6f}, Max: {large_scale_pixels.max():.6f}")
    print(f"  Mean: {large_scale_pixels.mean():.6f}, Std: {large_scale_pixels.std():.6f}")
    print(f"  Dynamic Range: {large_scale_pixels.max() - large_scale_pixels.min():.6f}")
    print(f"  5th/95th percentiles: {np.percentile(large_scale_pixels, 5):.6f} / {np.percentile(large_scale_pixels, 95):.6f}")
    
    try:
        # Calculate correlation between small and large scales
        per_image_correlations = []
        for i in range(len(small_scale_stats['mean'])):
            per_image_correlations.append({
                'small_mean': small_scale_stats['mean'][i],
                'small_std': small_scale_stats['std'][i],
                'large_mean': large_scale_stats['mean'][i],
                'large_std': large_scale_stats['std'][i],
                'dynamic_ratio': small_scale_stats['dynamic_range'][i] / large_scale_stats['dynamic_range'][i]
                if large_scale_stats['dynamic_range'][i] != 0 else 0
            })
        
        # Save statistics to text file
        with open(f"{output_dir}/{dataset_name}_statistics_{dust_model}_{gal_inclusion}.txt", "w") as f:
            f.write(f"Small Scale (Conditioning):\n")
            f.write(f"  Min: {small_scale_pixels.min():.6f}, Max: {small_scale_pixels.max():.6f}\n")
            f.write(f"  Mean: {small_scale_pixels.mean():.6f}, Std: {small_scale_pixels.std():.6f}\n")
            f.write(f"  Dynamic Range: {small_scale_pixels.max() - small_scale_pixels.min():.6f}\n")
            f.write(f"  5th/95th percentiles: {np.percentile(small_scale_pixels, 5):.6f} / {np.percentile(small_scale_pixels, 95):.6f}\n\n")
            
            f.write(f"Large Scale (Target):\n")
            f.write(f"  Min: {large_scale_pixels.min():.6f}, Max: {large_scale_pixels.max():.6f}\n")
            f.write(f"  Mean: {large_scale_pixels.mean():.6f}, Std: {large_scale_pixels.std():.6f}\n")
            f.write(f"  Dynamic Range: {large_scale_pixels.max() - large_scale_pixels.min():.6f}\n")
            f.write(f"  5th/95th percentiles: {np.percentile(large_scale_pixels, 5):.6f} / {np.percentile(large_scale_pixels, 95):.6f}\n")
        
        # Plots
        # 1. Histogram of pixel values
        plt.figure(figsize=(15, 7))
        
        plt.subplot(1, 2, 1)
        plt.hist(small_scale_pixels, bins=100, alpha=0.7, label='Small Scale')
        plt.title('Small Scale Pixel Value Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(large_scale_pixels, bins=100, alpha=0.7, label='Large Scale')
        plt.title('Large Scale Pixel Value Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{dust_model}_{gal_inclusion}_{dataset_name}_pixel_distribution.png", dpi=300)
        plt.close()
        
        # 2. Simplified comparison of distributions with log scale
        plt.figure(figsize=(10, 6))
        # Use alpha for the histograms to make them more visible
        counts1, bins1, _ = plt.hist(small_scale_pixels, bins=100, alpha=0.5, label='Small Scale', log=True)
        counts2, bins2, _ = plt.hist(large_scale_pixels, bins=100, alpha=0.5, label='Large Scale', log=True)
        plt.title('Pixel Value Distribution Comparison (Log Scale)')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency (log scale)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{output_dir}/{dust_model}_{gal_inclusion}_{dataset_name}_pixel_distribution_comparison_log.png", dpi=300)
        plt.close()
        
        # 3. Per-Image Statistics
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.scatter([x['small_mean'] for x in per_image_correlations], 
                    [x['large_mean'] for x in per_image_correlations], alpha=0.5)
        plt.title('Mean Value Correlation')
        plt.xlabel('Small Scale Mean')
        plt.ylabel('Large Scale Mean')
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.scatter([x['small_std'] for x in per_image_correlations], 
                    [x['large_std'] for x in per_image_correlations], alpha=0.5)
        plt.title('Standard Deviation Correlation')
        plt.xlabel('Small Scale Std')
        plt.ylabel('Large Scale Std')
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.hist([x['dynamic_ratio'] for x in per_image_correlations], bins=30, alpha=0.7)
        plt.title('Dynamic Range Ratio (Small/Large)')
        plt.xlabel('Ratio')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{dataset_name}_statistical_correlations.png", dpi=300)
        plt.close()
        
        # 4. Visualize sample patches
        try:
            fig, axes = plt.subplots(2, num_samples, figsize=(20, 5))
            
            for i in range(num_samples):
                # Small scale
                axes[0, i].imshow(sample_small_scales[i][0], cmap='viridis')
                axes[0, i].set_title(f"Small #{i+1}")
                axes[0, i].axis('off')
                
                # Large scale
                axes[1, i].imshow(sample_large_scales[i][0], cmap='viridis')
                axes[1, i].set_title(f"Large #{i+1}")
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{dataset_name}_sample_patches.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create sample patches visualization: {e}")
    
    except Exception as e:
        print(f"Warning: Error in plotting: {e}")
        # Continue execution even if plotting fails
    
    return {
        'small': {
            'min': small_scale_pixels.min(),
            'max': small_scale_pixels.max(),
            'mean': small_scale_pixels.mean(),
            'std': small_scale_pixels.std(),
            '5th': np.percentile(small_scale_pixels, 5),
            '95th': np.percentile(small_scale_pixels, 95)
        },
        'large': {
            'min': large_scale_pixels.min(),
            'max': large_scale_pixels.max(),
            'mean': large_scale_pixels.mean(),
            'std': large_scale_pixels.std(),
            '5th': np.percentile(large_scale_pixels, 5),
            '95th': np.percentile(large_scale_pixels, 95)
        }
    }

def recommend_normalization(stats):
    """Provide recommendations for normalization based on data statistics."""
    print("\n=== Normalization Recommendations ===")
    
    small_range = stats['small']['max'] - stats['small']['min']
    large_range = stats['large']['max'] - stats['large']['min']
    
    print(f"Small scale dynamic range: {small_range:.6f}")
    print(f"Large scale dynamic range: {large_range:.6f}")
    
    if small_range > 10 * large_range:
        print("\nWARNING: Small scale data has much larger dynamic range than large scale data.")
        print("This could make conditioning difficult as small changes in the conditioning are drowned out.")
        print("\nRecommendation: Normalize the small scale data to a similar range as the large scale data.")
        
        # Suggest normalization code
        print("\nSuggested normalization code:")
        print("```python")
        print("# Option 1: Min-max normalization to [0, 1] range")
        print("def normalize_minmax(x):")
        print("    return (x - x.min()) / (x.max() - x.min() + 1e-8)")
        print("\n# Option 2: Standardization to mean=0, std=1")
        print("def normalize_standard(x):")
        print("    return (x - x.mean()) / (x.std() + 1e-8)")
        print("\n# Option 3: Percentile-based normalization to reduce impact of outliers")
        print("def normalize_percentile(x, lower=5, upper=95):")
        print("    low, high = np.percentile(x, [lower, upper])")
        print("    return np.clip((x - low) / (high - low + 1e-8), 0, 1)")
        print("```")
        
    elif small_range < 0.1 * large_range:
        print("\nWARNING: Small scale data has much smaller dynamic range than large scale data.")
        print("This could make conditioning ineffective as the conditioning signal is too weak.")
        print("\nRecommendation: Amplify the small scale data to a similar range as the large scale data.")
    
    if abs(stats['small']['mean']) > 10 * abs(stats['large']['mean']):
        print("\nWARNING: Small scale data has a much larger mean than large scale data.")
        print("Consider centering both datasets to have similar means.")
    
    print("\nFor best results with conditional diffusion models, consider:")
    print("1. Using the same normalization scheme for both small and large scale data")
    print("2. Ensuring similar dynamic ranges to make conditioning effective")
    print("3. Optionally applying instance normalization (per-patch) if global statistics vary greatly")
    print("4. Consider using log-scale normalization if the data has a large dynamic range")

if __name__ == "__main__":
    print("Loading dataloaders...")
    # Get data loaders - using exec to maintain the original behavior
    # This makes variables defined in load_data.py available in the current scope
    with open("load_data.py") as f:
        exec(f.read())
    
    # Analyze training data
    train_stats = analyze_dataset(train_loader, "train")
    
    # Analyze validation data
    valid_stats = analyze_dataset(valid_loader, "valid")
    
    # Make recommendations based on training data statistics
    recommend_normalization(train_stats)
    
    print("\nAnalysis complete! Check the 'data_analysis' directory for visualizations.") 
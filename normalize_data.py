import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import argparse
import random
from tqdm import tqdm

# Completely disable LaTeX everywhere
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
warnings.filterwarnings("ignore")

# Add paths
sys.path.extend([
    os.path.abspath("/scratch/gpfs/hshao/PartIII-Project"),
    "/scratch/gpfs/hshao/projectron/",
    "/scratch/gpfs/hshao/ILC_ML/reproducing/unet_vanilla_4_freq"
])

# Get data loaders    
with open("load_data.py") as f:
    exec(f.read())
    
class Normalizer:
    """Simple normalizer class for data normalization."""
    def __init__(self, method="standard", percentile_range=None):
        self.method = method
        self.percentile_range = percentile_range or (1, 99)
        self.small_scale_params = {}
        self.large_scale_params = {}
        self.is_fit = False
        
    def fit(self, small_scale_data, large_scale_data):
        """Fit the normalizer parameters to the data."""
        # Convert to numpy arrays if tensors
        if isinstance(small_scale_data, torch.Tensor):
            small_scale_data = small_scale_data.detach().cpu().numpy()
        if isinstance(large_scale_data, torch.Tensor):
            large_scale_data = large_scale_data.detach().cpu().numpy()
            
        if self.method == "minmax":
            self.small_scale_params = {
                "min": float(np.min(small_scale_data)),
                "max": float(np.max(small_scale_data))
            }
            self.large_scale_params = {
                "min": float(np.min(large_scale_data)),
                "max": float(np.max(large_scale_data))
            }
        elif self.method == "standard":
            self.small_scale_params = {
                "mean": float(np.mean(small_scale_data)),
                "std": float(np.std(small_scale_data))
            }
            self.large_scale_params = {
                "mean": float(np.mean(large_scale_data)),
                "std": float(np.std(large_scale_data))
            }
        elif self.method == "percentile":
            self.small_scale_params = {
                "low": float(np.percentile(small_scale_data, self.percentile_range[0])),
                "high": float(np.percentile(small_scale_data, self.percentile_range[1]))
            }
            self.large_scale_params = {
                "low": float(np.percentile(large_scale_data, self.percentile_range[0])),
                "high": float(np.percentile(large_scale_data, self.percentile_range[1]))
            }
        elif self.method == "match_range":
            # Match the range of small_scale to large_scale
            small_min = float(np.min(small_scale_data))
            small_max = float(np.max(small_scale_data))
            large_min = float(np.min(large_scale_data))
            large_max = float(np.max(large_scale_data))
            
            self.small_scale_params = {
                "min": small_min,
                "max": small_max,
                "target_min": large_min,
                "target_max": large_max
            }
            self.large_scale_params = {
                "min": large_min,
                "max": large_max
            }
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
            
        self.is_fit = True
        return self
    
    def normalize_small_scale(self, x):
        """Normalize small scale data."""
        if not self.is_fit:
            raise ValueError("Normalizer not fit. Call fit() first.")
            
        # Convert to tensor if numpy array
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.tensor(x)
            
        if self.method == "minmax":
            normalized = (x - self.small_scale_params["min"]) / (self.small_scale_params["max"] - self.small_scale_params["min"] + 1e-8)
        elif self.method == "standard":
            normalized = (x - self.small_scale_params["mean"]) / (self.small_scale_params["std"] + 1e-8)
        elif self.method == "percentile":
            normalized = (x - self.small_scale_params["low"]) / (self.small_scale_params["high"] - self.small_scale_params["low"] + 1e-8)
            normalized = torch.clamp(normalized, 0, 1)
        elif self.method == "match_range":
            # Scale small_scale to match large_scale range
            small_range = self.small_scale_params["max"] - self.small_scale_params["min"] + 1e-8
            large_range = self.small_scale_params["target_max"] - self.small_scale_params["target_min"] + 1e-8
            normalized = (x - self.small_scale_params["min"]) / small_range
            normalized = normalized * large_range + self.small_scale_params["target_min"]
            
        return normalized.numpy() if is_numpy else normalized
    
    def normalize_large_scale(self, x):
        """Normalize large scale data."""
        if not self.is_fit:
            raise ValueError("Normalizer not fit. Call fit() first.")
            
        # Convert to tensor if numpy array
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.tensor(x)
            
        if self.method == "minmax":
            normalized = (x - self.large_scale_params["min"]) / (self.large_scale_params["max"] - self.large_scale_params["min"] + 1e-8)
        elif self.method == "standard":
            normalized = (x - self.large_scale_params["mean"]) / (self.large_scale_params["std"] + 1e-8)
        elif self.method == "percentile":
            normalized = (x - self.large_scale_params["low"]) / (self.large_scale_params["high"] - self.large_scale_params["low"] + 1e-8)
            normalized = torch.clamp(normalized, 0, 1)
        elif self.method == "match_range":
            # No change needed for large_scale in match_range
            normalized = x
            
        return normalized.numpy() if is_numpy else normalized
    
    def save(self, filepath):
        """Save normalizer parameters to file."""
        if not self.is_fit:
            raise ValueError("Normalizer not fit. Call fit() first.")
            
        params = {
            "method": self.method,
            "percentile_range": self.percentile_range,
            "small_scale_params": self.small_scale_params,
            "large_scale_params": self.large_scale_params
        }
        torch.save(params, filepath)
        print(f"Normalizer saved to {filepath}")
        
    @classmethod
    def load(cls, filepath):
        """Load normalizer from file."""
        params = torch.load(filepath)
        normalizer = cls(
            method=params["method"],
            percentile_range=params["percentile_range"]
        )
        normalizer.small_scale_params = params["small_scale_params"]
        normalizer.large_scale_params = params["large_scale_params"]
        normalizer.is_fit = True
        return normalizer
    
    def __str__(self):
        """String representation with parameters."""
        if not self.is_fit:
            return f"Normalizer(method={self.method}, not fit)"
            
        if self.method == "minmax":
            small_info = f"min={self.small_scale_params['min']:.4f}, max={self.small_scale_params['max']:.4f}"
            large_info = f"min={self.large_scale_params['min']:.4f}, max={self.large_scale_params['max']:.4f}"
        elif self.method == "standard":
            small_info = f"mean={self.small_scale_params['mean']:.4f}, std={self.small_scale_params['std']:.4f}"
            large_info = f"mean={self.large_scale_params['mean']:.4f}, std={self.large_scale_params['std']:.4f}"
        elif self.method == "percentile":
            small_info = f"p{self.percentile_range[0]}={self.small_scale_params['low']:.4f}, p{self.percentile_range[1]}={self.small_scale_params['high']:.4f}"
            large_info = f"p{self.percentile_range[0]}={self.large_scale_params['low']:.4f}, p{self.percentile_range[1]}={self.large_scale_params['high']:.4f}"
        elif self.method == "match_range":
            small_info = f"min={self.small_scale_params['min']:.4f}, max={self.small_scale_params['max']:.4f}, target_min={self.small_scale_params['target_min']:.4f}, target_max={self.small_scale_params['target_max']:.4f}"
            large_info = f"min={self.large_scale_params['min']:.4f}, max={self.large_scale_params['max']:.4f}"
            
        return f"Normalizer(method={self.method}, small_scale: {small_info}, large_scale: {large_info})"

class NormalizedDataLoader:
    """Wrapper for data loader that applies normalization."""
    def __init__(self, data_loader, normalizer):
        self.data_loader = data_loader
        self.normalizer = normalizer
        
    def __iter__(self):
        for small_scale, large_scale in self.data_loader:
            yield (
                self.normalizer.normalize_small_scale(small_scale),
                self.normalizer.normalize_large_scale(large_scale)
            )
            
    def __len__(self):
        return len(self.data_loader)

def collect_batch_data(data_loader, max_batches=20):
    """Collect data from loader into tensors."""
    small_scales = []
    large_scales = []
    
    print("Collecting data for normalization...")
    for i, (small_scale, large_scale) in enumerate(tqdm(data_loader)):
        small_scales.append(small_scale)
        large_scales.append(large_scale)
        
        if i >= max_batches - 1:
            break
            
    # Concatenate all collected batches
    all_small_scale = torch.cat(small_scales)
    all_large_scale = torch.cat(large_scales)
    
    return all_small_scale, all_large_scale

def visualize_normalization(small_scale, large_scale, normalized_small, normalized_large, output_dir="normalization_viz"):
    """Visualize original and normalized data."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get numpy versions for plotting
        small_np = small_scale.detach().cpu().numpy()
        large_np = large_scale.detach().cpu().numpy()
        norm_small_np = normalized_small.detach().cpu().numpy()
        norm_large_np = normalized_large.detach().cpu().numpy()
        
        # Print statistics to file - this doesn't require visualization
        with open(f"{output_dir}/statistics.txt", "w") as f:
            f.write("Small Scale - Original:\n")
            f.write(f"  Min: {small_np.min():.6f}, Max: {small_np.max():.6f}\n")
            f.write(f"  Mean: {small_np.mean():.6f}, Std: {small_np.std():.6f}\n")
            f.write(f"  Dynamic Range: {small_np.max() - small_np.min():.6f}\n\n")
            
            f.write("Small Scale - Normalized:\n")
            f.write(f"  Min: {norm_small_np.min():.6f}, Max: {norm_small_np.max():.6f}\n")
            f.write(f"  Mean: {norm_small_np.mean():.6f}, Std: {norm_small_np.std():.6f}\n")
            f.write(f"  Dynamic Range: {norm_small_np.max() - norm_small_np.min():.6f}\n\n")
            
            f.write("Large Scale - Original:\n")
            f.write(f"  Min: {large_np.min():.6f}, Max: {large_np.max():.6f}\n")
            f.write(f"  Mean: {large_np.mean():.6f}, Std: {large_np.std():.6f}\n")
            f.write(f"  Dynamic Range: {large_np.max() - large_np.min():.6f}\n\n")
            
            f.write("Large Scale - Normalized:\n")
            f.write(f"  Min: {norm_large_np.min():.6f}, Max: {norm_large_np.max():.6f}\n")
            f.write(f"  Mean: {norm_large_np.mean():.6f}, Std: {norm_large_np.std():.6f}\n")
            f.write(f"  Dynamic Range: {norm_large_np.max() - norm_large_np.min():.6f}\n")
            
        print(f"Statistics saved to {output_dir}/statistics.txt")
        
        # Try to create visualizations, but don't fail if they can't be created
        try:
            # Use simplest possible matplotlib settings
            import matplotlib
            matplotlib.use('Agg')  # Force Agg backend
            matplotlib.rcParams['text.usetex'] = False
            matplotlib.rcParams['font.family'] = 'sans-serif'
            import matplotlib.pyplot as plt
            
            # Plot histograms - simple version avoiding tight_layout
            plt.figure(figsize=(16, 8))
            
            # Small scale before
            plt.subplot(2, 2, 1)
            plt.hist(small_np.flatten(), bins=50, alpha=0.7)
            plt.title("Small Scale - Original", fontfamily='sans-serif')
            plt.xlabel("Value", fontfamily='sans-serif')
            plt.ylabel("Frequency", fontfamily='sans-serif')
            plt.grid(alpha=0.3)
            
            # Small scale after
            plt.subplot(2, 2, 2)
            plt.hist(norm_small_np.flatten(), bins=50, alpha=0.7)
            plt.title("Small Scale - Normalized", fontfamily='sans-serif')
            plt.xlabel("Value", fontfamily='sans-serif')
            plt.ylabel("Frequency", fontfamily='sans-serif')
            plt.grid(alpha=0.3)
            
            # Large scale before
            plt.subplot(2, 2, 3)
            plt.hist(large_np.flatten(), bins=50, alpha=0.7)
            plt.title("Large Scale - Original", fontfamily='sans-serif')
            plt.xlabel("Value", fontfamily='sans-serif')
            plt.ylabel("Frequency", fontfamily='sans-serif')
            plt.grid(alpha=0.3)
            
            # Large scale after
            plt.subplot(2, 2, 4)
            plt.hist(norm_large_np.flatten(), bins=50, alpha=0.7)
            plt.title("Large Scale - Normalized", fontfamily='sans-serif')
            plt.xlabel("Value", fontfamily='sans-serif')
            plt.ylabel("Frequency", fontfamily='sans-serif')
            plt.grid(alpha=0.3)
            
            # Use subplots_adjust instead of tight_layout
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
            plt.savefig(f"{output_dir}/histograms.png", dpi=300)
            plt.close()
            
            print(f"Histogram visualization saved to {output_dir}/histograms.png")
        except Exception as e:
            print(f"Warning: Could not create histogram visualization: {e}")
            print("This doesn't affect the normalization process, only the visualization.")
    except Exception as e:
        print(f"Warning: Error in visualization: {e}")
        print("This doesn't affect the normalization process, only the visualization.")

def modify_architecture_for_normalization(file_path):
    """Add normalization function to the model architecture file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the normalization code is already added
    if "NormalizedClassConditionedUnet" in content:
        print("Normalization already added to architecture.")
        return False
    
    # Create the new code to add using triple single quotes to avoid docstring conflicts
    new_code = '''

class NormalizedClassConditionedUnet(nn.Module):
    """Modified UNet model that applies normalization to inputs."""
    def __init__(self, base_model, normalizer_path=None):
        super().__init__()
        self.base_model = base_model
        self.normalizer = None
        
        if normalizer_path:
            self.load_normalizer(normalizer_path)
    
    def load_normalizer(self, normalizer_path):
        """Load normalizer from file."""
        import sys
        import os
        
        # Add the directory containing normalize_data.py to the path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import the Normalizer class
        from normalize_data import Normalizer
        
        self.normalizer = Normalizer.load(normalizer_path)
        print(f"Loaded normalizer: {self.normalizer}")
    
    def forward(self, x, t, condition):
        """Forward pass with normalization."""
        if self.normalizer:
            # Apply normalization
            condition = self.normalizer.normalize_small_scale(condition)
            x = self.normalizer.normalize_large_scale(x)
        
        # Forward through base model
        return self.base_model(x, t, condition)
'''
    
    # Add the new class to the end of the file
    with open(file_path, 'a') as f:
        f.write(new_code)
    
    print(f"Added NormalizedClassConditionedUnet to {file_path}")
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Normalize diffusion model data')
    parser.add_argument('--method', type=str, default='percentile',
                       choices=['minmax', 'standard', 'percentile', 'match_range'],
                       help='Normalization method')
    parser.add_argument('--percentile', type=int, nargs=2, default=[1, 99],
                       help='Percentile range for percentile normalization (e.g. --percentile 1 99)')
    parser.add_argument('--save_path', type=str, default='normalizer.pt',
                       help='Path to save normalizer')
    parser.add_argument('--max_batches', type=int, default=20,
                       help='Maximum number of batches to use for fitting')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize normalization results')
    parser.add_argument('--modify_architecture', action='store_true',
                       help='Add normalization to architecture.py')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Using normalization method: {args.method}")
    
    # Collect data for fitting
    small_scale, large_scale = collect_batch_data(train_loader, args.max_batches)
    
    # Create and fit normalizer
    normalizer = Normalizer(method=args.method, percentile_range=args.percentile)
    normalizer.fit(small_scale, large_scale)
    
    # Print normalizer info
    print(f"Fitted normalizer: {normalizer}")
    
    # Save normalizer
    normalizer.save(args.save_path)
    
    # Visualize if requested
    if args.visualize:
        try:
            # Get a batch of data
            small_batch, large_batch = next(iter(valid_loader))
            
            # Normalize
            norm_small = normalizer.normalize_small_scale(small_batch)
            norm_large = normalizer.normalize_large_scale(large_batch)
            
            # Visualize
            visualize_normalization(small_batch, large_batch, norm_small, norm_large)
        except Exception as e:
            print(f"Error during visualization: {e}")
            print("This doesn't affect the normalizer itself, which has been saved successfully.")
    
    # Modify architecture if requested
    if args.modify_architecture:
        try:
            modify_architecture_for_normalization("architecture.py")
        except Exception as e:
            print(f"Error modifying architecture: {e}")
            print("This doesn't affect the normalizer itself, which has been saved successfully.")
    
    print("\nHow to use the normalizer in your code:")
    print("```python")
    print("# Method 1: Using NormalizedDataLoader")
    print("from normalize_data import Normalizer, NormalizedDataLoader")
    print(f"normalizer = Normalizer.load('{args.save_path}')")
    print("normalized_train_loader = NormalizedDataLoader(train_loader, normalizer)")
    print("normalized_valid_loader = NormalizedDataLoader(valid_loader, normalizer)")
    print("normalized_test_loader = NormalizedDataLoader(test_loader, normalizer)")
    print("")
    print("# Method 2: Using NormalizedClassConditionedUnet")
    print("from architecture import NormalizedClassConditionedUnet, ClassConditionedUnet")
    print("base_model = ClassConditionedUnet(...)")
    print(f"model = NormalizedClassConditionedUnet(base_model, normalizer_path='{args.save_path}')")
    print("```")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 
import torch
import torch.nn as nn
from normalization import DataNormalizer

class NormalizedConditionalModel(nn.Module):
    """
    A model wrapper that applies normalization to inputs and denormalization to outputs.
    This ensures that both small_scale and large_scale data have appropriate ranges
    for effective conditioning.
    """
    def __init__(self, base_model, normalizer):
        super().__init__()
        self.base_model = base_model
        self.normalizer = normalizer
        
    def forward(self, x, t, condition):
        """
        Forward pass with normalization.
        
        Args:
            x: The input tensor (noisy large_scale)
            t: Diffusion timestep
            condition: The conditioning tensor (small_scale)
            
        Returns:
            The model output with inverse normalization applied
        """
        # Normalize inputs
        condition_normalized = self.normalizer.normalize_small_scale(condition)
        x_normalized = self.normalizer.normalize_large_scale(x)
        
        # Debug print for first batch only (to avoid spamming)
        if not hasattr(self, 'first_forward_done'):
            print("\nNormalized inputs stats:")
            print(f"  condition: min={condition_normalized.min().item():.6f}, max={condition_normalized.max().item():.6f}")
            print(f"  x: min={x_normalized.min().item():.6f}, max={x_normalized.max().item():.6f}")
            self.first_forward_done = True
        
        # Forward pass through base model
        output = self.base_model(x_normalized, t, condition_normalized)
        
        # For diffusion models, we typically don't denormalize the noise prediction
        # as it's trained to predict the normalized noise
        return output
    
    def denormalize_sample(self, x):
        """Denormalize a generated sample back to the original data distribution."""
        return self.normalizer.inverse_large_scale(x)

def create_normalized_model(base_model, train_data=None, method="standard", normalizer_path=None):
    """
    Create a normalized model from a base model and training data.
    
    Args:
        base_model: The base model to wrap
        train_data: A tuple of (small_scale, large_scale) tensors for fitting normalization
        method: Normalization method ("minmax", "standard", "percentile", "log")
        normalizer_path: Path to load a pre-fit normalizer from
        
    Returns:
        A normalized model instance
    """
    if normalizer_path:
        print(f"Loading normalizer from {normalizer_path}")
        normalizer = DataNormalizer.load(normalizer_path)
    else:
        print(f"Creating {method} normalizer...")
        normalizer = DataNormalizer(method=method)
        
        if train_data:
            print("Fitting normalizer to training data...")
            small_scale, large_scale = train_data
            normalizer.fit(small_scale, large_scale)
            
            # Save the normalizer for future use
            save_path = f"normalizer_{method}.pt"
            normalizer.save(save_path)
            print(f"Normalizer saved to {save_path}")
        else:
            raise ValueError("Either train_data or normalizer_path must be provided")
    
    return NormalizedConditionalModel(base_model, normalizer)

def fit_normalizer_from_data_loaders(data_loader, method="standard", save_path=None, max_batches=10):
    """
    Fit a normalizer from data loaders.
    
    Args:
        data_loader: DataLoader containing (small_scale, large_scale) batches
        method: Normalization method
        save_path: Where to save the fitted normalizer
        max_batches: Maximum number of batches to process for fitting
        
    Returns:
        A fitted DataNormalizer instance
    """
    print(f"Creating {method} normalizer...")
    normalizer = DataNormalizer(method=method)
    
    print("Collecting data for normalization...")
    small_scale_samples = []
    large_scale_samples = []
    
    for i, (small_scale, large_scale) in enumerate(data_loader):
        small_scale_samples.append(small_scale)
        large_scale_samples.append(large_scale)
        
        if i >= max_batches:
            break
    
    # Concatenate all batches
    small_scale_data = torch.cat(small_scale_samples)
    large_scale_data = torch.cat(large_scale_samples)
    
    print("Fitting normalizer...")
    normalizer.fit(small_scale_data, large_scale_data)
    
    # Print statistics
    if method == "minmax":
        print(f"Small scale min: {normalizer.small_scale_normalizer.min_val:.6f}, max: {normalizer.small_scale_normalizer.max_val:.6f}")
        print(f"Large scale min: {normalizer.large_scale_normalizer.min_val:.6f}, max: {normalizer.large_scale_normalizer.max_val:.6f}")
    elif method == "standard":
        print(f"Small scale mean: {normalizer.small_scale_normalizer.mean:.6f}, std: {normalizer.small_scale_normalizer.std:.6f}")
        print(f"Large scale mean: {normalizer.large_scale_normalizer.mean:.6f}, std: {normalizer.large_scale_normalizer.std:.6f}")
    elif method == "percentile":
        print(f"Small scale {normalizer.fit_percentile[0]}th percentile: {normalizer.small_scale_normalizer.low_val:.6f}")
        print(f"Small scale {normalizer.fit_percentile[1]}th percentile: {normalizer.small_scale_normalizer.high_val:.6f}")
        print(f"Large scale {normalizer.fit_percentile[0]}th percentile: {normalizer.large_scale_normalizer.low_val:.6f}")
        print(f"Large scale {normalizer.fit_percentile[1]}th percentile: {normalizer.large_scale_normalizer.high_val:.6f}")
    
    if save_path:
        normalizer.save(save_path)
        print(f"Normalizer saved to {save_path}")
    
    return normalizer 
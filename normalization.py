import torch
import torch.nn as nn
import numpy as np

class MinMaxNormalizer(nn.Module):
    """Min-max normalization that maps data to [0, 1] range."""
    def __init__(self, min_val=None, max_val=None, eps=1e-8):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps
        self.fitted = min_val is not None and max_val is not None

    def fit(self, x):
        """Compute min and max from the data."""
        if isinstance(x, torch.Tensor):
            self.min_val = float(x.min().item())
            self.max_val = float(x.max().item())
        else:
            self.min_val = float(np.min(x))
            self.max_val = float(np.max(x))
        self.fitted = True
        return self

    def forward(self, x):
        """Normalize input to [0, 1] range."""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (x - self.min_val) / (self.max_val - self.min_val + self.eps)

    def inverse(self, x):
        """Convert normalized data back to original range."""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return x * (self.max_val - self.min_val + self.eps) + self.min_val


class StandardNormalizer(nn.Module):
    """Standardization that maps data to mean=0, std=1."""
    def __init__(self, mean=None, std=None, eps=1e-8):
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps
        self.fitted = mean is not None and std is not None

    def fit(self, x):
        """Compute mean and std from the data."""
        if isinstance(x, torch.Tensor):
            self.mean = float(x.mean().item())
            self.std = float(x.std().item())
        else:
            self.mean = float(np.mean(x))
            self.std = float(np.std(x))
        self.fitted = True
        return self

    def forward(self, x):
        """Standardize input to mean=0, std=1."""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (x - self.mean) / (self.std + self.eps)

    def inverse(self, x):
        """Convert standardized data back to original distribution."""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return x * (self.std + self.eps) + self.mean


class PercentileNormalizer(nn.Module):
    """Normalization based on percentiles to handle outliers."""
    def __init__(self, lower_percentile=5, upper_percentile=95, low_val=None, high_val=None, eps=1e-8):
        super().__init__()
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.low_val = low_val
        self.high_val = high_val
        self.eps = eps
        self.fitted = low_val is not None and high_val is not None

    def fit(self, x):
        """Compute percentile values from the data."""
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        self.low_val = float(np.percentile(x_np, self.lower_percentile))
        self.high_val = float(np.percentile(x_np, self.upper_percentile))
        self.fitted = True
        return self

    def forward(self, x):
        """Normalize input based on percentile range."""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        normalized = (x - self.low_val) / (self.high_val - self.low_val + self.eps)
        
        if isinstance(x, torch.Tensor):
            return torch.clamp(normalized, 0, 1)
        else:
            return np.clip(normalized, 0, 1)

    def inverse(self, x):
        """Convert normalized data back to original range (within percentiles)."""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return x * (self.high_val - self.low_val + self.eps) + self.low_val


class LogNormalizer(nn.Module):
    """Log-based normalization for data with large dynamic range."""
    def __init__(self, offset=1.0, base=10):
        super().__init__()
        self.offset = offset
        self.base = base
        self.log_fn = torch.log if base == torch.e else lambda x: torch.log(x) / torch.log(torch.tensor(base, device=x.device))
        self.exp_fn = torch.exp if base == torch.e else lambda x: torch.pow(torch.tensor(base, device=x.device), x)

    def forward(self, x):
        """Apply log transformation to input data."""
        # Ensure all values are positive by adding offset
        return self.log_fn(x + self.offset)

    def inverse(self, x):
        """Convert log-transformed data back to original scale."""
        return self.exp_fn(x) - self.offset


class DataNormalizer:
    """Utility class to handle data normalization for both small and large scale data."""
    def __init__(self, method="standard", fit_percentile=None):
        self.method = method
        self.fit_percentile = fit_percentile or (5, 95)
        self.small_scale_normalizer = None
        self.large_scale_normalizer = None
        
        if method == "minmax":
            self.small_scale_normalizer = MinMaxNormalizer()
            self.large_scale_normalizer = MinMaxNormalizer()
        elif method == "standard":
            self.small_scale_normalizer = StandardNormalizer()
            self.large_scale_normalizer = StandardNormalizer()
        elif method == "percentile":
            self.small_scale_normalizer = PercentileNormalizer(
                lower_percentile=self.fit_percentile[0],
                upper_percentile=self.fit_percentile[1]
            )
            self.large_scale_normalizer = PercentileNormalizer(
                lower_percentile=self.fit_percentile[0],
                upper_percentile=self.fit_percentile[1]
            )
        elif method == "log":
            self.small_scale_normalizer = LogNormalizer()
            self.large_scale_normalizer = LogNormalizer()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def fit(self, small_scale_data, large_scale_data):
        """Fit normalizers to the data."""
        self.small_scale_normalizer.fit(small_scale_data)
        self.large_scale_normalizer.fit(large_scale_data)
        return self
    
    def normalize_small_scale(self, x):
        """Normalize small scale data."""
        return self.small_scale_normalizer(x)
    
    def normalize_large_scale(self, x):
        """Normalize large scale data."""
        return self.large_scale_normalizer(x)
    
    def inverse_small_scale(self, x):
        """Convert normalized small scale data back to original range."""
        return self.small_scale_normalizer.inverse(x)
    
    def inverse_large_scale(self, x):
        """Convert normalized large scale data back to original range."""
        return self.large_scale_normalizer.inverse(x)
    
    def save(self, path):
        """Save normalizer parameters."""
        state = {
            'method': self.method,
            'fit_percentile': self.fit_percentile,
            'small_scale': {
                'min_val': getattr(self.small_scale_normalizer, 'min_val', None),
                'max_val': getattr(self.small_scale_normalizer, 'max_val', None),
                'mean': getattr(self.small_scale_normalizer, 'mean', None),
                'std': getattr(self.small_scale_normalizer, 'std', None),
                'low_val': getattr(self.small_scale_normalizer, 'low_val', None),
                'high_val': getattr(self.small_scale_normalizer, 'high_val', None),
                'offset': getattr(self.small_scale_normalizer, 'offset', None),
                'base': getattr(self.small_scale_normalizer, 'base', None),
            },
            'large_scale': {
                'min_val': getattr(self.large_scale_normalizer, 'min_val', None),
                'max_val': getattr(self.large_scale_normalizer, 'max_val', None),
                'mean': getattr(self.large_scale_normalizer, 'mean', None),
                'std': getattr(self.large_scale_normalizer, 'std', None),
                'low_val': getattr(self.large_scale_normalizer, 'low_val', None),
                'high_val': getattr(self.large_scale_normalizer, 'high_val', None),
                'offset': getattr(self.large_scale_normalizer, 'offset', None),
                'base': getattr(self.large_scale_normalizer, 'base', None),
            }
        }
        torch.save(state, path)
    
    @classmethod
    def load(cls, path):
        """Load normalizer from saved parameters."""
        state = torch.load(path)
        normalizer = cls(method=state['method'], fit_percentile=state['fit_percentile'])
        
        small_params = state['small_scale']
        large_params = state['large_scale']
        
        if normalizer.method == "minmax":
            normalizer.small_scale_normalizer = MinMaxNormalizer(
                min_val=small_params['min_val'],
                max_val=small_params['max_val']
            )
            normalizer.large_scale_normalizer = MinMaxNormalizer(
                min_val=large_params['min_val'],
                max_val=large_params['max_val']
            )
        elif normalizer.method == "standard":
            normalizer.small_scale_normalizer = StandardNormalizer(
                mean=small_params['mean'],
                std=small_params['std']
            )
            normalizer.large_scale_normalizer = StandardNormalizer(
                mean=large_params['mean'],
                std=large_params['std']
            )
        elif normalizer.method == "percentile":
            normalizer.small_scale_normalizer = PercentileNormalizer(
                lower_percentile=normalizer.fit_percentile[0],
                upper_percentile=normalizer.fit_percentile[1],
                low_val=small_params['low_val'],
                high_val=small_params['high_val']
            )
            normalizer.large_scale_normalizer = PercentileNormalizer(
                lower_percentile=normalizer.fit_percentile[0],
                upper_percentile=normalizer.fit_percentile[1],
                low_val=large_params['low_val'],
                high_val=large_params['high_val']
            )
        elif normalizer.method == "log":
            normalizer.small_scale_normalizer = LogNormalizer(
                offset=small_params['offset'],
                base=small_params['base']
            )
            normalizer.large_scale_normalizer = LogNormalizer(
                offset=large_params['offset'],
                base=large_params['base']
            )
            
        return normalizer 
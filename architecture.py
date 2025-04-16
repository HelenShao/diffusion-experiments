import torch
import torch.nn as nn
from diffusers import UNet2DModel

class ClassConditionedUnet(nn.Module):
  def __init__(self, num_cond=4, sample_size=28, layers_per_block=2, block_out_channels=(32, 64, 128, 256, 512, 1024)):
    super().__init__()

    # The embedding layer will map the class label to a vector of size class_emb_size
    self.class_emb = nn.Embedding(10, num_cond) # 10 classes for 10 labels. maps each integer to a vector of size num_cond

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=sample_size,           # Use the parameter instead of hardcoded value
        in_channels=1 + num_cond,          # Additional input channels for class cond.
        out_channels=1,                    # the number of output channels
        layers_per_block=layers_per_block, # Use the parameter
        block_out_channels=block_out_channels, # Use the parameter
        down_block_types=len(block_out_channels) * ("DownBlock2D",),  # Ensure length matches block_out_channels
        up_block_types=len(block_out_channels) * ("UpBlock2D",),      # Ensure length matches block_out_channels
    )
    
    # For diagnostic purposes, to track if this is the first forward pass
    self.first_forward = True

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels):
    # Shape of x:
    bs, ch, w, h = x.shape
      
    # class conditioning in right shape to add as additional input channels
    class_cond = self.class_emb(class_labels) # Map to embedding dimension, matrix shape (bs, 4)
    class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

    # Net input is now x and cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1) # (bs, 5, 256, 256)
    
    # Print diagnostic information on the first forward pass
    if self.first_forward:
        print(f"\nForward pass diagnostic in ClassConditionedUnet:")
        print(f"  x shape: {x.shape}, condition shape: {class_cond.shape}, net_input shape: {net_input.shape}")
        print(f"  x stats - Min/Max: {x.min().item():.6f} / {x.max().item():.6f}, Mean/Std: {x.mean().item():.6f} / {x.std().item():.6f}")
        print(f"  condition stats - Min/Max: {class_cond.min().item():.6f} / {class_cond.max().item():.6f}, Mean/Std: {class_cond.mean().item():.6f} / {class_cond.std().item():.6f}")
        print(f"  condition sum: {class_cond.abs().sum().item()}")
        print("   condition: ", class_cond)
        # Check if condition has any effect on the input
        if class_cond.abs().sum().item() < 1e-8:
            print("  WARNING: Condition tensor is effectively zero!")
        
        # Check if all elements in a batch have the same conditioning
        if bs > 1:
            for i in range(1, bs):
                if torch.allclose(class_cond[0], class_cond[i]):
                    print(f"  WARNING: All samples in the batch have the same conditioning!")
                    break
                    
        self.first_forward = False  # Only print once
    
    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, t).sample # (bs, 1, 28, 28)

class ConditionedUNetWithGroupNorm(nn.Module):
    """
    A conditioned diffusion model that integrates a U-Net with GroupNorm layers.
    This model is conditioned on an additional input (e.g., class labels, etc.)
    and utilizes GroupNorm for normalization.
    """
    def __init__(self, num_cond=1, sample_size=256, layers_per_block=2, block_out_channels=[64, 128, 256, 512, 1024]):
        super().__init__()

        # UNet2DModel with GroupNorm
        self.model = UNet2DModel(
            sample_size=sample_size,  # Adjust the target image resolution
            in_channels=1 + num_cond,  # Add extra channels for conditioning input
            out_channels=1,            # Output channels (grayscale image, adjust if different)
            layers_per_block=layers_per_block,  # ResNet layers per block
            block_out_channels=block_out_channels,  # Number of channels per block
            
            # Specify downsampling and upsampling blocks (residual blocks with GroupNorm instead of BatchNorm)
            down_block_types= len(block_out_channels)*(
                "DownBlock2D",
            ),
            up_block_types= len(block_out_channels)*(
                "UpBlock2D",
            ),
        )

        # Modify down_blocks and up_blocks to replace BatchNorm with GroupNorm
        self.down_blocks = nn.ModuleList([self._replace_batchnorm_with_groupnorm(block) for block in self.model.down_blocks])
        self.up_blocks = nn.ModuleList([self._replace_batchnorm_with_groupnorm(block) for block in self.model.up_blocks])

    def _replace_batchnorm_with_groupnorm(self, block):
        """
        This helper function iterates over the layers of the downsampling or upsampling blocks
        and replaces BatchNorm2d with GroupNorm.
        """
        new_layers = []
        
        for name, module in block.named_children():
            if isinstance(module, nn.BatchNorm2d):
                # Replace BatchNorm2d with GroupNorm (32 groups as an example)
                new_layers.append(nn.GroupNorm(32, module.num_features))
            else:
                new_layers.append(module)
        
        # Return a new block with the modified layers
        return nn.Sequential(*new_layers)

    def forward(self, x, t, condition):
        """
        Forward pass through the conditioned UNet. The input is conditioned on an additional input.
        :param x: The noisy image tensor (batch size, channels, width, height).
        :param t: The time step for the diffusion model (often a scalar or tensor).
        :param condition: The conditioning tensor (e.g., class labels, additional features).
        :return: The predicted denoised image tensor.
        """
        # Shape of x: (batch_size, 1, width, height) and condition: (batch_size, num_cond, width, height)
        
        # Concatenate the conditioning information with the noisy input image along the channel dimension
        net_input = torch.cat((x, condition), 1)  # Concatenate along channels: (batch_size, 1+num_cond, width, height)

        # Pass the concatenated input through the model along with the timestep
        return self.model(net_input, t).sample  # Return the denoised sample

def replace_batchnorm_with_groupnorm(module: nn.Module, num_groups: int = 32):
    """
    Recursively replaces all BatchNorm2d layers in a module with GroupNorm.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            setattr(module, name, nn.GroupNorm(num_groups, num_channels))
        else:
            replace_batchnorm_with_groupnorm(child, num_groups)
    return module



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


class ClassConditionedUnetGroupNorm_Attn(nn.Module):
    def __init__(self, num_cond=1, sample_size=256, layers_per_block=2, block_out_channels=(32, 64, 128, 256, 512, 1024)):
        super().__init__()

        # Make sure down_block_types and up_block_types match the length of block_out_channels
        num_blocks = len(block_out_channels)
        
        # Dynamically create down_block_types based on the length of block_out_channels
        down_block_types = []
        for i in range(num_blocks):
            if i < 2 or i == num_blocks - 1:  # First 2 blocks and bottleneck: no attention
                down_block_types.append("DownBlock2D")
            else:  # Middle blocks: with attention
                down_block_types.append("AttnDownBlock2D")
        
        # Dynamically create up_block_types based on the length of block_out_channels
        up_block_types = []
        for i in range(num_blocks):
            if i == 0 or i >= num_blocks - 2:  # Bottleneck and last 2 blocks: no attention
                up_block_types.append("UpBlock2D")
            else:  # Middle blocks: with attention
                up_block_types.append("AttnUpBlock2D")

        self.model = UNet2DModel(
            sample_size=sample_size,
            in_channels=1 + num_cond,
            out_channels=1,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
        )

        # Replace BatchNorm with GroupNorm
        replace_batchnorm_with_groupnorm(self.model)

    def forward(self, x, t, condition):
        bs, ch, h, w = x.shape
        net_input = torch.cat((x, condition), dim=1)
        return self.model(net_input, t).sample

class EnhancedConditionedUnet(nn.Module):
    """
    An enhanced version of the ClassConditionedUnet that implements conditioning in a more 
    robust way to ensure the conditional input actually affects the output.
    """
    def __init__(self, num_cond=1, sample_size=256, layers_per_block=2, block_out_channels=(32, 64, 128, 256, 512, 1024)):
        super().__init__()
        
        # Feature extraction for conditioning
        self.cond_feature_extractor = nn.Sequential(
            nn.Conv2d(num_cond, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        
        # Self.model is an unconditional UNet with extra input channels
        self.model = UNet2DModel(
            sample_size=sample_size,
            in_channels=1 + 32,          # 1 for input + 32 for conditioned features
            out_channels=1,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=len(block_out_channels) * ("DownBlock2D",),
            up_block_types=len(block_out_channels) * ("UpBlock2D",),
        )
        
        # For diagnostic purposes
        self.first_forward = True
        
    def forward(self, x, t, condition):
        # Extract features from condition
        cond_features = self.cond_feature_extractor(condition)
        
        # Print diagnostic on first forward pass
        if self.first_forward:
            print(f"\nForward pass diagnostic in EnhancedConditionedUnet:")
            print(f"  x shape: {x.shape}, condition shape: {condition.shape}")
            print(f"  Extracted condition features shape: {cond_features.shape}")
            print(f"  x stats - Min/Max: {x.min().item():.6f} / {x.max().item():.6f}")
            print(f"  condition stats - Min/Max: {condition.min().item():.6f} / {condition.max().item():.6f}")
            print(f"  cond_features stats - Min/Max: {cond_features.min().item():.6f} / {cond_features.max().item():.6f}")
            
            if condition.abs().sum().item() < 1e-8:
                print("  WARNING: Condition tensor is effectively zero!")
                
            self.first_forward = False
        
        # Concatenate input with conditioned features
        net_input = torch.cat((x, cond_features), dim=1)
        
        # Process through UNet
        return self.model(net_input, t).sample

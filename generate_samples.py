import os
import sys
import time
import math
import pickle
import numpy as np
import json
import torch
import torch.nn as nn
import h5py
import healpy as hp
from tqdm.auto import tqdm
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler

# Additional imports for specific utilities and projects
sys.path.extend([
    os.path.abspath("/scratch/gpfs/hshao/PartIII-Project"),
    "/scratch/gpfs/hshao/projectron/",
    "/scratch/gpfs/hshao/ILC_ML/reproducing/unet_vanilla_4_freq"
])

import pixell.enmap as enmap
import pixell.utils as utils
import pixell.curvedsky as curvedsky
import pixell.enplot as enplot
import pixell.reproject as reproject
import reprojection.reprojector as reprojector

from architecture import *

# Device setup
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the configuration
checkpoint_dir = '/scratch/gpfs/hshao/ILC_ML/1_freq_scale/220_dust_only/d9/d9_ell_200_no_norm/diffusion/d12_rm_UNet2DModel/d12_rm_normalize_no_cond_ddpm_wg_UNet2DModel'
config_file_path = os.path.join(checkpoint_dir, "config.json")
print(f"Loading configuration from {config_file_path}")
with open(config_file_path, 'r') as f:
    config = json.load(f)

# Load variables from config
scheduler_type = config['scheduler']
num_inference_steps = config['num_inference_steps']
num_cond = config['num_cond']
sample_size = config['sample_size']
layers_per_block = config['layers_per_block']
block_out_channels = tuple(config['block_out_channels'])
experiment = config['experiment']
num_train_timesteps = config['num_train_timesteps']
archi = config['archi']
model_type = config['model_type']

print("Model configuration:")
print(f"  Experiment: {experiment}")
print(f"  Scheduler: {scheduler_type}")
print(f"  Sample size: {sample_size}")
print(f"  Num conditional inputs: {num_cond}")
print(f"  Block out channels: {block_out_channels}")

# Setup noise scheduler
if scheduler_type == "ddpm":
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
elif scheduler_type == "ddim":
    noise_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
    noise_scheduler.set_timesteps(num_inference_steps)
else:
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")

# Model setup
if archi == "ConditionedUNetWithGroupNorm":
    base_model = ConditionedUNetWithGroupNorm(num_cond=num_cond, sample_size=sample_size, layers_per_block=layers_per_block, block_out_channels=block_out_channels)
elif archi == "ClassConditionedUnetGroupNorm_Attn":
    base_model = ClassConditionedUnetGroupNorm_Attn(num_cond=num_cond, sample_size=sample_size, layers_per_block=layers_per_block, block_out_channels=block_out_channels)
elif archi == "UNet2DModel":
    if model_type == 'enhanced':
        print("Using enhanced conditioning model for stronger conditioning effect")
        base_model = EnhancedConditionedUnet(num_cond=num_cond, sample_size=sample_size, layers_per_block=layers_per_block, block_out_channels=block_out_channels)
    elif model_type == 'standard':
        base_model = ClassConditionedUnet(num_cond=num_cond, sample_size=sample_size, layers_per_block=layers_per_block, block_out_channels=block_out_channels)

# Wrap with normalizer
if config['normalizer_path'] is None:
    net = base_model
    print("Running WITHOUT normalization")
else:
    net = NormalizedClassConditionedUnet(base_model, normalizer_path=config['normalizer_path'])
    print("Running WITH normalization")

# Load the best model
best_model_path = os.path.join(checkpoint_dir, 'f_best_model.pth')
checkpoint = torch.load(best_model_path, map_location=device)
net.load_state_dict(checkpoint['model_state_dict'])
net.to(device)
net.eval()

print(f"Loaded model from {best_model_path} (epoch {checkpoint['epoch']})")
print(f"Validation loss: {checkpoint['min_valid_loss']:.6f}")

# Load test data
print("Loading test data...")
# Import data module dynamically - since you already have it in main.py
with open("load_data.py") as f:
    exec(f.read())

# Create output directories for generated samples
output_dir = os.path.join(checkpoint_dir, "generated_samples")
os.makedirs(output_dir, exist_ok=True)

# Generate samples
num_samples_to_generate = 50  # Adjust as needed
samples_per_batch = min(4, num_samples_to_generate)
num_batches = (num_samples_to_generate + samples_per_batch - 1) // samples_per_batch  # Ceiling division

# Set timesteps for the scheduler
if scheduler_type == "ddim":
    noise_scheduler.set_timesteps(num_inference_steps)
else:
    # For DDPM, can use more steps or the default
    timesteps = list(range(999, 0, -1))  # Reverse order from 999 to 0

print(f"Generating {num_samples_to_generate} samples...")

# Loop through test data to get conditioning samples
test_loader_iter = iter(test_loader)
generated_count = 0

try:
    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        # Get conditioning from the test set
        try:
            small_scale, large_scale = next(test_loader_iter)
        except StopIteration:
            # If we run out of test data, restart the iterator
            test_loader_iter = iter(test_loader)
            small_scale, large_scale = next(test_loader_iter)
            
        small_scale = small_scale.to(device)
        large_scale = large_scale.to(device)
        batch_size = small_scale.shape[0]
        
        # Start from random noise
        noisy_large_scale = torch.randn_like(large_scale).to(device)
        initial_noise = noisy_large_scale.clone()
        
        # Store steps for visualization later
        if batch_idx == 0:
            denoising_steps = []
            denoising_timesteps = []
        
        with torch.no_grad():
            # Perform the denoising process
            timesteps = noise_scheduler.timesteps #if scheduler_type == "ddim" else list(range(999, 0, -10))
            for t in tqdm(timesteps, desc=f"Batch {batch_idx+1}"):
                # Get model prediction (predicted noise)
                # if scheduler_type == "ddim":
                #     timestep = t
                # else:
                #     timestep = torch.full((batch_size,), t, device=device).long()
                
                residual = net(noisy_large_scale, t, small_scale)

                # Update sample with step
                noisy_large_scale = noise_scheduler.step(residual, t, noisy_large_scale).prev_sample
                
                # try:
                #     # Try the simple approach first
                #     if scheduler_type == "ddim" or True:  # We'll try this for both schedulers
                #         if isinstance(timestep, torch.Tensor) and timestep.numel() > 1:
                #             # Use first timestep if it's a batch (scheduler expects scalar)
                #             t_scalar = timestep[0].item() if scheduler_type == "ddpm" else t
                #         else:
                #             t_scalar = timestep.item() if isinstance(timestep, torch.Tensor) else t
                            
                #         noisy_large_scale = noise_scheduler.step(residual, t_scalar, noisy_large_scale).prev_sample
                    
                # except RuntimeError as e:
                #     # If simple approach fails, fall back to manual implementation
                #     print(f"Warning: Built-in scheduler step failed, using manual step. Error: {e}")
                    
                #     # Manual step for DDPM (fallback)
                #     alpha_t = 1.0 - noise_scheduler.betas[t]
                #     alpha_t_sqrt = alpha_t ** 0.5
                #     next_x = (noisy_large_scale - (1 - alpha_t_sqrt) * residual) / alpha_t_sqrt
                    
                #     # Only add noise for DDPM and when t > 0
                #     if scheduler_type == "ddpm" and t > 0:
                #         noise_scale = 0.002 * torch.ones_like(next_x)
                #         noise = torch.randn_like(next_x) * noise_scale
                #         next_x = next_x + noise
                        
                #     noisy_large_scale = next_x
                
                # Save intermediate steps for the first sample of the first batch
                if batch_idx == 0 and len(denoising_steps) < 10:
                    if t % max(len(timesteps) // 10, 1) == 0:
                        # Store the intermediate result as NumPy array
                        denoising_steps.append(noisy_large_scale[0].detach().cpu().numpy())
                        denoising_timesteps.append(t)
            
            # Final result contains the generated samples
            generated_large_scale = noisy_large_scale
            
            # Save the results
            for i in range(batch_size):
                if generated_count >= num_samples_to_generate:
                    break
                
                # Save numpy arrays
                np.save(f"{output_dir}/sample_{generated_count}_small_scale.npy", 
                       small_scale[i].detach().cpu().numpy())
                np.save(f"{output_dir}/sample_{generated_count}_ground_truth.npy", 
                       large_scale[i].detach().cpu().numpy())
                np.save(f"{output_dir}/sample_{generated_count}_generated.npy",
                      generated_large_scale[i].detach().cpu().numpy())
                
                generated_count += 1
    
    # Also save the denoising steps for the first sample
    if denoising_steps:
        denoising_dir = os.path.join(output_dir, "denoising_steps")
        os.makedirs(denoising_dir, exist_ok=True)
        
        for step_idx, (step_img, step_t) in enumerate(zip(denoising_steps, denoising_timesteps)):
            np.save(f"{denoising_dir}/step_{step_idx}_t_{step_t}.npy", step_img)
        
    print(f"Successfully generated {generated_count} samples.")
    print(f"Results saved to {output_dir}")
    if denoising_steps:
        print(f"Denoising steps saved to {denoising_dir}")

except Exception as e:
    print(f"Error during generation: {e}")
    import traceback
    traceback.print_exc() 
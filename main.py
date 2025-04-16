import os
import sys
import time
import math
import pickle
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import h5py
import warnings
import optuna
import healpy as hp
import pysm3
import argparse
from scipy.stats import pearsonr
from torchsummary import summary
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler
from accelerate import Accelerator
from accelerate.utils import find_batch_size
from tqdm.auto import tqdm
from architecture import NormalizedClassConditionedUnet

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train diffusion model')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'enhanced'], 
                   help='Model type: standard or enhanced conditioning')
parser.add_argument('--no_norm', action='store_true', help='Disable normalization')
parser.add_argument('--config', type=str, help='Path to JSON configuration file')
args = parser.parse_args()

# Set debug factor based on command line argument
debug_factor = 0.0 if args.debug else 1.0
model_type = args.model_type
print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}, debug_factor: {debug_factor}")
print(f"Using model type: {model_type}")

import architecture
from architecture import *

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
from common_imports import *
from cmb_tools_main import *
import data
import common_vars
from common_vars import *

# Global configurations
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
warnings.filterwarnings("ignore")

# Device setup
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Constants and variables
seed = 4
np.random.seed(seed)
unit = r"$\mu K$"
mollview_min, mollview_max = -10, 10  # Mollview colorbar min/max
freq_obs = [93., 143., 220., 280.]  # GHz
freq_names = [f'{freq}GHz' for freq in freq_obs]
fg_models = ["d1", "s1", "a2"]
delta_ell, nside, lmin, lmax = 50, 2048, 0, min(2048 * 2, 2000)
N_freq, N_constraints = 2, 1
ell_hp = np.arange(lmax + 1)
cl_scaling_hp = ell_hp * (ell_hp + 1) / (2 * np.pi)
res_arcmin = math.ceil(hp.nside2resol(nside, arcmin=True))
stamp_size_pixels, taper_width_pixels, pad_width_pixels = 256, 5, 0
ell_cutoff = 200 

# Data loading
with open("/scratch/gpfs/hshao/ILC_ML/1_freq_scale/220_dust_only/d9/d9_ell_200_no_norm/diffusion/fashion_mnist/load_data.py") as f:
    exec(f.read())

config = {
            'experiment': "vanilla",
            'archi': 'UNet2DModel',
            'model_type': model_type,
            'dataset_type': 'wg',
            'scheduler': 'ddpm',
            'gen_samples': False,
            'num_inference_steps': 50,
            'n_epochs': 100,
            'loss_fn': 'mse',
            'num_cond': 4,
            'sample_size': 28,
            'layers_per_block': 1,
            'block_out_channels': (32, 64, 64),
            'device': str(device),
            'learning_rate': 1e-3,
            'patience': 2000,
            'num_train_timesteps': 1000,
            'min_valid_loss': float('inf'),
            'seed': seed,
            'freq_obs': freq_obs,
            'fg_models': fg_models,
            'nside': nside,
            'lmin': lmin,
            'lmax': lmax,
            'ell_cutoff': ell_cutoff,
            'stamp_size_pixels': stamp_size_pixels,
            'taper_width_pixels': taper_width_pixels,
            'pad_width_pixels': pad_width_pixels
        }

################ Configuration #################
if args.config:
    try:
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
        print("Configuration loaded successfully")
    except Exception as e:
        print(f"Error loading configuration file: {str(e)}")
        print("Using default configuration instead")
        config = config
else:
    config = config

# Setup logging to both console and file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, "a")
        # Write a header to the log file
        self.log.write(f"\n{'='*50}\n")
        self.log.write(f"Log started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"{'='*50}\n\n")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Create log directory based on experiment name and setup logging
experiment_name = config['experiment']
checkpoint_dir = f'{experiment_name}_{config["scheduler"]}_{config["dataset_type"]}_{config["archi"]}'
log_filename = f"{checkpoint_dir}/{experiment_name}_log.txt"
os.makedirs(checkpoint_dir, exist_ok=True)
sys.stdout = Logger(log_filename)
print(f"Logging output to {log_filename}")

# Load variables from the config dictionary into Python variables
for key, value in config.items():
    globals()[key] = value

# Accelerator setup
accelerator = Accelerator(mixed_precision='fp16', cpu=False)
device = accelerator.device
print(f"Using device: {device}")
print(f"Process count: {accelerator.num_processes}")
print(f"Using block_out_channels: {block_out_channels}")

# To map the string to the actual loss function
def get_loss_function(loss_name):
    if loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'L1Loss':
        return nn.L1Loss()
    # You can add more loss functions here as needed
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

loss_fn_str = config['loss_fn']
loss_fn = get_loss_function(loss_fn_str)

# Noise scheduler setup     
num_train_timesteps = 1000
if scheduler == "ddpm":
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
elif scheduler == "ddim":
    noise_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
    noise_scheduler.set_timesteps(num_inference_steps)

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
if args.no_norm:
    net = base_model  # Use without normalization
    print("Running WITHOUT normalization")
    config['normalizer_path'] = None
else:
    net = NormalizedClassConditionedUnet(base_model, normalizer_path='/scratch/gpfs/hshao/ILC_ML/1_freq_scale/220_dust_only/d9/d9_ell_200_no_norm/diffusion/normalizer.pt')
    print("Running WITH normalization")
    config['normalizer_path'] = '/scratch/gpfs/hshao/ILC_ML/1_freq_scale/220_dust_only/d9/d9_ell_200_no_norm/diffusion/normalizer.pt'

# Debug print to verify model creation
print(f"\nModel structure debug:")
print(f"  - UNet with {len(block_out_channels)} blocks")
try:
    if hasattr(net, 'model'):
        # For non-normalized model with direct model attribute
        print(f"  - Down block types: {net.model.down_block_types}")
        print(f"  - Up block types: {net.model.up_block_types}")
        print(f"  - Block out channels: {net.model.config.block_out_channels}\n")
    elif hasattr(net, 'base_model'):
        # For normalized model
        if hasattr(net.base_model, 'model'):
            # If base_model has a model attribute (ClassConditionedUnet)
            print(f"  - Down block types: {net.base_model.model.down_block_types}")
            print(f"  - Up block types: {net.base_model.model.up_block_types}")
            print(f"  - Block out channels: {net.base_model.model.config.block_out_channels}\n")
        else:
            # Fallback for other model types
            print(f"  - Unable to access detailed model structure attributes\n")
    else:
        print(f"  - Unable to access model structure attributes\n")
except Exception as e:
    print(f"  - Error accessing model structure: {str(e)}\n")

# Optimizer and scheduler
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, verbose=True)

# Checkpoint paths
best_model_path = f'f_best_model.pth'
save_directory = checkpoint_dir
os.makedirs(checkpoint_dir, exist_ok=True)
path_to_checkpoint = os.path.join(save_directory, best_model_path)

# Initialize variables for early stopping
min_valid_loss, patience, patience_counter = float('inf'), config['patience'], 0

# Check if a saved model exists and resume from it
if os.path.exists(path_to_checkpoint):
    print("Loading checkpoint from ", path_to_checkpoint)
    checkpoint = torch.load(path_to_checkpoint, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    min_valid_loss = checkpoint['min_valid_loss']
    patience_counter = checkpoint['patience_counter']
    
    # Load scheduler state if it exists in checkpoint
    if 'lr_scheduler_state_dict' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print("Learning rate scheduler state loaded")
    
    net.to(device)
    print(f"Resumed from epoch {start_epoch}")
else:
    print("No saved model found, starting from scratch")
    net.to(device)
    start_epoch = 0

# Prepare model, optimizer, and dataloaders with accelerator
net, opt, train_loader, valid_loader, test_loader = accelerator.prepare(net, opt, train_loader, valid_loader, test_loader)

# Track loss history
train_loss_history, valid_loss_history, test_loss_history = [], [], []

# Calculate total number of parameters
total_params = sum(p.numel() for p in net.parameters())
print(f"Num. parameters: {total_params:,}")

# Create save directories
save_dir = os.path.join(save_directory, "inspect_samples")
train_save, valid_save, test_save = os.path.join(save_dir, "train"), os.path.join(save_dir, "valid"), os.path.join(save_dir, "test")
for path in [train_save, valid_save, test_save]:
    os.makedirs(path, exist_ok=True)

# Directory to save the config file
config_file_path = os.path.join(save_directory, "config.json")

# Save the config dictionary as a JSON file
with open(config_file_path, 'w') as f:
    json.dump(config, f, indent=4)

print(f"Configuration saved to {config_file_path}")

# Setup Training Log File
log_file_path = os.path.join(save_directory, "training_log.txt")
if start_epoch == 0:
    # If starting from scratch, create/overwrite the log file and write the header
    with open(log_file_path, 'w') as log_f:
        log_f.write("Epoch TrainLoss ValidLoss TestLoss\n")
    print(f"Initialized training log at {log_file_path}")
else:
    # If resuming, just print that we'll append
    print(f"Appending to existing training log at {log_file_path}")

# Training loop
for epoch in range(start_epoch, n_epochs):
    ########################### Train ###########################
    net.train()
    train_losses = []
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
    
    for x, y in progress_bar:
        batch_size = find_batch_size(x)
        # Convert tensors to appropriate types
        x = x.to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
        y = y.to(device) # Keep y as int/long for embedding
        
        # In debug mode, use zero tensor of same type as y for conditioning
        if debug_factor == 0.0:
            y_cond = torch.zeros_like(y) # Creates a zero tensor with same shape and dtype
        else:
            y_cond = y
        
        timesteps = torch.randint(0, num_train_timesteps - 1, (x.shape[0],)).long().to(device)
        noise = torch.randn_like(x)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Print diagnostic information to check conditioning
        if epoch == start_epoch : #and len(train_losses) == 0:  # Only print for the first batch of the first epoch
            print(f"\nDiagnostic check - y stats:")
            print(f"  Shape: {x.shape}")
            print(f"  Type: {y.dtype}")
            # Convert y to float for calculations if it's not already
            y_float = y.float() if y.dtype != torch.float32 else y
            print(f"  Min/Max: {y_float.min().item():.6f} / {y_float.max().item():.6f}")
            print(f"  Mean/Std: {y_float.mean().item():.6f} / {y_float.std().item():.6f}")
            print(f"  Fraction of zeros: {(y == 0).float().mean().item():.6f}")
            print(f"  debug_factor: {debug_factor}")
            print(f"  Original y values: {y}")
            print(f"  Conditioned y values (with debug_factor={debug_factor}): {y_cond}")

        # For embedding layers, y should be long/int type
        pred = net(noisy_x, timesteps, y_cond)
        loss = loss_fn(pred, noise)

        accelerator.backward(loss)
        opt.step()
        opt.zero_grad()

        train_losses.append(accelerator.gather(loss).mean().item())
        progress_bar.set_postfix({"train_loss": train_losses[-1]})

    avg_train_loss = sum(train_losses) / len(train_losses)
    train_loss_history.append(avg_train_loss)

    ########################### Validation ###########################
    net.eval()
    valid_losses = []
    progress_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Valid]")
    
    with torch.no_grad():
        for x, y in progress_bar:
            # Convert tensors to appropriate types
            x = x.float()
            y = y.to(device)  # Keep y as int/long for embedding
            
            # In debug mode, use zero tensor of same type as y for conditioning
            if debug_factor == 0.0:
                y_cond = torch.zeros_like(y) # Creates a zero tensor with same shape and dtype
            else:
                y_cond = y
            
            timesteps = torch.randint(0, num_train_timesteps - 1, (x.shape[0],)).long().to(device)
            noise = torch.randn_like(x)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            pred = net(noisy_x, timesteps, y_cond)
            loss = loss_fn(pred, noise)

            valid_losses.append(accelerator.gather(loss).mean().item())
            progress_bar.set_postfix({"valid_loss": valid_losses[-1]})

    avg_valid_loss = sum(valid_losses) / len(valid_losses)
    valid_loss_history.append(avg_valid_loss)

    # Step the scheduler, monitoring the validation loss
    lr_scheduler.step(avg_valid_loss)
    print(f"Learning Rate: {lr_scheduler.optimizer.param_groups[0]['lr']}")

    ########################### Test ###########################
    test_losses = []
    progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Test]")
    
    with torch.no_grad():
        for x, y in progress_bar:
            # Convert tensors to appropriate types
            x = x.float()
            y = y.to(device)  # Keep y as int/long for embedding
            
            # In debug mode, use zero tensor of same type as y for conditioning
            if debug_factor == 0.0:
                y_cond = torch.zeros_like(y) # Creates a zero tensor with same shape and dtype
            else:
                y_cond = y
            
            timesteps = torch.randint(0, num_train_timesteps - 1, (x.shape[0],)).long().to(device)
            noise = torch.randn_like(x)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            pred = net(noisy_x, timesteps, y_cond)
            loss = loss_fn(pred, noise)

            test_losses.append(accelerator.gather(loss).mean().item())
            progress_bar.set_postfix({"test_loss": test_losses[-1]})

    avg_test_loss = sum(test_losses) / len(test_losses)
    test_loss_history.append(avg_test_loss)

    # Print losses
    accelerator.print(f'Epoch {epoch+1}/{n_epochs}: Train = {avg_train_loss:.5f}, Valid = {avg_valid_loss:.5f}')
    accelerator.print(f'Epoch {epoch+1}/{n_epochs}: Test  = {avg_test_loss:.5f}')

    # Append results to log file
    if accelerator.is_main_process:
        with open(log_file_path, 'a') as log_f:
            log_f.write(f"{epoch+1} {avg_train_loss} {avg_valid_loss} {avg_test_loss}\n")

    ########################### Early Stopping & Checkpoint ###########################
    if accelerator.is_main_process:
        if avg_valid_loss < min_valid_loss:
            min_valid_loss = avg_valid_loss
            patience_counter = 0

            accelerator.wait_for_everyone()
            accelerator.save_model(net, save_directory)
            accelerator.save_state(checkpoint_dir)

            unwrapped_model = accelerator.unwrap_model(net)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'min_valid_loss': min_valid_loss,
                'patience_counter': patience_counter,
                'model_type': model_type,  # Save model_type in checkpoint
                'lr_scheduler_state_dict': lr_scheduler.state_dict()  # Save scheduler state
            }
            accelerator.save(checkpoint, path_to_checkpoint)
            print(f'New best model saved with validation loss: {min_valid_loss:.5f}')

            if gen_samples:
                for phase, loader, save_path in [("Valid", valid_loader, valid_save), ("Test", test_loader, test_save)]:
                    print(f"Generating {phase} samples...")
                    progress_bar = tqdm(loader, desc=f"Sampling {phase} Set (Best @ Epoch {epoch+1})")
                    with torch.no_grad():
                        for batch_idx, (x, y) in enumerate(progress_bar):
                            # Convert tensors to float type if they aren't already
                            x = x.float()
                            y = y.float()
                            
                            # Set the number of timesteps for the scheduler
                            if scheduler == "ddim":
                                noise_scheduler.set_timesteps(num_inference_steps)
                            
                            # Start with pure noise - we're denoising x
                            noisy_x = torch.randn_like(x).to(device)
                            
                            # Store the initial noise for comparison
                            initial_noise = noisy_x.clone()
                            
                            # Conditioning is y input - keep as int/long for embedding
                            y = y.to(device)
                            
                            # In debug mode, use zero tensor of same type as y for conditioning
                            if debug_factor == 0.0:
                                y_cond = torch.zeros_like(y) # Creates a zero tensor with same shape and dtype
                            else:
                                y_cond = y
                            
                            # Perform the denoising process
                            for t in tqdm(noise_scheduler.timesteps, desc=f"{phase} Batch {batch_idx+1}"):
                                # Get model prediction (predicted noise)
                                residual = net(noisy_x, t, y_cond)
                                
                                # Step through the scheduler
                                noisy_x = noise_scheduler.step(residual, t, noisy_x).prev_sample
                            
                            # noisy_x now contains the generated sample
                            generated_x = noisy_x
                            
                            # Optional: Calculate the prediction by applying the model once to the initial noise
                            initial_t = torch.full((x.shape[0],), noise_scheduler.timesteps[0], device=device)
                            pred = net(initial_noise, initial_t, y_cond)
                            
                            # Save the results
                            for img_idx in range(len(x)):
                                try:
                                    np.save(f"{save_path}/batch{batch_idx}_img{img_idx}_y_conditioning.npy", 
                                            y[img_idx].detach().cpu().numpy())
                                    np.save(f"{save_path}/batch{batch_idx}_img{img_idx}_initial_noise.npy", 
                                            initial_noise[img_idx].detach().cpu().numpy())
                                    np.save(f"{save_path}/batch{batch_idx}_img{img_idx}_predicted_noise.npy", 
                                            pred[img_idx].detach().cpu().numpy()) 
                                    np.save(f"{save_path}/batch{batch_idx}_img{img_idx}_x_ground_truth.npy", 
                                            x[img_idx].detach().cpu().numpy())
                                    np.save(f"{save_path}/batch{batch_idx}_img{img_idx}_x_generated.npy", 
                                            generated_x[img_idx].detach().cpu().numpy())
                                except IndexError:
                                    print(f"Skipping image {img_idx} in batch {batch_idx} due to indexing error.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("patience_counter: ", patience_counter)
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break

    net.train() 

    ########################### Plotting ###########################
    # if accelerator.is_main_process and epoch % 5 == 0 and epoch != 0:
    #     print("Plotting training vs validation loss...")
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(train_loss_history, label='Training Loss')
    #     plt.plot(valid_loss_history, label=f'Validation Loss (Min: {min_valid_loss:.3e})')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.title(f'Training vs Valid Loss (Conditioning Factor: {debug_factor})')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(f"{checkpoint_dir}/train_valid_loss.png", dpi=300)
    #     plt.close()

# At the end of training, update the min_valid_loss in config
print("Plotting training vs validation loss...")
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Training Loss')
plt.plot(valid_loss_history, label=f'Validation Loss (Min: {min_valid_loss:.3e})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training vs Valid Loss (Conditioning Factor: {debug_factor})')
plt.grid(True)
plt.legend()
plt.savefig(f"{checkpoint_dir}/train_valid_loss.png", dpi=300)
plt.close()

config['min_valid_loss'] = min_valid_loss

# Save the updated config dictionary to the config file
config_file_path = os.path.join(save_directory, "config.json")

# Write the updated config to the file
with open(config_file_path, 'w') as f:
    json.dump(config, f, indent=4)

print(f"Updated configuration saved to {config_file_path}")
accelerator.print('Training completed!')

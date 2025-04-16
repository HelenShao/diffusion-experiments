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
from scipy.stats import pearsonr
from torchsummary import summary
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler
from accelerate import Accelerator
from accelerate.utils import find_batch_size
from tqdm.auto import tqdm

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
import architecture
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

# Model files
with open("load_data.py") as f:
    exec(f.read())
    
from architecture import *

################ Configuration #################
config = {
    'experiment': "group_norm",
    'archi': 'UNet2DModel',
    'dataset_type': 'wg',
    'scheduler': 'ddpm',
    'gen_samples': False,
    'num_inference_steps': 50,
    'n_epochs': 100,
    'loss_fn': 'mse',
    'num_cond': 1,
    'sample_size': 256,
    'layers_per_block': 2,
    'block_out_channels': [64, 128, 256, 512],
    'device': str(device),
    'learning_rate': 1e-3,
    'patience': 10,
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

# Load variables from the config dictionary into Python variables
for key, value in config.items():
    globals()[key] = value

# Accelerator setup
accelerator = Accelerator(mixed_precision='fp16', cpu=False)
device = accelerator.device
print(f"Using device: {device}")
print(f"Process count: {accelerator.num_processes}")

# A simple function to map the string to the actual loss function
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
net = ConditionedUNetWithGroupNorm(num_cond=num_cond, sample_size=sample_size, layers_per_block=layers_per_block, block_out_channels=block_out_channels)

# Optimizer and scheduler
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, verbose=True)

# Checkpoint paths
best_model_path = f'{archi}_{scheduler}_{dataset_type}.pth'
checkpoint_dir = f'{experiment}_cond_diff_checkpoints_{scheduler}_{dataset_type}_{archi}'
save_directory = checkpoint_dir
os.makedirs(checkpoint_dir, exist_ok=True)
path_to_checkpoint = os.path.join(save_directory, best_model_path)

# Check if the saved model exists
if os.path.exists(path_to_checkpoint):
    net = accelerator.unwrap_model(net)
    checkpoint = torch.load(path_to_checkpoint, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    #net.load_state_dict(checkpoint)

    #checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
    print(f"Loaded saved model")
    net.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])

    #Restore other relevant states
    start_epoch = checkpoint['epoch']
    min_valid_loss = checkpoint['min_valid_loss']
    patience_counter = checkpoint['patience_counter']
    net.to(device)
    print(f"Resumed from epoch {start_epoch}")
else:
    net.to(device)
    start_epoch = 0  # Initialize start_epoch when no checkpoint exists

# Wrap in accelerate
net, opt, train_loader, valid_loader = accelerator.prepare(net, opt, train_loader, valid_loader)

# To save predicted and noise during training/validating
save_dir = os.path.join(save_directory, "inspect_samples")
train_save = os.path.join(save_dir, "train")
valid_save = os.path.join(save_dir, "valid")
test_save = os.path.join(save_dir, "test")

# Create directories if they don't exist
os.makedirs(save_dir, exist_ok=True)
os.makedirs(train_save, exist_ok=True)
os.makedirs(valid_save, exist_ok=True)
os.makedirs(test_save, exist_ok=True)


import torch.nn.functional as F
from piq import psnr, ssim

# To save predicted and noise during training/validating
save_dir = os.path.join(save_directory, "inspect_samples")
train_save = os.path.join(save_dir, "train")
valid_save = os.path.join(save_dir, "valid")
test_save = os.path.join(save_dir, "test")

# Create directories if they don't exist
os.makedirs(save_dir, exist_ok=True)
os.makedirs(train_save, exist_ok=True)
os.makedirs(valid_save, exist_ok=True)
os.makedirs(test_save, exist_ok=True)

def generate_samples(loader, save_path, scheduler, net, noise_scheduler, start_epoch, max_batches=None):
    net.eval()
    progress_bar = tqdm(loader, desc=f"Sampling Set (Best @ Epoch {start_epoch+1})")

    mse_list, psnr_list, ssim_list = [], [], []

    for batch_idx, (y, fg_large) in enumerate(progress_bar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = torch.randn_like(fg_large)
        noise = x.clone()

        for t in tqdm(noise_scheduler.timesteps, desc=f"Generating Samples: Batch {batch_idx+1}"):
            with torch.no_grad():
                residual = net(x, t, y)

            if scheduler == "ddim":
                noise_scheduler.eta = 0.0
                x = noise_scheduler.step(residual, t, x, noise=noise).prev_sample
            else:
                x = noise_scheduler.step(residual, t, x).prev_sample

        for img_idx in range(fg_large.size(0)):
            try:
                gt = fg_large[img_idx].detach().cpu()
                pred = x[img_idx].detach().cpu()

                # Save outputs
                np.save(f"{save_path}/batch{batch_idx}_img{img_idx}_fg_small.npy", y[img_idx].detach().cpu().numpy())
                np.save(f"{save_path}/batch{batch_idx}_img{img_idx}_noise.npy", noise[img_idx].detach().cpu().numpy())
                np.save(f"{save_path}/batch{batch_idx}_img{img_idx}_predicted_noise.npy", residual[img_idx].detach().cpu().numpy())
                np.save(f"{save_path}/batch{batch_idx}_img{img_idx}_ground_truth.npy", gt.numpy())
                np.save(f"{save_path}/batch{batch_idx}_img{img_idx}_generated.npy", pred.numpy())

                # Metrics calculation
                gen_img = pred.unsqueeze(0).clamp(0, 1).float()
                gt_img = gt.unsqueeze(0).clamp(0, 1).float()

                if gen_img.dim() == 3:
                    gen_img = gen_img.unsqueeze(0)
                    gt_img = gt_img.unsqueeze(0)

                mse_val = F.mse_loss(gen_img, gt_img).item()
                psnr_val = psnr(gen_img, gt_img, data_range=1.0).item()
                ssim_val = ssim(gen_img, gt_img, data_range=1.0).item()

                mse_list.append(mse_val)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)

            except IndexError:
                print(f"Skipping sample {img_idx} in batch {batch_idx} due to indexing error.")
                continue

    # Verbose printing of evaluation results
    num_samples = len(mse_list)
    avg_mse = np.mean(mse_list)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print(f"\n--- Evaluation Results (over {num_samples} samples) ---")
    print(f"Total number of samples evaluated: {num_samples}")
    print(f"Average Mean Squared Error (MSE)   : {avg_mse:.6f}")
    print(f"Average Peak Signal-to-Noise Ratio (PSNR): {avg_psnr:.2f} dB")
    print(f"Average Structural Similarity (SSIM): {avg_ssim:.4f}")
    print("\n--- End of Evaluation ---")

# Generate and save samples for different sets
#generate_samples(train_loader, train_save, scheduler, net, noise_scheduler, start_epoch, max_batches=4)
generate_samples(valid_loader, valid_save, scheduler, net, noise_scheduler, start_epoch)
generate_samples(test_loader, test_save, scheduler, net, noise_scheduler, start_epoch)
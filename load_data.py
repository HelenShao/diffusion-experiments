import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  # Import PdfPages
import seaborn as sns
import random
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import torchvision

import math
import seaborn as sns
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")
import sys, os

# Add PartIII-project to the Python path
sys.path.append(os.path.abspath("/scratch/gpfs/hshao/PartIII-Project"))
from cmb_tools_main import *

# Add the path of 'folder1' to the system path
files_dir = "/scratch/gpfs/hshao/ILC_ML/reproducing/unet_vanilla_4_freq"
sys.path.append(os.path.abspath(files_dir))

import loss_funcs
import data
import common_vars
from common_vars import *

if __name__ == "__main__":
    # Function to set the seed for reproducibility
    def set_seed(seed):
        # Set seed for PyTorch, NumPy, and Python's random module
        print(f"Set seed to {seed} for torch, numpy, and python random")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # For multi-GPU, make sure all GPUs are initialized with the same seed
        torch.cuda.manual_seed_all(seed)
        
        # Ensure deterministic behavior for data loading across multiple workers
        def worker_init_fn(worker_id):
            seed_all = seed + worker_id
            np.random.seed(seed_all)
            random.seed(seed_all)
        
        return worker_init_fn
    
    # Set the seed
    seed = 4
    worker_init_fn = set_seed(seed)
    batch_size = 128
    
    # Use GPUs if avaiable
    if torch.cuda.is_available():
        print("CUDA Available")
        device = torch.device('cuda')
    else:
        print('CUDA Not Available')
        device = torch.device('cpu')
    
    # Load the dataset
    fashion_mnist_path = "/scratch/gpfs/hshao/ILC_ML/1_freq_scale/diffusion-demos/fashion_mnist/"
    train_dataset = torchvision.datasets.FashionMNIST(root=fashion_mnist_path, train=True, download=False, transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.FashionMNIST(root=fashion_mnist_path, train=False, download=False, transform=torchvision.transforms.ToTensor())
    
    # Split training dataset into train and validation sets
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
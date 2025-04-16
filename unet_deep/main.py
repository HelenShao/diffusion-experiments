# unet_deep: Deep UNet with 4 levels, 4 layers per block, requires image padding to 32x32
# Generated from the original main.py with modified UNet architecture
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import argparse
import os
import warnings
import numpy as np

#=========================================================================
# Global configurations
#=========================================================================
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
warnings.filterwarnings("ignore")

#=========================================================================
# Parse command-line arguments
#=========================================================================
parser = argparse.ArgumentParser(description='Fashion MNIST Diffusion Model')
parser.add_argument('--sample', action='store_true', help='Sample from a trained model instead of training')
parser.add_argument('--model_name', type=str, default='simple_case_model', help='Base name for saving/loading model files (without extension)')
parser.add_argument('--checkpoint', type=str, help='Legacy: Path to model checkpoint (overrides model_name if provided)')
args = parser.parse_args()

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Determine which model file to use
model_file = f"{args.model_name}.pt"
if args.checkpoint:
    print(f"Warning: --checkpoint is deprecated, use --model_name instead")
    model_file = args.checkpoint
print(f"Using model file: {model_file} for {'sampling' if args.sample else 'training'}")

#=========================================================================
# Data loading
#=========================================================================
with open("/scratch/gpfs/hshao/ILC_ML/1_freq_scale/220_dust_only/d9/d9_ell_200_no_norm/diffusion/fashion_mnist/load_data.py") as f:
    exec(f.read())

#=========================================================================
# Model definition
#=========================================================================
class ClassConditionedUnet(nn.Module):
  def __init__(self, num_classes=10, class_emb_size=4):
    super().__init__()

    # The embedding layer will map the class label to a vector of size class_emb_size
    self.class_emb = nn.Embedding(num_classes, class_emb_size)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=32,           # padded to 32x32
        in_channels=1 + class_emb_size, # Additional input channels for class cond.
        out_channels=1,           # the number of output channels
        layers_per_block=4,       # 4 layers per UNet block
        block_out_channels=(32, 64, 128, 256),  # 4 levels
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
    )

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels):
    # Shape of x:
    bs, ch, w, h = x.shape

    # class conditioning in right shape to add as additional input channels
    class_cond = self.class_emb(class_labels) # Map to embedding dimension
    class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
    # x is shape (bs, 1, 32, 32) and class_cond is now (bs, 4, 32, 32)

    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1) # (bs, 5, 32, 32)

    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, t).sample # (bs, 1, 32, 32)

#=========================================================================
# Create a noise scheduler
#=========================================================================
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, clip_sample=False) #, beta_schedule='squaredcos_cap_v2')

# Our network
net = ClassConditionedUnet().to(device)

# Print model architecture
print("\nModel Architecture:")
print(f"Layers per block: {net.model.config.layers_per_block}")
print(f"Block output channels: {net.model.config.block_out_channels}")
print(f"Down block types: {net.model.config.down_block_types}")
print(f"Up block types: {net.model.config.up_block_types}")
print(f"Total parameters: {sum(p.numel() for p in net.parameters())}")
print(f"Class embedding size: {net.class_emb.embedding_dim}")
print()

#=========================================================================
# Sampling or Training Mode
#=========================================================================
if args.sample:
    #=====================================================================
    # SAMPLING MODE
    #=====================================================================
    # Load model from checkpoint if in sampling mode
    if os.path.exists(model_file):
        print(f"Loading model from {model_file}")
        
        # Determine if this is a full model save with architecture or just weights
        checkpoint = torch.load(model_file, map_location=device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # This is a full model save with architecture details
            print("Found full model save with architecture details")
            
            # Print architecture details
            arch = checkpoint['architecture']
            print(f"Architecture: {arch['layers_per_block']} layers per block, " +
                  f"{len(arch['block_out_channels'])} resolution levels, " +
                  f"channels: {arch['block_out_channels']}")
            
            # If needed, you could reconstruct the model with these parameters
            # For now, we'll use the existing architecture and just load weights
            net.load_state_dict(checkpoint['state_dict'])
            
            # Print training parameters if available
            if 'training_params' in checkpoint:
                train_params = checkpoint['training_params']
                print(f"Trained for {train_params['epochs']} epochs using {train_params['optimizer']} " +
                      f"with learning rate {train_params['learning_rate']} on {train_params['device']}")
        else:
            # This is just the model weights
            print("Found model weights only (no architecture details)")
            net.load_state_dict(checkpoint)
        
        net.eval()
    else:
        print(f"Error: Model file {model_file} not found")
        exit(1)
        
    #Sampling some different digits:
    print("Sampling from trained diffusion model...")
    
    # Prepare random x to start from, plus some desired labels y
    x = torch.randn(80, 1, 32, 32).to(device)  # Changed from 28x28 to 32x32 to match model architecture
    y = torch.tensor([[i]*8 for i in range(10)]).flatten().to(device)
    
    # Sampling loop
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
    
        # Get model pred
        with torch.no_grad():
            residual = net(x, t, y)  # Again, note that we pass in our labels y
    
        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample
    
    # Get actual examples from the dataset for each class
    class_examples = {}
    for images, labels in test_loader:
        for img, label in zip(images, labels):
            label_int = label.item()
            if label_int not in class_examples and len(class_examples) < 10:
                class_examples[label_int] = img
            if len(class_examples) == 10:
                break
        if len(class_examples) == 10:
            break
    
    # Define class names for Fashion MNIST
    class_names = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }
    
    # Try to create visualization with LaTeX rendering
    try:
        # Enable LaTeX rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif"
        })
        
        # Create the figure with LaTeX labels
        fig, axes = plt.subplots(10, 10, figsize=(15, 15))
        plt.subplots_adjust(hspace=0.5)
        
        # Plot generated samples
        for i in range(10):
            for j in range(8):
                idx = i * 8 + j
                sample = x[idx].detach().cpu().clip(-1, 1).squeeze()
                sample = (sample + 1) / 2  # Convert from [-1, 1] to [0, 1]
                axes[i, j].imshow(sample, cmap='gray')
                axes[i, j].axis('off')
                
                # For the first sample of each class, add a title with the class number
                if j == 0:
                    axes[i, j].set_title(f"Class {i}")
        
        # Plot real examples and class names
        for i in range(10):
            # Real example in column 8
            real_example = class_examples[i].squeeze()
            axes[i, 8].imshow(real_example, cmap='gray')
            axes[i, 8].axis('off')
            
            # Text label in column 9
            axes[i, 9].text(0.5, 0.5, class_names[i], 
                           ha='center', va='center', 
                           fontsize=10, wrap=True)
            axes[i, 9].axis('off')
        
        # Add column headers with LaTeX
        fig.text(0.5, 0.98, r'Generated Samples', ha='center', fontsize=16)
        fig.text(0.85, 0.98, r'Real$\backslash$nSample', ha='center', fontsize=14)
        fig.text(0.95, 0.98, r'Class$\backslash$nName', ha='center', fontsize=14)
        
        plt.savefig('sampling_results_latex.png', bbox_inches='tight')
        print("Sampling results with LaTeX rendering saved to sampling_results_latex.png")
        
    except Exception as e:
        print(f"Error with LaTeX rendering: {e}")
        print("Falling back to standard rendering...")
        
        # Disable LaTeX for fallback
        plt.rcParams.update({"text.usetex": False})
        
        # Create the figure with standard labels
        fig, axes = plt.subplots(10, 10, figsize=(15, 15))
        plt.subplots_adjust(hspace=0.5)
        
        # Plot generated samples
        for i in range(10):
            for j in range(8):
                idx = i * 8 + j
                sample = x[idx].detach().cpu().clip(-1, 1).squeeze()
                sample = (sample + 1) / 2  # Convert from [-1, 1] to [0, 1]
                axes[i, j].imshow(sample, cmap='gray')
                axes[i, j].axis('off')
                
                # For the first sample of each class, add a title with the class number
                if j == 0:
                    axes[i, j].set_title(f"Class {i}")
        
        # Plot real examples and class names
        for i in range(10):
            # Real example in column 8
            real_example = class_examples[i].squeeze()
            axes[i, 8].imshow(real_example, cmap='gray')
            axes[i, 8].axis('off')
            
            # Text label in column 9
            axes[i, 9].text(0.5, 0.5, class_names[i], 
                           ha='center', va='center', 
                           fontsize=10, wrap=True)
            axes[i, 9].axis('off')
        
        # Add column headers
        fig.text(0.5, 0.98, 'Generated Samples', ha='center', fontsize=16)
        fig.text(0.85, 0.98, 'Real\nSample', ha='center', fontsize=14)
        fig.text(0.95, 0.98, 'Class\nName', ha='center', fontsize=14)
        
        plt.savefig('sampling_results.png', bbox_inches='tight')
        print("Sampling results saved to sampling_results.png")
    
    print("Sampling complete.")
    
else:
    #=====================================================================
    # TRAINING MODE
    #=====================================================================
    # How many runs through the data should we do?
    n_epochs = 30
    
    # Our loss function
    loss_fn = nn.MSELoss()
    
    # The optimizer
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    # Keeping a record of the losses for later viewing
    losses = []
    val_losses = []
    test_losses = []
    
    # Create log file for real-time tracking
    with open('training_log.txt', 'w') as f:
        f.write("Epoch,Train,Valid,Test\n")
    
    #---------------------------------------------------------------------
    # The training loop
    #---------------------------------------------------------------------
    for epoch in range(n_epochs):
        for x, y in tqdm(train_loader):
    
            # Get some data and prepare the corrupted version
            x = x.to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
            # Pad images from 28x28 to 32x32
            x = torch.nn.functional.pad(x, (2, 2, 2, 2), "constant", -1)  # Pad with -1 (black)
            y = y.to(device)
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
    
            # Get the model prediction
            pred = net(noisy_x, timesteps, y) # Note that we pass in the labels y
    
            # Calculate the loss
            loss = loss_fn(pred, noise) # How close is the output to the noise
    
            # Backprop and update the params:
            opt.zero_grad()
            loss.backward()
            opt.step()
    
            # Store the loss for later
            losses.append(loss.item())
    
        #-------------------------------------------------------------------
        # Validation loop
        #-------------------------------------------------------------------
        net.eval()  # Set model to evaluation mode
        epoch_val_losses = []
        with torch.no_grad():  # No gradients needed for validation
            for x, y in tqdm(valid_loader, desc="Validation"):
                x = x.to(device) * 2 - 1
                # Pad images from 28x28 to 32x32
                x = torch.nn.functional.pad(x, (2, 2, 2, 2), "constant", -1)  # Pad with -1 (black)
                y = y.to(device)
                noise = torch.randn_like(x)
                timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
                noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
                
                pred = net(noisy_x, timesteps, y)
                val_loss = loss_fn(pred, noise)
                epoch_val_losses.append(val_loss.item())
        
        # Calculate average validation loss for this epoch
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)
        
        #-------------------------------------------------------------------
        # Test loop
        #-------------------------------------------------------------------
        epoch_test_losses = []
        with torch.no_grad():  # No gradients needed for testing
            for x, y in tqdm(test_loader, desc="Testing"):
                x = x.to(device) * 2 - 1
                # Pad images from 28x28 to 32x32
                x = torch.nn.functional.pad(x, (2, 2, 2, 2), "constant", -1)  # Pad with -1 (black)
                y = y.to(device)
                noise = torch.randn_like(x)
                timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
                noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
                
                pred = net(noisy_x, timesteps, y)
                test_loss = loss_fn(pred, noise)
                epoch_test_losses.append(test_loss.item())
        
        # Calculate average test loss for this epoch
        avg_test_loss = sum(epoch_test_losses) / len(epoch_test_losses)
        test_losses.append(avg_test_loss)
        
        # Print out the average of the last 100 loss values to get an idea of progress:
        avg_loss = sum(losses[-100:])/100
        print(f'Finished epoch {epoch}. Train loss: {avg_loss:05f}, Validation loss: {avg_val_loss:05f}, Test loss: {avg_test_loss:05f}')
        
        # Log losses to file in real time
        with open('training_log.txt', 'a') as f:
            f.write(f"{epoch},{avg_loss:.6f},{avg_val_loss:.6f},{avg_test_loss:.6f}\n")
            f.flush()  # Ensure it's written immediately
        
        # Set model back to training mode
        net.train()
    
    #---------------------------------------------------------------------
    # Save model and results
    #---------------------------------------------------------------------
    # Create a dictionary with both model state and architecture parameters
    model_info = {
        'state_dict': net.state_dict(),
        'architecture': {
            'num_classes': 10,
            'class_emb_size': 4,
            'layers_per_block': 4,
            'block_out_channels': (32, 64, 128, 256),
            'down_block_types': ('DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
            'up_block_types': ('AttnUpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D'),
        },
        'training_params': {
            'epochs': n_epochs,
            'optimizer': 'Adam',
            'learning_rate': 1e-3,
            'device': device,
        }
    }
    
    # Save the comprehensive model information
    torch.save(model_info, f'{args.model_name}_full.pt')
    print(f"Model with architecture details saved to {args.model_name}_full.pt")
    
    # Also save just the state dict for compatibility with existing code
    torch.save(net.state_dict(), f'{args.model_name}.pt')
    print(f"Model state dict saved to {args.model_name}.pt (for compatibility)")
    
    # Save losses as numpy files
    np.save(f'{args.model_name}_train_losses.npy', np.array(losses))
    np.save(f'{args.model_name}_valid_losses.npy', np.array(val_losses))
    np.save(f'{args.model_name}_test_losses.npy', np.array(test_losses))
    print(f"Losses saved as numpy files: {args.model_name}_train_losses.npy, {args.model_name}_valid_losses.npy, {args.model_name}_test_losses.npy")
    
    #---------------------------------------------------------------------
    # Visualization
    #---------------------------------------------------------------------
    # Try plotting with LaTeX rendering
    try:
        # Enable LaTeX rendering
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif"
        })
        
        # View the loss curves with LaTeX
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label=r'Training Loss')
        plt.plot([sum(losses[i:i+100])/100 for i in range(0, len(losses), 100)], label=r'Smoothed Training Loss')
        plt.plot(range(0, len(losses), len(losses)//n_epochs), val_losses, label=r'Validation Loss')
        plt.plot(range(0, len(losses), len(losses)//n_epochs), test_losses, label=r'Test Loss')
        plt.xlabel(r'Iterations')
        plt.ylabel(r'Loss')
        plt.title(r'Training, Validation and Test Losses')
        plt.legend()
        plt.savefig('loss_curves_latex.png')
        print("Loss curves with LaTeX rendering saved as loss_curves_latex.png")
        
        # View the raw loss curve with LaTeX
        plt.figure()
        plt.plot(losses)
        plt.xlabel(r'Iterations')
        plt.ylabel(r'Loss')
        plt.title(r'Raw Training Loss')
        plt.savefig('loss_curve_latex.png')
        print("Raw loss curve with LaTeX rendering saved as loss_curve_latex.png")
        
    except Exception as e:
        print(f"Error with LaTeX rendering: {e}")
        print("Falling back to standard rendering...")
        
        # Disable LaTeX for fallback
        plt.rcParams.update({"text.usetex": False})
        
        # View the loss curves with standard rendering
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        plt.plot([sum(losses[i:i+100])/100 for i in range(0, len(losses), 100)], label='Smoothed Training Loss')
        plt.plot(range(0, len(losses), len(losses)//n_epochs), val_losses, label='Validation Loss')
        plt.plot(range(0, len(losses), len(losses)//n_epochs), test_losses, label='Test Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training, Validation and Test Losses')
        plt.legend()
        plt.savefig('loss_curves.png')
        print("Loss curves saved as loss_curves.png")
        
        # View the raw loss curve with standard rendering
        plt.figure()
        plt.plot(losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Raw Training Loss')
        plt.savefig('loss_curve.png')
        print("Raw loss curve saved as loss_curve.png")
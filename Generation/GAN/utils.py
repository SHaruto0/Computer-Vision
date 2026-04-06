import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.utils as vutils

BASE_PATH = Path(__file__).resolve().parent

def set_seed(seed: int = 42):
    random.seed(seed)                     # Python random
    np.random.seed(seed)                  # NumPy
    torch.manual_seed(seed)               # CPU
    torch.cuda.manual_seed(seed)          # GPU
    torch.cuda.manual_seed_all(seed)      # All GPUs
    torch.backends.cudnn.deterministic = True  # Deterministic convs
    torch.backends.cudnn.benchmark = False     # Disable auto-tuner for reproducibility
    print(f"Random seed set to {seed}")

def save_training_plots(
    model_name,
    loss1_history,
    loss2_history,
    training1_time,
    training2_time,
    output_dir="outputs/plots"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(loss1_history) + 1)

    # Loss plot
    plt.figure()
    plt.plot(epochs, loss1_history, label="Generator Loss")
    plt.plot(epochs, loss2_history, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f"{model_name}_loss.png")
    plt.close()

    # Time plot
    plt.figure()
    plt.plot(epochs, training1_time, label="Generator Training Time (s)")
    plt.plot(epochs, training2_time, label="Discriminator Training Time (s)")
    plt.xlabel("Number of Batches")
    plt.ylabel("Seconds")
    plt.title(f"Training Time - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f"{model_name}_training_time.png")
    plt.close()

    print(f"\nPlots saved to: {output_dir.resolve()}")

def summarize_checkpoint_times(ckpt_path):
    ckpt = torch.load(BASE_PATH / Path("outputs/checkpoints/") / ckpt_path, map_location="cpu")
    
    # Check if epoch_times exists
    if "epoch_times" not in ckpt:
        print("Checkpoint does not contain 'epoch_times'.")
        return None

    epoch_times = ckpt["epoch_times"]
    total_time = sum(epoch_times)
    avg_time = total_time / len(epoch_times)

    def format_hms(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h {m}m {s}s"

    print(f"Average epoch time: {format_hms(avg_time)}")
    print(f"Total training time: {format_hms(total_time)}")
    
    return avg_time, total_time

def print_model_size(model, device, input_size=(3, 224, 224)):
    """
    Prints the number of parameters and approximate memory size of a PyTorch model.

    Args:
        model (nn.Module): The model to inspect.
        input_size (tuple): Input size (C, H, W), default is (3, 224, 224).
    """
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate model size (bytes)
    # float32 = 4 bytes
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 ** 2)

    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Approximate size: {size_mb:.2f} MB")

    # Measure memory usage
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    dummy_input = torch.randn(1, *input_size).to(device)
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)

    if device.type == 'cuda':
        mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
        mem_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"GPU Memory Allocated: {mem_alloc:.2f} MB")
        print(f"GPU Peak Memory Allocated: {mem_peak:.2f} MB")

def save_progression_images(netG, fixed_noise, epoch):
    netG.eval()
    
    with torch.no_grad():
        fake = netG(fixed_noise).cpu()
    
    # make grid (8x8)
    grid = vutils.make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))
    
    # create directory if it doesn't exist
    save_dir = Path("outputs/plots/progression")
    save_dir.mkdir(parents=True, exist_ok=True) 
    
    # save directly
    save_path = save_dir / f"epoch_{epoch:03d}.png"
    vutils.save_image(grid, save_path)
    
    netG.train()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
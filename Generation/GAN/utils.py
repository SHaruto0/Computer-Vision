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

def analyze_dcgan_performance(checkpoint, plots_dir):
    def format_hms(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h {m}m {s}s"

    print("--- Model Statistics & Training Analysis ---")
    
    # 1. Parameter and Size Analysis
    total_params_g = sum(p.numel() for p in checkpoint["model_state_G"].values())
    total_params_d = sum(p.numel() for p in checkpoint["model_state_D"].values())
    size_mb_g = (total_params_g * 4) / (1024 ** 2)
    size_mb_d = (total_params_d * 4) / (1024 ** 2)

    print(f"Generator - Total Parameters: {total_params_g:,} | Approx Size: {size_mb_g:.2f} MB")
    print(f"Discriminator - Total Parameters: {total_params_d:,} | Approx Size: {size_mb_d:.2f} MB")

    # 2. History Extraction
    g_loss = checkpoint.get("G_loss_history", [])
    d_loss = checkpoint.get("D_loss_history", [])
    g_times = checkpoint.get("G_epoch_times", [])
    d_times = checkpoint.get("D_epoch_times", [])

    # 3. Time Analysis
    if g_times and d_times:
        print(f"Generator Mean Time: {np.mean(g_times):.4f}s (Std: {np.std(g_times):.4f}s)")
        print(f"Discriminator Mean Time: {np.mean(d_times):.4f}s (Std: {np.std(d_times):.4f}s)")
        total_time = sum(g_times) + sum(d_times)
        print(f"Total Training Time: {format_hms(total_time)}")

    # 4. Stability Analysis
    if len(g_loss) > 10:
        split = len(g_loss) // 10
        print(f"Generator Loss Variance (Start): {np.var(g_loss[:split]):.6f}")
        print(f"Generator Loss Variance (End): {np.var(g_loss[-split:]):.6f}")

    print(f"Last Recorded Epoch: {checkpoint.get('epoch', 'N/A')}")
    if g_loss:
        print(f"Generator Loss Spike: {max(g_loss):.4f} at batch index {np.argmax(g_loss)}")

    # 5. Plotting Losses and Times
    if g_loss and d_loss:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss Plot
        ax1.plot(g_loss, label="G Loss", alpha=0.7)
        ax1.plot(d_loss, label="D Loss", alpha=0.7)
        ax1.set_title("Training Loss History")
        ax1.set_xlabel("Batch Iteration")
        ax1.set_ylabel("Loss")
        ax1.legend()
        
        # Time Plot
        if g_times:
            ax2.plot(g_times, label="G Batch Time", color='green', alpha=0.5)
            ax2.plot(d_times, label="D Batch Time", color='red', alpha=0.5)
            ax2.set_title("Processing Time per Batch")
            ax2.set_xlabel("Batch Iteration")
            ax2.set_ylabel("Seconds")
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"metrics_epoch_{checkpoint['epoch']}.png")
        print(f"Metrics plot saved to: {plots_dir}")
        
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
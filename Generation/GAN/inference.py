import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils

from models.generator import Generator
from models.discriminator import Discriminator
from utils import BASE_PATH, analyze_dcgan_performance, set_seed
from configs.dcgan import DCGAN_CONFIG

def inference(model_path):
    # Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create output directory for plots
    plots_dir = BASE_PATH / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    latent_space = torch.randn(64, DCGAN_CONFIG["latent_dim"], 1, 1, device=device)

    checkpoint = torch.load(BASE_PATH / "outputs" / "checkpoints" / model_path, map_location=device)

    analyze_dcgan_performance(checkpoint, plots_dir)
    
    # Model
    netG = Generator(
        latent_dim=DCGAN_CONFIG["latent_dim"], 
        img_channels=3, 
        feature_maps=DCGAN_CONFIG["feature_maps"]
    ).to(device)
    netD = Discriminator(
        img_channels=3, 
        feature_maps=DCGAN_CONFIG["feature_maps"]
    ).to(device)

    netG.load_state_dict(checkpoint["model_state_G"])
    netD.load_state_dict(checkpoint["model_state_D"])
    netG.eval()
    netD.eval()

    # Memory Tracking
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            dummy_z = torch.randn(1, DCGAN_CONFIG["latent_dim"], 1, 1, device=device)
            _ = netG(dummy_z)
        mem_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Inference GPU Peak Memory: {mem_peak:.2f} MB")

    # Image Generation
    print("\nGenerating sample grid...")
    latent_space = torch.randn(64, DCGAN_CONFIG["latent_dim"], 1, 1, device=device)
    with torch.no_grad():
        fake_images = netG(latent_space).detach().cpu()

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title(f"Generated Images - Epoch {checkpoint['epoch']}")
    grid = vutils.make_grid(fake_images, padding=2, normalize=True)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    
    save_img_path = plots_dir / f"inference_grid_epoch_{checkpoint['epoch']}.png"
    plt.savefig(save_img_path)
    plt.show()
    print(f"Inference grid saved to: {save_img_path}")

if __name__ == "__main__":
    model_path = "epoch_100.pth"
    inference(model_path)
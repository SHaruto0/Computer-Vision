import time
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.generator import Generator
from models.discriminator import Discriminator
from dataset import CelebADataset, build_transforms, calculate_mean_std
from utils import save_progression_images, set_seed, save_training_plots, BASE_PATH, weights_init

from configs.data import DATA_CFG
from configs.dcgan import DCGAN_CONFIG

def train():
    """
    Train a GAN model on the CelebA dataset.
    """
    # Config
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    FIXED_NOISE = torch.randn(64, DCGAN_CONFIG["latent_dim"], 1, 1, device=device)
    REAL_LABEL = 1.
    FAKE_LABEL = 0.

    # Datasets & loaders
    train_datasets = CelebADataset(
        root=DATA_CFG["root"], 
        transform=build_transforms(DATA_CFG["image_size"]))
    train_loader = DataLoader(train_datasets,
                              batch_size=DATA_CFG["batch_size"], 
                              shuffle=True, 
                              num_workers=DATA_CFG["num_workers"],
                              drop_last=False)
    
    # Model, loss, optimizer
    netG = Generator(
        latent_dim=DCGAN_CONFIG["latent_dim"], 
        img_channels=3, 
        feature_maps=DCGAN_CONFIG["feature_maps"]).to(device)
    netD = Discriminator(
        img_channels=3, 
        feature_maps=DCGAN_CONFIG["feature_maps"]).to(device)
    
    netG.apply(weights_init)
    netD.apply(weights_init)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=DCGAN_CONFIG.get("lr_g", 0.0002), betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=DCGAN_CONFIG.get("lr_d", 0.0002), betas=(0.5, 0.999))

    # Checkpoint
    num_epochs = int(DCGAN_CONFIG.get("epochs", 50))
    output_dir = BASE_PATH / Path("outputs/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1

    G_loss_history = []
    D_loss_history = []
    G_training_times = []
    D_training_times = []

    if DCGAN_CONFIG.get("start_from", None) is not None and not isinstance(DCGAN_CONFIG.get("start_from", None), str):
        ckpt_epoch = int(DCGAN_CONFIG["start_from"])
        ckpt_path = output_dir / f"epoch_{ckpt_epoch}.pth"

        checkpoint = torch.load(ckpt_path, map_location=device)

        netG.load_state_dict(checkpoint["model_state_G"])
        netD.load_state_dict(checkpoint["model_state_D"])
        optimizerG.load_state_dict(checkpoint["optimizer_state_G"])
        optimizerD.load_state_dict(checkpoint["optimizer_state_D"])

        G_loss_history = checkpoint.get("G_loss_history", [])
        D_loss_history = checkpoint.get("D_loss_history", [])
        G_training_times = checkpoint.get("G_epoch_times", [])
        D_training_times = checkpoint.get("D_epoch_times", [])

        start_epoch = checkpoint["epoch"] + 1

        print(f"Resumed from epoch {start_epoch}")
    
    # Training Loop
    for epoch in range(start_epoch, num_epochs+1):
        
        netD.train()
        netG.train()
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            images = images.to(device)
            b_size = images.size(0)

            # Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            start_time_D = time.time()
            netD.zero_grad()

            # Real images
            label = torch.full((b_size,), REAL_LABEL, device=device)
            output_real = netD(images).view(-1)
            loss_D_real = criterion(output_real, label)
            loss_D_real.backward()
            D_x = output_real.mean().item()

            # Fake images
            noise = torch.randn(b_size, DCGAN_CONFIG["latent_dim"], 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(FAKE_LABEL)
            output_fake = netD(fake_images.detach()).view(-1)
            loss_D_fake = criterion(output_fake, label)
            loss_D_fake.backward()
            D_G_z1 = output_fake.mean().item()

            # Total discriminator loss
            loss_D = loss_D_real + loss_D_fake
            optimizerD.step()
            D_training_times.append(time.time() - start_time_D)

            # Train Generator: maximize log(D(G(z)))
            start_time_G = time.time()
            netG.zero_grad()

            label.fill_(REAL_LABEL)
            output_gen = netD(fake_images).view(-1)
            loss_G = criterion(output_gen, label)
            loss_G.backward()
            D_G_z2 = output_gen.mean().item()
            optimizerG.step()
            G_training_times.append(time.time() - start_time_G)

            D_loss_history.append(loss_D.item())
            G_loss_history.append(loss_G.item())

        save_progression_images(netG, FIXED_NOISE, epoch)

        print(f"Epoch {epoch}/{num_epochs} | D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")
        
        # Save plots
        save_training_plots(
            model_name="DCGAN",
            loss1_history=G_loss_history,
            loss2_history=D_loss_history,
            training1_time=G_training_times,
            training2_time=D_training_times,
            output_dir="outputs/plots/metrics"
        )

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_G": netG.state_dict(),
            "model_state_D": netD.state_dict(),
            "optimizer_state_G": optimizerG.state_dict(),
            "optimizer_state_D": optimizerD.state_dict(),

            # history
            "G_loss_history": G_loss_history,
            "D_loss_history": D_loss_history,
            "G_epoch_times": G_training_times,
            "D_epoch_times": D_training_times
        }

        ckpt_path = output_dir / f"epoch_{epoch}.pth"
        torch.save(ckpt, ckpt_path)

    print("\nTraining Summary")
    print(f"Total time: {sum(G_training_times + D_training_times):.2f} seconds")
    print(f"Avg time/batch: {np.mean(G_training_times + D_training_times):.2f} seconds")
    print(f"Min batch time: {np.min(G_training_times + D_training_times):.2f} seconds")
    print(f"Max batch time: {np.max(G_training_times + D_training_times):.2f} seconds")

if __name__ == "__main__":
    train()
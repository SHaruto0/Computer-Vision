import time
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.resnet import ResNet50, ResNet101, ResNet152
from dataset import SportsDataset, build_transforms
from utils import set_seed, save_training_plots, BASE_PATH, DATA_CFG, RESNET_CFG

def train(model_name):
    """
    Train a ResNet model on the sports dataset.

    Args:
        model_name (str): One of "resnet50", "resnet101", "resnet152"
    """
    # Config
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Datasets & loaders
    train_dataset = SportsDataset(
        root=DATA_CFG["root"], 
        split="train", 
        transform=build_transforms(DATA_CFG["image_size"], train=True))
    val_dataset = SportsDataset(
        root=DATA_CFG["root"], 
        split="valid", 
        transform=build_transforms(DATA_CFG["image_size"], train=False))

    train_loader = DataLoader(train_dataset, 
                              batch_size=DATA_CFG["batch_size"], 
                              shuffle=True, 
                              num_workers=DATA_CFG["num_workers"],
                              drop_last=True)
    val_loader = DataLoader(val_dataset, 
                              batch_size=DATA_CFG["batch_size"], 
                              shuffle=True, 
                              num_workers=DATA_CFG["num_workers"],
                              drop_last=True)

    # Model, loss, optimizer
    if model_name == "resnet50":
        model = ResNet50(img_channels=3, num_classes=DATA_CFG.get("num_classes", 100)).to(device)
    elif model_name == "resnet101":
        model = ResNet101(img_channels=3, num_classes=DATA_CFG.get("num_classes", 100)).to(device)
    elif model_name == "resnet152":
        model = ResNet152(img_channels=3, num_classes=DATA_CFG.get("num_classes", 100)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(RESNET_CFG.get("lr", 0.001)),
        momentum=float(RESNET_CFG.get("momentum", 0.9)),
        weight_decay=float(RESNET_CFG.get("weight_decay", 1e-4)),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(RESNET_CFG.get("step_size", 30)),
        gamma=float(RESNET_CFG.get("gamma", 0.1)),
    )

    # Checkpoint
    num_epochs = RESNET_CFG.get("epochs", 50)
    output_dir = BASE_PATH / Path("outputs/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_acc = 0.0

    loss_history = []
    train_acc_history = []
    val_acc_history = []
    epoch_times = []

    if RESNET_CFG.get("start_from", None) is not None and not isinstance(RESNET_CFG.get("start_from", None), str):
        ckpt_epoch = int(RESNET_CFG["start_from"])
        ckpt_path = output_dir / f"{model_name}_epoch_{ckpt_epoch}.pth"

        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])

        best_acc = checkpoint.get("best_acc", 0.0)

        loss_history = checkpoint.get("loss_history", [])
        train_acc_history = checkpoint.get("train_acc_history", [])
        val_acc_history = checkpoint.get("val_acc_history", [])
        epoch_times = checkpoint.get("epoch_times", [])

        start_epoch = checkpoint["epoch"] + 1

        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, num_epochs+1):
        start_time = time.time()

        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train
        loss_history.append(epoch_loss)
        train_acc_history.append(train_acc)

        # Validations
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"[Val] Epoch {epoch}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        val_acc_history.append(val_acc)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Time: {epoch_time:.2f}s")

        # Save plots
        save_training_plots(
            model_name=model_name,
            loss_history=loss_history,
            train_acc_history=train_acc_history,
            val_acc_history=val_acc_history,
            epoch_times=epoch_times,
            output_dir="outputs/plots"
        )

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_acc": best_acc,
        
            # histories
            "loss_history": loss_history,
            "train_acc_history": train_acc_history,
            "val_acc_history": val_acc_history,
            "epoch_times": epoch_times,
        }

        ckpt_path = output_dir / f"{model_name}_epoch_{epoch}.pth"
        torch.save(ckpt, ckpt_path)

        if val_acc > best_acc:
            best_acc = val_acc
            best_ckpt_path = output_dir / f"{model_name}_best.pth"
            torch.save(ckpt, best_ckpt_path)
            print(f"Saved best model to {best_ckpt_path}")
        
        scheduler.step()
    
    print("\nTraining Summary")
    print(f"Best Val Accuracy: {best_acc*100:.2f}%")
    print(f"Total time: {sum(epoch_times):.2f} seconds")
    print(f"Avg time/epoch: {np.mean(epoch_times):.2f} seconds")
    print(f"Min epoch time: {np.min(epoch_times):.2f} seconds")
    print(f"Max epoch time: {np.max(epoch_times):.2f} seconds")

if __name__ == "__main__":
    model_name = "resnet50"
    train(model_name)
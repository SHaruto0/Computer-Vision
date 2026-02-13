import time
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.vgg import VGG16
from dataset import ImageNetDataset, build_transforms
from utils import set_seed, save_training_plots, BASE_PATH, DATA_CFG, VGG_CFG

def train():
    # Config
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Datasets & loaders
    train_dataset = ImageNetDataset(
        root=DATA_CFG["root"], 
        split="train", 
        transform=build_transforms(DATA_CFG["image_size"], train=True))
    test_dataset = ImageNetDataset(
        root=DATA_CFG["root"], 
        split="test",  
        transform=build_transforms(DATA_CFG["image_size"], train=False))

    train_loader = DataLoader(train_dataset, 
                              batch_size=DATA_CFG["batch_size"], 
                              shuffle=True, 
                              num_workers=DATA_CFG["num_workers"],
                              drop_last=True)
    test_loader = DataLoader(test_dataset, 
                              batch_size=DATA_CFG["batch_size"], 
                              shuffle=False, 
                              num_workers=DATA_CFG["num_workers"],
                              drop_last=True)

    # Model, loss, optimizer
    model = VGG16(num_classes=DATA_CFG.get("num_classes", 1000)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(VGG_CFG.get("lr", 0.001)),
        momentum=float(VGG_CFG.get("momentum", 0.9)),
        weight_decay=float(VGG_CFG.get("weight_decay", 1e-4)),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(VGG_CFG.get("step_size", 30)),
        gamma=float(VGG_CFG.get("gamma", 0.1)),
    )

    # Checkpoint
    num_epochs = VGG_CFG.get("epochs", 50)
    output_dir = BASE_PATH / Path("outputs/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_acc = 0.0

    loss_history = []
    train_acc_history = []
    test_acc_history = []
    epoch_times = []

    if VGG_CFG.get("start_from", None) is not None and not isinstance(VGG_CFG.get("start_from", None), str):
        ckpt_epoch = int(VGG_CFG["start_from"])
        ckpt_path = output_dir / f"vgg16_epoch_{ckpt_epoch}.pth"

        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])

        best_acc = checkpoint.get("best_acc", 0.0)

        loss_history = checkpoint.get("loss_history", [])
        train_acc_history = checkpoint.get("train_acc_history", [])
        test_acc_history = checkpoint.get("test_acc_history", [])
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
        for images, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}"):
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

        # Test
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"[Test] Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

        test_acc = correct_test / total_test
        test_acc_history.append(test_acc)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Time: {epoch_time:.2f}s")

        # Save plots
        save_training_plots(
            loss_history=loss_history,
            train_acc_history=train_acc_history,
            test_acc_history=test_acc_history,
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
            "test_acc_history": test_acc_history,
            "epoch_times": epoch_times,
        }

        ckpt_path = output_dir / f"vgg16_epoch_{epoch}.pth"
        torch.save(ckpt, ckpt_path)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_ckpt_path = output_dir / "vgg16_best.pth"
            torch.save(ckpt, best_ckpt_path)
            print(f"Saved best model to {best_ckpt_path}")

        scheduler.step()
    
    print("\nTraining Summary")
    print(f"Best Test Accuracy: {best_acc*100:.2f}%")
    print(f"Total time: {sum(epoch_times):.2f} seconds")
    print(f"Avg time/epoch: {np.mean(epoch_times):.2f} seconds")
    print(f"Min epoch time: {np.min(epoch_times):.2f} seconds")
    print(f"Max epoch time: {np.max(epoch_times):.2f} seconds")

if __name__ == "__main__":
    train()
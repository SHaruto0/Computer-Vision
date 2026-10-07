import time
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.mobilenet import MobileNetV4ConvS
from models.resnet import ResNet50, ResNet101, ResNet152
from models.densenet import DenseNet121, DenseNet169, DenseNet201, DenseNet264
from dataset import ImageNetDataset, build_transforms
from utils import set_seed, save_training_plots, BASE_PATH

from configs.data import DATA_CFG
from configs.mobilenet import MOBILENET_CONFIG
from configs.resnet import RESNET_CONFIG
from configs.densenet import DENSENET_CONFIG

from models.registry import MODEL_REGISTRY

def build_optimizer(model, cfg):
    name = cfg.get("optimizer", "adamw").lower()
    if name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=float(cfg.get("lr", 2e-3)),
            weight_decay=float(cfg.get("weight_decay", 0.01)),
        )
    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=float(cfg.get("lr", 0.1)),
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer: {name}")

def train(model_name):
    """
    Train Mobile Network on ImageNet dataset.

    Args:
        model_name (str): "conv-s"
    """
    # Config
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Options: {list(MODEL_REGISTRY)}")
    build_fn, MODEL_CFG = MODEL_REGISTRY[model_name]

    # Datasets & loaders
    train_dataset = ImageNetDataset(
        root=DATA_CFG["root"], 
        split="train", 
        transform=build_transforms(DATA_CFG["image_size"], train=True))
    test_dataset = ImageNetDataset(
        root=DATA_CFG["root"], 
        split="test", 
        transform=build_transforms(DATA_CFG["image_size"], train=False))

    loader_kwargs = dict(
        batch_size=MODEL_CFG.get("batch_size", DATA_CFG["batch_size"]),
        num_workers=DATA_CFG["num_workers"],
        pin_memory=True,
        persistent_workers=DATA_CFG["num_workers"] > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    # Model, loss, optimizer
    num_classes = DATA_CFG.get("num_classes", 100)
    model = build_fn(num_classes).to(device)
    print(f"{model_name}: {sum(p.numel() for p in model.parameters())/1e6:.2f} M params")

    num_epochs = MODEL_CFG.get("epochs", 100)
    warmup_epochs = int(MODEL_CFG.get("warmup_epochs", 5))

    criterion = nn.CrossEntropyLoss(label_smoothing=float(MODEL_CFG.get("label_smoothing", 0.0)))
    optimizer = build_optimizer(model, MODEL_CFG)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=warmup_epochs),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs),
        ],
        milestones=[warmup_epochs],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Checkpoint
    output_dir = BASE_PATH / Path("outputs/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_acc = 0.0

    loss_history = []
    train_acc_history = []
    test_acc_history = []
    epoch_times = []

    start_from = MODEL_CFG.get("start_from", None)
    if start_from is not None and not isinstance(start_from, str):
        ckpt_path = output_dir / f"{model_name}_last.pth"
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])

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
        for images, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{num_epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss = running_loss / total_train
        train_acc = correct_train / total_train
        loss_history.append(epoch_loss)
        train_acc_history.append(train_acc)

        # Testing
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"[Test] Epoch {epoch}/{num_epochs}"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    outputs = model(images)

                correct_test += (outputs.argmax(1) == labels).sum().item()
                total_test += labels.size(0)

        test_acc = correct_test / total_test
        test_acc_history.append(test_acc)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Test Acc: {test_acc*100:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save plots
        save_training_plots(
            model_name=model_name,
            loss_history=loss_history,
            train_acc_history=train_acc_history,
            test_acc_history=test_acc_history,
            epoch_times=epoch_times,
            output_dir="outputs/plots"
        )

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_acc": best_acc,
            "classes": train_dataset.classes,

            # histories
            "loss_history": loss_history,
            "train_acc_history": train_acc_history,
            "test_acc_history": test_acc_history,
            "epoch_times": epoch_times,
        }

        torch.save(ckpt, output_dir / f"{model_name}_last.pth")
        if is_best:
            torch.save(ckpt, output_dir / f"{model_name}_best.pth")
            print(f"New best: {best_acc*100:.2f}%")

        scheduler.step()
    
    print("\nTraining Summary")
    print(f"Best Test Accuracy: {best_acc*100:.2f}%")
    print(f"Total time: {sum(epoch_times):.2f} seconds")
    print(f"Avg time/epoch: {np.mean(epoch_times):.2f} seconds")
    print(f"Min epoch time: {np.min(epoch_times):.2f} seconds")
    print(f"Max epoch time: {np.max(epoch_times):.2f} seconds")

if __name__ == "__main__":
    import sys
    train(sys.argv[1] if len(sys.argv) > 1 else "conv-s")
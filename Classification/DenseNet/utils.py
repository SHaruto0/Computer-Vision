import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent

def load_yaml(path):
    path = BASE_PATH / path
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

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
    loss_history,
    train_acc_history,
    test_acc_history,
    epoch_times,
    output_dir="outputs/plots"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(loss_history) + 1)

    # Loss plot
    plt.figure()
    plt.plot(epochs, loss_history, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f"{model_name}_loss.png")
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, train_acc_history, label="Train Accuracy")
    plt.plot(epochs, test_acc_history, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f"{model_name}_accuracy.png")
    plt.close()

    # Time per epoch plot
    plt.figure()
    plt.plot(epochs, epoch_times, label="Time per Epoch (s)")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.title("Epoch Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f"{model_name}_epoch_time.png")
    plt.close()

    print(f"\nPlots saved to: {output_dir.resolve()}")

DATA_CFG = load_yaml("configs/data.yaml")
DENSENET_CFG = load_yaml("configs/densenet.yaml")
RESNET_CFG = load_yaml("configs/resnet.yaml")
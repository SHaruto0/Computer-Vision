import torch
from torch.utils.data import DataLoader

import csv
import random
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

from models.wrn import WRN28_10
from models.resnet import ResNet50, ResNet101, ResNet152
from dataset import CIFAR100, build_transforms
from utils import set_seed, summarize_checkpoint_times, BASE_PATH, DATA_CFG

def inference(params_path, topk=(1,5)):
    model_name = "_".join(params_path.split("_")[:2])
    # Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create output directory for plots
    plots_dir = BASE_PATH / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create output directory for metrics
    metric_dir = BASE_PATH / "outputs" / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    # Data
    test_dataset = CIFAR100(
        root=DATA_CFG["root"], 
        split="test",  
        transform=build_transforms(DATA_CFG["image_size"], train=False)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=1
    )

    idx_to_class = test_dataset.idx_to_class

    # Model, loss, optimizer
    if model_name == "wrn28_10":
        model = WRN28_10(img_channels=3, num_classes=DATA_CFG.get("num_classes", 100), withDropout=True).to(device)
    elif "resnet50" in model_name:
        model_name = params_path.split("_")[0]
        model = ResNet50(img_channels=3, num_classes=DATA_CFG.get("num_classes", 100)).to(device)
    elif "resnet101" in model_name:
        model_name = params_path.split("_")[0]
        model = ResNet101(img_channels=3, num_classes=DATA_CFG.get("num_classes", 100)).to(device)
    elif "resnet152" in model_name:
        model_name = params_path.split("_")[0]
        model = ResNet152(img_channels=3, num_classes=DATA_CFG.get("num_classes", 100)).to(device)
   
    ckpt_path = BASE_PATH / "outputs" / "checkpoints" / params_path
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Metrics Tracking
    total = 0
    topk_correct = [0] * len(topk)
    confusion_counter = Counter()      # (true, pred)
    per_class_total = Counter()        # true
    per_class_correct = Counter()      # true & correct

    # Inference Loop
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"[Inference]"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            # Top-k accuracy
            for i, k in enumerate(topk):
                topk_preds = torch.topk(probs, k, dim=1).indices
                topk_correct[i] += (
                    topk_preds == labels.unsqueeze(1)
                ).any(dim=1).sum().item()

            # Top-1 predictions
            preds = torch.argmax(probs, dim=1)

            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                per_class_total[t] += 1
                if t == p:
                    per_class_correct[t] += 1
                else:
                    confusion_counter[(t, p)] += 1

            total += labels.size(0)

    # Print accuracy
    print("\nAccuracy:")
    for i, k in enumerate(topk):
        acc = topk_correct[i] / total
        print(f"Top-{k}: {acc:.4f}")

    # Confusion analysis
    most_confused = confusion_counter.most_common(10)

    print("\nTop 10 most confused class pairs (true -> predicted):")
    for (t, p), count in most_confused:
        print(f"{idx_to_class[t]} -> {idx_to_class[p]} : {count}")

    if most_confused:
        # Bar plot for top 10 most confused
        labels_plot = [
            f"{idx_to_class[t]}->{idx_to_class[p]}"
            for (t, p), _ in most_confused
        ]
        counts = [c for _, c in most_confused]

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(counts)), counts)
        plt.xticks(range(len(counts)), labels_plot, rotation=45)
        plt.ylabel("Count")
        plt.title("Top 10 Most Confused Class Pairs")
        plt.tight_layout()

        plot_path = plots_dir / f"{model_name}_most_confused_pairs.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"\nConfusion plot saved to: {plot_path}")

        # Automatic Top-10 Confused Image Grid

        fig, axes = plt.subplots(5, 4, figsize=(18, 20))
        axes = axes.reshape(5, 4)

        for idx, ((t, p), _) in enumerate(most_confused):
            row = idx // 2
            col = (idx % 2) * 2

            true_name = idx_to_class[t]
            pred_name = idx_to_class[p]

            # Get filepaths for true and predicted classes
            t_imgs = [img_path for img_path, label in test_dataset.data_path if label == t]
            p_imgs = [img_path for img_path, label in test_dataset.data_path if label == p]

            # Sample up to 2 images per class safely
            t_sample = random.sample(t_imgs, min(2, len(t_imgs)))
            p_sample = random.sample(p_imgs, min(2, len(p_imgs)))

            # Fill 2 columns (true vs predicted)
            for i in range(2):
                if i < len(t_sample):
                    img_path = Path(DATA_CFG["root"]) / t_sample[i]
                    axes[row, col].imshow(Image.open(img_path).convert("RGB"))
                    axes[row, col].set_title(f"True: {true_name}", fontsize=9)
                    axes[row, col].axis("off")

                if i < len(p_sample):
                    img_path = Path(DATA_CFG["root"]) / p_sample[i]
                    axes[row, col + 1].imshow(Image.open(img_path).convert("RGB"))
                    axes[row, col + 1].set_title(f"Pred: {pred_name}", fontsize=9)
                    axes[row, col + 1].axis("off")

        plt.tight_layout()
        sample_img_path = plots_dir / f"{model_name}_most_confused_pairs_samples.png"
        plt.savefig(sample_img_path)
        plt.close()
        print(f"\nSample images of confused pairs saved to: {sample_img_path}")

    # Per-class accuracy CSV
    class_accuracy = []
    for cls in per_class_total:
        acc = per_class_correct[cls] / per_class_total[cls]
        class_accuracy.append(
            (cls, idx_to_class[cls], acc, per_class_correct[cls], per_class_total[cls])
        )

    # Sort high -> low accuracy
    class_accuracy.sort(key=lambda x: x[2], reverse=True)

    csv_path = metric_dir / f"{model_name}_per_class_accuracy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "accuracy", "correct", "total"])
        for cls, name, acc, correct, total_cls in class_accuracy:
            writer.writerow([cls, name, f"{acc:.4f}", correct, total_cls])

    print(f"\nPer-class accuracy CSV saved to: {csv_path}")


if __name__ == "__main__":
    param_path = "wrn28_10_epoch_140.pth"
    inference(param_path)
    summarize_checkpoint_times(param_path)
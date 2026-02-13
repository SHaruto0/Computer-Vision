import torch
from torch.utils.data import DataLoader

import csv
import random
from tqdm import tqdm
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

from models.vgg import VGG16
from dataset import ImageNetDataset, build_transforms
from utils import set_seed, summarize_checkpoint_times, BASE_PATH, DATA_CFG

def inference(params_path, topk=(1,5)):
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
    test_dataset = ImageNetDataset(
        root=DATA_CFG["root"], 
        split="test",  
        transform=build_transforms(DATA_CFG["image_size"], train=False)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=1
    )

    idx_to_class = test_dataset.classes

    # Model
    model = VGG16(num_classes=DATA_CFG.get("num_classes", len(idx_to_class))).to(device)

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

    # Bar plot for top 10 most confused
    if most_confused:
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

        plot_path = plots_dir / "most_confused_pairs.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"\nConfusion plot saved to: {plot_path}")

    # Selected pairs for 3x4 image grid
    selected_pairs = [
        ("sidewinder", "horned_viper"),
        ("desktop_computer", "screen"),
        ("blenheim_spaniel", "welsh_springer_spaniel"),
        ("barn_spider", "wolf_spider"),
        ("potpie", "bagel"),
        ("bedlington_terrier", "miniature_poodle")
    ]
    class_name_to_idx = {name: idx for idx, name in enumerate(idx_to_class)}

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.reshape(3, 4)  # ensure shape

    for idx, (true_name, pred_name) in enumerate(selected_pairs):
        row = idx // 2
        col = (idx % 2) * 2  # 0 or 2

        t = class_name_to_idx[true_name]
        p = class_name_to_idx[pred_name]

        # Get all images for true and predicted classes
        t_imgs = [img_path for img_path, label in test_dataset.samples if label == t]
        p_imgs = [img_path for img_path, label in test_dataset.samples if label == p]

        # Randomly pick 2 images per class
        t_img1, t_img2 = random.sample(t_imgs, 2)
        p_img1, p_img2 = random.sample(p_imgs, 2)

        img_list = [
            (t_img1, true_name),
            (p_img1, pred_name),
            (t_img2, true_name),
            (p_img2, pred_name)
        ]

        for i in range(2):
            axes[row, col + i].imshow(Image.open(img_list[i][0]).convert("RGB"))
            axes[row, col + i].axis("off")
            axes[row, col + i].set_title(img_list[i][1], fontsize=10)

        # Black vertical line between pair columns
        axes[row, col + 1].spines['left'].set_color('black')
        axes[row, col + 1].spines['left'].set_linewidth(2)

    plt.tight_layout()
    sample_img_path = plots_dir / "most_confused_pairs_samples.png"
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

    csv_path = metric_dir / "per_class_accuracy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "accuracy", "correct", "total"])
        for cls, name, acc, correct, total_cls in class_accuracy:
            writer.writerow([cls, name, f"{acc:.4f}", correct, total_cls])

    print(f"\nPer-class accuracy CSV saved to: {csv_path}")


if __name__ == "__main__":
    param_path = "vgg16_epoch_100.pth"
    inference(param_path)
    summarize_checkpoint_times(param_path)
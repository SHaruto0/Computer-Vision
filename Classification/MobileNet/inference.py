import csv
import time
import random
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.registry import MODEL_REGISTRY, build_model
from dataset import ImageNetDataset, build_transforms
from utils import save_training_plots, set_seed, summarize_checkpoint_times, BASE_PATH

from configs.data import DATA_CFG

CKPT_DIR = BASE_PATH / "outputs" / "checkpoints"
PLOTS_DIR = BASE_PATH / "outputs" / "plots"
METRIC_DIR = BASE_PATH / "outputs" / "metrics"


def count_macs(model, input_size=(1, 3, 224, 224), device="cpu"):
    """Multiply-accumulate count for Conv2d and Linear layers."""
    macs = {"total": 0}
    handles = []

    def conv_hook(m, inp, out):
        kernel_macs = (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        macs["total"] += out.numel() * kernel_macs

    def linear_hook(m, inp, out):
        macs["total"] += out.numel() * m.in_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_size, device=device))
    for h in handles:
        h.remove()

    return macs["total"]


@torch.no_grad()
def benchmark(model, device, image_size=224, batch_sizes=(1, 64), warmup=10, iters=50):
    """Latency and throughput per batch size, plus peak memory on GPU."""
    model.eval()
    results = {}

    for bs in batch_sizes:
        x = torch.randn(bs, 3, image_size, image_size, device=device)

        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        results[bs] = {
            "latency_ms": elapsed / iters * 1000,
            "throughput": bs * iters / elapsed,
        }

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model(torch.randn(batch_sizes[-1], 3, image_size, image_size, device=device))
        torch.cuda.synchronize()
        results["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6

    return results


def cpu_sweep(model_names=None, thread_counts=(1, 4), image_size=None):
    """Batch-1 CPU latency at different thread counts. The roofline experiment."""
    import platform
    names = model_names or ["conv-s", "resnet50", "densenet121"]
    image_size = image_size or DATA_CFG["image_size"]
    num_classes = DATA_CFG.get("num_classes", 100)
    cpu = torch.device("cpu")

    print(f"\nCPU: {platform.processor()}")
    print(f"Cores: {torch.get_num_threads()} default\n")

    rows = []
    for name in names:
        model = build_model(name, num_classes, cpu)
        macs = count_macs(model, (1, 3, image_size, image_size), cpu)
        row = {"model": name, "macs_G": macs / 1e9}

        for nt in thread_counts:
            torch.set_num_threads(nt)
            r = benchmark(model, cpu, image_size, batch_sizes=(1,), warmup=5, iters=20)
            row[f"cpu_{nt}thread_ms"] = r[1]["latency_ms"]
            print(f"{name:<14} {nt} thread(s): {r[1]['latency_ms']:7.1f} ms")

        rows.append(row)
        del model

    torch.set_num_threads(torch.get_num_threads())

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    out = METRIC_DIR / "cpu_benchmark.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to: {out}")

    # Ratios relative to resnet50
    base = next((r for r in rows if r["model"] == "resnet50"), None)
    if base:
        print(f"\n{'model':<14}{'MACs ratio':>12}{'CPU 1t speedup':>16}")
        print("-" * 42)
        for r in rows:
            print(f"{r['model']:<14}"
                  f"{base['macs_G']/r['macs_G']:>12.1f}x"
                  f"{base['cpu_1thread_ms']/r['cpu_1thread_ms']:>15.1f}x")

    return rows


def profile_model(model_name, num_classes=None, image_size=None, csv_out=True):
    """Params, MACs, GPU and CPU timing. No checkpoint needed."""
    set_seed(42)
    num_classes = num_classes or DATA_CFG.get("num_classes", 100)
    image_size = image_size or DATA_CFG["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_name, num_classes, device)
    params = sum(p.numel() for p in model.parameters())
    macs = count_macs(model, (1, 3, image_size, image_size), device)

    print(f"\n=== {model_name} ===")
    print(f"Params: {params/1e6:.2f} M")
    print(f"MACs:   {macs/1e9:.3f} G  ({2*macs/1e9:.3f} GFLOPs)")

    gpu = benchmark(model, device, image_size) if device.type == "cuda" else None
    if gpu:
        print(f"GPU  bs=1:  {gpu[1]['latency_ms']:7.2f} ms  {gpu[1]['throughput']:8.0f} img/s")
        print(f"GPU  bs=64: {gpu[64]['latency_ms']:7.2f} ms  {gpu[64]['throughput']:8.0f} img/s")
        print(f"Peak mem (bs=64): {gpu['peak_mem_mb']:.0f} MB")

    cpu_model = build_model(model_name, num_classes, torch.device("cpu"))
    cpu = benchmark(cpu_model, torch.device("cpu"), image_size,
                    batch_sizes=(1,), warmup=3, iters=10)
    print(f"CPU  bs=1:  {cpu[1]['latency_ms']:7.2f} ms  {cpu[1]['throughput']:8.1f} img/s")

    if csv_out:
        METRIC_DIR.mkdir(parents=True, exist_ok=True)
        path = METRIC_DIR / "model_profiles.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["model", "params_M", "macs_G", "gpu_bs1_ms", "gpu_bs64_ms",
                            "gpu_bs64_imgs", "peak_mem_MB", "cpu_bs1_ms"])
            w.writerow([
                model_name, f"{params/1e6:.2f}", f"{macs/1e9:.3f}",
                f"{gpu[1]['latency_ms']:.2f}" if gpu else "",
                f"{gpu[64]['latency_ms']:.2f}" if gpu else "",
                f"{gpu[64]['throughput']:.0f}" if gpu else "",
                f"{gpu['peak_mem_mb']:.0f}" if gpu else "",
                f"{cpu[1]['latency_ms']:.2f}",
            ])
        print(f"Appended to {path}")

    result = {"model": model_name, "params": params, "macs": macs, "gpu": gpu, "cpu": cpu}

    del model, cpu_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def profile_all(model_names=None):
    names = model_names or ["conv-s", "resnet50", "densenet121"]
    return [profile_model(n) for n in names]


def inference(model_name, ckpt_name=None, ckpt_dir=None, topk=(1, 5)):
    """
    Evaluate a trained checkpoint on the test split.

    Args:
        model_name (str): key in MODEL_REGISTRY
        ckpt_name (str): filename, defaults to "<model_name>_best.pth"
        ckpt_dir (str|Path): where checkpoints live, defaults to outputs/checkpoints
        topk (tuple): which top-k accuracies to report
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt_dir = Path(ckpt_dir) if ckpt_dir else CKPT_DIR
    ckpt_name = ckpt_name or f"{model_name}_best.pth"
    ckpt_path = ckpt_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    plots_dir = PLOTS_DIR / model_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    # Data
    test_dataset = ImageNetDataset(
        root=DATA_CFG["root"],
        split="test",
        transform=build_transforms(DATA_CFG["image_size"], train=False),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=DATA_CFG["batch_size"],
        shuffle=False,
        num_workers=DATA_CFG["num_workers"],
        pin_memory=True,
    )
    idx_to_class = test_dataset.classes          # list: label index -> class name

    # Model
    model = build_model(model_name, DATA_CFG.get("num_classes", 100), device)
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "classes" in checkpoint:
        assert checkpoint["classes"] == test_dataset.classes, (
            "Class list mismatch between checkpoint and dataset. "
            "Accuracy would be meaningless.")
    if checkpoint.get("model_name", model_name) != model_name:
        raise ValueError(
            f"Checkpoint is for {checkpoint['model_name']}, not {model_name}")

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Metrics tracking
    total = 0
    topk_correct = [0] * len(topk)
    confusion_counter = Counter()      # (true, pred)
    per_class_total = Counter()        # true
    per_class_correct = Counter()      # true & correct

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"[Inference] {model_name}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)

            maxk = max(topk)
            topk_preds = logits.topk(maxk, dim=1).indices    # softmax is unnecessary
            hit = topk_preds == labels.unsqueeze(1)
            for i, k in enumerate(topk):
                topk_correct[i] += hit[:, :k].any(dim=1).sum().item()

            preds = topk_preds[:, 0]
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                per_class_total[t] += 1
                if t == p:
                    per_class_correct[t] += 1
                else:
                    confusion_counter[(t, p)] += 1

            total += labels.size(0)

    accs = {k: topk_correct[i] / total for i, k in enumerate(topk)}
    print(f"\n{model_name} accuracy:")
    for k in topk:
        print(f"Top-{k}: {accs[k]:.4f}")

    # Confusion analysis
    most_confused = confusion_counter.most_common(10)
    print("\nTop 10 most confused class pairs (true -> predicted):")
    for (t, p), count in most_confused:
        print(f"{idx_to_class[t]} -> {idx_to_class[p]} : {count}")

    if most_confused:
        labels_plot = [f"{idx_to_class[t]}->{idx_to_class[p]}" for (t, p), _ in most_confused]
        counts = [c for _, c in most_confused]

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(counts)), counts)
        plt.xticks(range(len(counts)), labels_plot, rotation=45, ha="right")
        plt.ylabel("Count")
        plt.title(f"Top 10 Most Confused Class Pairs - {model_name}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{model_name}_most_confused_pairs.png", dpi=150)
        plt.close()

        # One row per pair: [true sample | predicted sample]
        by_class = {}
        for img_path, label in test_dataset.samples:
            by_class.setdefault(label, []).append(img_path)

        n = len(most_confused)
        fig, axes = plt.subplots(n, 2, figsize=(7, 3.2 * n))
        axes = axes.reshape(n, 2)
        for idx, ((t, p), count) in enumerate(most_confused):
            for col, cls in enumerate((t, p)):
                ax = axes[idx, col]
                pool = by_class.get(cls, [])
                if pool:
                    ax.imshow(Image.open(random.choice(pool)).convert("RGB"))
                tag = "True" if col == 0 else "Pred"
                ax.set_title(f"{tag}: {idx_to_class[cls]} ({count})", fontsize=9)
                ax.axis("off")

        plt.tight_layout()
        plt.savefig(plots_dir / f"{model_name}_most_confused_pairs_samples.png", dpi=150)
        plt.close()
        print(f"\nConfusion plots saved to: {plots_dir}")

    # Per-class accuracy CSV
    class_accuracy = [
        (cls, idx_to_class[cls], per_class_correct[cls] / per_class_total[cls],
         per_class_correct[cls], per_class_total[cls])
        for cls in per_class_total
    ]
    class_accuracy.sort(key=lambda x: x[2], reverse=True)

    csv_path = METRIC_DIR / f"{model_name}_per_class_accuracy.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "accuracy", "correct", "total"])
        w.writerows([[c, n_, f"{a:.4f}", ok, tot] for c, n_, a, ok, tot in class_accuracy])
    print(f"Per-class accuracy CSV saved to: {csv_path}")

    # Training curves from the checkpoint histories
    save_training_plots(
        model_name=model_name,
        loss_history=checkpoint.get("loss_history", []),
        train_acc_history=checkpoint.get("train_acc_history", []),
        test_acc_history=checkpoint.get("test_acc_history", []),
        epoch_times=checkpoint.get("epoch_times", []),
        output_dir=plots_dir,
    )

    return {
        "model": model_name,
        "top1": accs.get(1),
        "top5": accs.get(5),
        "epochs": checkpoint.get("epoch"),
        "epoch_times": checkpoint.get("epoch_times", []),
    }


def plot_comparison(rows, out_dir=None):
    """Cross-model figures: accuracy vs cost, and efficiency bars."""
    out_dir = Path(out_dir) if out_dir else PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = [r for r in rows if r.get("top1") is not None]
    if not valid:
        print("No evaluated models, skipping comparison plots.")
        return

    # Accuracy vs params, accuracy vs MACs
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, xlabel in [(axes[0], "params_M", "Parameters (M)"),
                            (axes[1], "macs_G", "MACs (G)")]:
        for r in valid:
            ax.scatter(r[key], r["top1"] * 100, s=90)
            ax.annotate(r["model"], (r[key], r["top1"] * 100),
                        textcoords="offset points", xytext=(6, 5), fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Accuracy vs model size")
    axes[1].set_title("Accuracy vs compute")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_accuracy_vs_cost.png", dpi=150)
    plt.close()

    # Efficiency bars
    names = [r["model"] for r in valid]
    metrics = [
        ("params_M", "Params (M)", False),
        ("macs_G", "MACs (G)", False),
        ("cpu_bs1_ms", "CPU latency bs=1 (ms)", False),
        ("gpu_bs64_imgs", "GPU throughput bs=64 (img/s)", True),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5))
    for ax, (key, title, higher_better) in zip(axes, metrics):
        vals = [r[key] if r.get(key) is not None else 0 for r in valid]
        ax.bar(names, vals)
        ax.set_title(title + ("\n(higher better)" if higher_better else "\n(lower better)"),
                     fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_efficiency.png", dpi=150)
    plt.close()
    print(f"Comparison plots saved to: {out_dir}")


def compare(model_names, ckpt_dir=None):
    """Profile and evaluate several models, print one summary table."""
    rows = []
    for name in model_names:
        prof = profile_model(name)
        try:
            ev = inference(name, ckpt_dir=ckpt_dir)
        except FileNotFoundError as e:
            print(f"Skipping eval for {name}: {e}")
            ev = {}

        times = ev.get("epoch_times", [])
        rows.append({
            "model": name,
            "params_M": prof["params"] / 1e6,
            "macs_G": prof["macs"] / 1e9,
            "top1": ev.get("top1"),
            "top5": ev.get("top5"),
            "gpu_bs64_imgs": prof["gpu"][64]["throughput"] if prof["gpu"] else None,
            "cpu_bs1_ms": prof["cpu"][1]["latency_ms"],
            "avg_epoch_s": sum(times) / len(times) if times else None,
        })

    hdr = (f"{'model':<14}{'params(M)':>10}{'MACs(G)':>9}{'top1':>8}{'top5':>8}"
           f"{'GPU img/s':>11}{'CPU ms':>9}{'epoch(s)':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        def f(v, spec):
            return format(v, spec) if v is not None else "-".rjust(int(spec.strip('>').split('.')[0]))
        print(f"{r['model']:<14}{r['params_M']:>10.2f}{r['macs_G']:>9.3f}"
              f"{f(r['top1'], '>8.4f')}{f(r['top5'], '>8.4f')}"
              f"{f(r['gpu_bs64_imgs'], '>11.0f')}{f(r['cpu_bs1_ms'], '>9.1f')}"
              f"{f(r['avg_epoch_s'], '>10.0f')}")

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    out = METRIC_DIR / "model_comparison.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nComparison saved to: {out}")

    plot_comparison(rows)
    return rows


if __name__ == "__main__":
    cpu_sweep()
    
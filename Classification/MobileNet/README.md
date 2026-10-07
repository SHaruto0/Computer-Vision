# MobileNetV4-Conv-S Implemented from Scratch

This project implements and trains **MobileNetV4-Conv-S** from scratch in PyTorch for image classification on ImageNet.

The architecture follows the MobileNetV4 design, whose central contribution is the **Universal Inverted Bottleneck (UIB)** block: a single block with two optional depthwise convolutions that subsumes the Inverted Bottleneck, ConvNext, and FFN designs, and introduces a fourth variant (ExtraDW). This lets architecture search choose the block type per layer instead of committing to one design network-wide.

In addition, we include a comparative study with **ResNet-50** and **DenseNet-121** to analyze the trade-off between **parameter count, FLOPs, and realized latency**, benchmarking across GPU and CPU at multiple batch sizes and thread counts.

---

## Hardware

- GPU: NVIDIA T4 ×2 (Kaggle Notebook)
- CPU: 13th Gen Intel Core i7-1355U, 10 cores @ 1.70 GHz (local benchmarking)

## Dataset

ImageNet

https://www.kaggle.com/datasets/dimensi0n/imagenet-256

Download and preprocess the data on your computer by running `dataset.py` or set DOWNLOAD flag to True in kaggle notebook and download the data.

- ImageNet-style folder structure:

```
data/imagenet/
├── train/
│   ├── abacus/
│   ├── abaya/
│   └── ...
└── test/
    ├── abacus/
    ├── abaya/
    └── ...
```

- Image size: 224×224
- Normalization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
- Train augmentation: RandomResizedCrop(224), RandomHorizontalFlip
- Eval transform: Resize(256), CenterCrop(224)
- Batch size: 64, 4 workers
- Classes: 100 (to reduce computation)
- Split: 80/20 train/test

> **Note:** the test split is a random holdout from the ImageNet training pool, not the official validation set. Accuracies are not directly comparable to published ImageNet numbers.

## Model

### MobileNetV4-Conv-S Architecture

```
224×224×3
  Conv 3×3 s2                 -> 112×112×32   stem
  FusedIB exp32  s2           -> 56×56×32
  FusedIB exp96  s2           -> 28×28×64
  6 × UIB (first s2)          -> 14×14×96
  6 × UIB (first s2)          -> 7×7×128
  Conv 1×1                    -> 7×7×960
  Global AvgPool              -> 1×1×960
  Conv 1×1 -> 1280 -> classes
```

### Block Types

**UIB (Universal Inverted Bottleneck)** has two optional depthwise slots around the expansion:

```
[DW K1]      optional, before expand
PW expand
[DW K2]      optional, after expand
PW project
+ skip if stride 1 and in == out
```

Which slots are filled determines the block identity:

| K1  | K2  | Block            |
| --- | --- | ---------------- |
| off | on  | IB (MobileNetV2) |
| on  | off | ConvNext         |
| on  | on  | ExtraDW          |
| off | off | FFN              |

**FusedIB** merges the expansion pointwise and the depthwise into one regular 3×3 convolution. Used only in the first two stages, where resolution is high and channel count is low, making a dense convolution more hardware-efficient than a depthwise one.

### Design Constraints

Following the paper, the implementation excludes Squeeze-and-Excite, GELU, and LayerNorm, all of which are poorly supported on mobile accelerators. ReLU is applied only in the expanded part of each block; the start depthwise and the projection are linear.

### Model Sizes

| Model              | Params  | MACs    |
| ------------------ | ------- | ------- |
| MobileNetV4-Conv-S | 2.62 M  | 0.185 G |
| DenseNet-121       | 7.06 M  | 2.833 G |
| ResNet-50          | 23.71 M | 4.087 G |

Parameter counts are for 100 classes. At the paper's 1000 classes, Conv-S is 3.77 M, matching the published 3.8 M.

## Training

Each architecture uses the recipe appropriate to its family. Data, split, seed, and epoch count are identical across all three.

|                 | MobileNetV4-Conv-S      | ResNet-50 / DenseNet-121      |
| --------------- | ----------------------- | ----------------------------- |
| Optimizer       | AdamW                   | SGD + Nesterov (momentum 0.9) |
| Learning rate   | 4e-3                    | 0.1                           |
| Weight decay    | 0.01                    | 1e-4                          |
| Label smoothing | 0.1                     | 0.0                           |
| Scheduler       | 5-epoch warmup → cosine | 5-epoch warmup → cosine       |
| Epochs          | 100                     | 100                           |
| Precision       | AMP (fp16)              | AMP (fp16)                    |

- Loss: CrossEntropy
- Supports checkpoint resume (`start_from` in the model config)
- Saves `<model>_last.pth` and `<model>_best.pth`

## Results

### MobileNetV4-Conv-S

**Accuracy:**

- Top-1: 79.45%
- Top-5: 93.14%

**Top 10 most confused class pairs (true → predicted):**

- castle → palace : 11
- conch → hermit_crab : 11
- palace → castle : 11
- rhinoceros_beetle → leaf_beetle : 10
- toy_poodle → teddy : 10
- tricycle → jinrikisha : 10
- crayfish → hermit_crab : 9
- schipperke → groenendael : 9
- apron → backpack : 8
- english_springer → saint_bernard : 8

**Training time:**

- Average epoch time: 0h 1m 20s

### ResNet-50

**Accuracy:**

- Top-1: 83.31%
- Top-5: 94.67%

**Top 10 most confused class pairs (true → predicted):**

- castle → palace : 13
- groenendael → schipperke : 13
- night_snake → king_snake : 11
- palace → castle : 11
- apron → backpack : 10
- crayfish → hermit_crab : 8
- king_snake → night_snake : 8
- soap_dispenser → perfume : 8
- coffeepot → soap_dispenser : 7
- conch → hermit_crab : 7

**Training time:**

- Average epoch time: 0h 2m 58s

### DenseNet-121

**Accuracy:**

- Top-1: 83.83%
- Top-5: 94.83%

**Top 10 most confused class pairs (true → predicted):**

- palace → castle : 12
- night_snake → king_snake : 11
- groenendael → schipperke : 10
- king_snake → night_snake : 10
- castle → palace : 9
- library → palace : 9
- tricycle → jinrikisha : 9
- hermit_crab → conch : 8
- schipperke → groenendael : 8
- water_bottle → perfume : 8

**Training time:**

- Average epoch time: 0h 3m 55s

### Summary

| Model              | Params (M) | MACs (G) | Top-1  | Top-5  | Epoch time |
| ------------------ | ---------- | -------- | ------ | ------ | ---------- |
| MobileNetV4-Conv-S | 2.62       | 0.185    | 79.45% | 93.14% | 1m 20s     |
| DenseNet-121       | 7.06       | 2.833    | 83.83% | 94.83% | 3m 55s     |
| ResNet-50          | 23.71      | 4.087    | 83.31% | 94.67% | 2m 58s     |

## Latency Benchmarks

### GPU (NVIDIA T4)

| Model              | bs=1 latency | bs=1 throughput | bs=64 latency | bs=64 throughput | Peak mem (bs=64) |
| ------------------ | ------------ | --------------- | ------------- | ---------------- | ---------------- |
| MobileNetV4-Conv-S | 4.70 ms      | 213 img/s       | 18.64 ms      | 3434 img/s       | 293 MB           |
| ResNet-50          | 6.05 ms      | 165 img/s       | 175.13 ms     | 365 img/s        | 850 MB           |
| DenseNet-121       | 14.92 ms     | 67 img/s        | 181.16 ms     | 353 img/s        | 629 MB           |

### CPU, batch size 1 (13th Gen Intel Core i7-1355U, 10 cores)

| Model              | 1 thread | 4 threads |
| ------------------ | -------- | --------- |
| MobileNetV4-Conv-S | 14.8 ms  | 23.3 ms   |
| DenseNet-121       | 162.9 ms | 118.0 ms  |
| ResNet-50          | 198.2 ms | 129.5 ms  |

### Speedup relative to ResNet-50

| Model              | MACs ratio | GPU bs=64 | GPU bs=1 | CPU 1 thread |
| ------------------ | ---------- | --------- | -------- | ------------ |
| MobileNetV4-Conv-S | 22.1×      | 9.4×      | 1.3×     | 13.4×        |
| DenseNet-121       | 1.4×       | 0.97×     | 0.41×    | 1.2×         |

## Inference & Outputs

Each model reports:

- Top-1 accuracy
- Top-5 accuracy
- Top 10 most confused class pairs
- Confusion bar plot
- Sample images of confused pairs
- Per-class accuracy CSV
- Loss plot
- Accuracy plot
- Epoch time plot
- Params, MACs, GPU and CPU latency

### Saved files:

```
outputs/
├── plots/
│   ├── comparison_accuracy_vs_cost.png
│   ├── comparison_efficiency.png
│   ├── conv-s/
│   │   ├── conv-s_loss.png
│   │   ├── conv-s_accuracy.png
│   │   ├── conv-s_epoch_time.png
│   │   ├── conv-s_most_confused_pairs.png
│   │   └── conv-s_most_confused_pairs_samples.png
│   ├── resnet50/
│   │   ├── resnet50_loss.png
│   │   ├── resnet50_accuracy.png
│   │   ├── resnet50_epoch_time.png
│   │   ├── resnet50_most_confused_pairs.png
│   │   └── resnet50_most_confused_pairs_samples.png
│   └── densenet121/
│       ├── densenet121_loss.png
│       ├── densenet121_accuracy.png
│       ├── densenet121_epoch_time.png
│       ├── densenet121_most_confused_pairs.png
│       └── densenet121_most_confused_pairs_samples.png
└── metrics/
    ├── model_comparison.csv
    ├── model_profiles.csv
    ├── cpu_benchmark.csv
    ├── conv-s_per_class_accuracy.csv
    ├── resnet50_per_class_accuracy.csv
    └── densenet121_per_class_accuracy.csv
```

### Plots

#### Cross-Model Comparison

<div align="center">

| Accuracy vs Cost                                                   | Efficiency                                             |
| ------------------------------------------------------------------ | ------------------------------------------------------ |
| ![Accuracy vs Cost](outputs/plots/comparison_accuracy_vs_cost.png) | ![Efficiency](outputs/plots/comparison_efficiency.png) |

</div>

#### MobileNetV4-Conv-S

<div align="center">

| Loss                                                 | Accuracy                                                     | Epoch Time                                                       |
| ---------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| ![Conv-S Loss](outputs/plots/conv-s/conv-s_loss.png) | ![Conv-S Accuracy](outputs/plots/conv-s/conv-s_accuracy.png) | ![Conv-S Epoch Time](outputs/plots/conv-s/conv-s_epoch_time.png) |

<img src="outputs/plots/conv-s/conv-s_most_confused_pairs_samples.png" width="400px">

</div>

#### ResNet-50

<div align="center">

| Loss                                                       | Accuracy                                                           | Epoch Time                                                             |
| ---------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| ![ResNet50 Loss](outputs/plots/resnet50/resnet50_loss.png) | ![ResNet50 Accuracy](outputs/plots/resnet50/resnet50_accuracy.png) | ![ResNet50 Epoch Time](outputs/plots/resnet50/resnet50_epoch_time.png) |

<img src="outputs/plots/resnet50/resnet50_most_confused_pairs_samples.png" width="400px">

</div>

#### DenseNet-121

<div align="center">

| Loss                                                                | Accuracy                                                                    | Epoch Time                                                                      |
| ------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| ![DenseNet121 Loss](outputs/plots/densenet121/densenet121_loss.png) | ![DenseNet121 Accuracy](outputs/plots/densenet121/densenet121_accuracy.png) | ![DenseNet121 Epoch Time](outputs/plots/densenet121/densenet121_epoch_time.png) |

<img src="outputs/plots/densenet121/densenet121_most_confused_pairs_samples.png" width="400px">

</div>

## Analysis

- **Overall Accuracy:**
  - DenseNet-121 achieves the highest Top-1 accuracy at **83.83%**, with ResNet-50 close behind (**83.31%**) and MobileNetV4-Conv-S lower (**79.45%**).
  - MobileNetV4-Conv-S gives up roughly 4 points of Top-1 while using **9× fewer parameters** than ResNet-50 and **22× fewer MACs**.
  - Top-5 accuracy is consistently strong across all models (93–95%).

- **FLOPs Do Not Predict Latency:**
  - The central finding. MobileNetV4-Conv-S has 22.1× fewer MACs than ResNet-50, but the realized speedup varies enormously with execution context: **9.4× at GPU batch 64, only 1.3× at GPU batch 1, and 13.4× on single-threaded CPU.**
  - At batch size 1 there is not enough parallel work to hide memory latency, so time is dominated by kernel launches and memory traffic rather than arithmetic. The MAC advantage nearly vanishes.
  - DenseNet-121 makes the point more sharply still: it has **1.4× fewer MACs and 3.4× fewer parameters** than ResNet-50, yet is **2.5× slower at GPU batch 1** and slowest of all three to train. Its dense concatenations keep many intermediate tensors alive, making it memory-bound despite a low FLOP count.
  - This is the effect MobileNetV4 is explicitly designed around: the paper's roofline analysis targets Pareto-optimality across hardware with very different compute-to-bandwidth ratios, rather than minimizing FLOPs alone.

- **Thread Scaling:**
  - ResNet-50 and DenseNet-121 both speed up going from 1 to 4 CPU threads (1.5× and 1.4× respectively).
  - MobileNetV4-Conv-S gets **slower** (14.8 ms → 23.3 ms). Its per-layer work is small enough that thread synchronization overhead exceeds the parallel gain.
  - Practical consequence: lightweight models should be pinned to a single thread for batch-1 inference.

- **Memory Footprint:**
  - Peak GPU memory at batch 64: Conv-S 293 MB, DenseNet-121 629 MB, ResNet-50 850 MB.
  - DenseNet-121's footprint is high relative to its parameter count, again reflecting concatenation-driven activation memory rather than weights.

- **Confusion Patterns:**
  - All three models struggle with the same visually similar classes:
    - architecture: _castle ↔ palace_, _library → palace_
    - marine life: _conch ↔ hermit_crab_, _crayfish → hermit_crab_
    - dog breeds: _schipperke ↔ groenendael_
    - snakes: _night_snake ↔ king_snake_
  - MobileNetV4-Conv-S shows additional coarse errors the larger models avoid (_toy_poodle → teddy_, _english_springer → saint_bernard_), suggesting its reduced capacity costs fine-grained discrimination specifically.

- **Training Efficiency:**
  - MobileNetV4-Conv-S: fastest (~1m 20s/epoch)
  - ResNet-50: moderate (~2m 58s/epoch)
  - DenseNet-121: slowest (~3m 55s/epoch) despite having 3.4× fewer parameters than ResNet-50

- **Key Takeaways:**
  - Parameter count and FLOP count are poor proxies for latency; memory access patterns often dominate.
  - Depthwise-separable architectures realize their efficiency advantage on compute-constrained hardware (CPU, mobile) far more than on GPUs with compute to spare.
  - Architecture evaluation should report measured latency on the target hardware, not FLOPs.

- **Potential Improvements:**
  - int8 quantization, the standard mobile deployment path, to measure accuracy retention and further CPU speedup
  - Per-layer roofline analysis sweeping the compute-to-bandwidth ridge point, reproducing the paper's hardware-independence argument analytically
  - Stronger augmentation (RandAugment, MixUp) and longer training; the paper trains Conv-S for 9600 epochs
  - MobileNetV4-Hybrid variants, adding Mobile MQA attention to the final stages
  - Evaluation on the official ImageNet validation set and all 1000 classes

## Limitations

- 100 classes rather than 1000, and a held-out split rather than the official validation set, so absolute accuracies are not comparable to published results.
- 100 training epochs versus the paper's 9600, so MobileNetV4-Conv-S is substantially undertrained relative to its reported 73.8% on full ImageNet.
- Optimizers differ by architecture (AdamW for MobileNetV4, SGD for ResNet/DenseNet). This is standard practice but means the comparison is architecture-plus-recipe, not architecture alone.
- Kaggle provided two T4 GPUs, but training and benchmarking ran on a single device (no DataParallel), so all GPU figures are single-T4.

## Running

### Local Computer

1. Prepare dataset:

```bash
python dataset.py
```

2. Train model:

```bash
python train.py conv-s
python train.py resnet50
python train.py densenet121
```

3. Run inference and benchmarks:

```bash
python inference.py
```

- Make sure checkpoints are in `outputs/checkpoints/` (e.g., `conv-s_best.pth`).
- `python inference.py profile` runs params/MACs/latency only, no checkpoints needed.

### Kaggle Notebook

- Or run the provided notebook in Kaggle or Colab (update paths if needed).
- Set `DOWNLOAD = True` on the first run to fetch and split the dataset.
- When inferencing, upload the model checkpoints and set `CKPT_DIR` to their path.

## Reference

Qin et al., _MobileNetV4: Universal Models for the Mobile Ecosystem_, ECCV 2024. https://arxiv.org/abs/2404.10518

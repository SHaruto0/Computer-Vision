# DCGAN: Deep Convolutional Generative Adversarial Network

This project implements a **DCGAN (Deep Convolutional Generative Adversarial Network)** from scratch in PyTorch, trained on the CelebA dataset to generate synthetic human faces. The implementation explores the delicate balance of adversarial training and documents the phenomenon of **catastrophic model collapse**.

---

## Hardware & Environment

- GPU: P100 (Kaggle Notebook)

### Critical Dependencies
Due to specific environment requirements and CUDA compatibility, the following installation command is required to ensure stability:

```bash
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Dataset

CelebA (Celebrities Attributes)

https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

```
data/celeba/img_align_celeba
├── 000001.jpg
├── 000002.jpg
├── 000003.jpg
└── ...
```

- Image Size: 64×64 (Downsampled from original)
- Normalization: Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5] (Scaled to [-1, 1])
- Batch Size: 128
- Total Images: ~202,599

## Model Architecture

The architecture follows the original DCGAN paper guidelines to ensure stable feature learning.

### Generator (G)
- Input: Latent vector $z \in \mathbb{R}^{100}$ (Gaussian noise).
- Structure: Series of four fractionally-strided convolutions (Transposed Convolutions).
- Activations: ReLU for all hidden layers, Tanh for the output layer.
- Normalization: Batch Normalization after every conv layer (except output).

### Discriminator (D)
- Input: 64x64x3 image (Real or Fake).
- Structure: Series of four strided convolutions (downsampling).
- Activations: LeakyReLU (0.2) for all layers, Sigmoid for final classification.
- Normalization: Batch Normalization to stabilize gradients.

## Training Configurations

The model was refined through two iterations to address stability issues.

### Iteration 1: The Baseline
- LR: G=0.0002, D=0.0002
- Labels: Hard (1.0 Real / 0.0 Fake)
- Outcome: Catastrophic Model Collapse at Epoch 62.

### Iteration 2: Stabilized Configuration
- LR: G=0.0002, D=0.0001 (Reduced D to prevent it from winning the arms race)
- Label Smoothing: `REAL_LABEL_SMOOTH = 0.9` (Prevents Discriminator over-confidence)

## Results & Analysis

### Run 1

explain what happened. include loss, training time, and progression fig.

### Run 2

explain what happened. include loss, training time, and progression fig.

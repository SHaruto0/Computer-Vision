# VGG16 Implemented from Scratch

This project implements and trains a VGG16-style CNN in PyTorch for image classification on an ImageNet-style dataset.

---

## Hardware

- GPU: P100 (Kaggle Notebook)

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
- Batch size: 64, 4 workers
- Classes: 100 (to reduce computation)

## Model

- VGG16 with BatchNorm
- 5 convolutional blocks with MaxPooling
- Fully connected layers: 4096 → 4096 → num_classes
- ReLU + Dropout + BatchNorm

## Training

- Optimizer: SGD with momentum
- Scheduler: StepLR (decay every 30 epochs)
- Loss: CrossEntropy
- Epochs: 100
- Can resume from checkpoint (`start_from` parameter)

## Inference & Results

- Reports **Top-1** and **Top-5** accuracy
- Shows **top 10 most confused class pairs**
- Saves **loss plot**: `outputs/plots/loss.png`
- Saves **accuracy plot**: `outputs/plots/accuracy.png`
- Saves **epoch time plot**: `outputs/plots/epoch_time.png`
- Saves **confusion plot**: `outputs/plots/most_confused_pairs.png`
- Saves **sample images of most confused pairs**: `outputs/plots/most_confused_pairs_samples.png`
- Saves **per-class accuracy CSV**: `outputs/metrics/per_class_accuracy.csv`

**Example Output:**

Top-1: 77.55%

Top-5: 93.58%

Top 10 most confused class pairs (true -> predicted):

- maillot -> tank_suit : 35
- tank_suit -> maillot : 26
- sidewinder -> horned_viper : 25
- horned_viper -> sidewinder : 21
- desktop_computer -> screen : 19
- blenheim_spaniel -> welsh_springer_spaniel : 18
- maillot -> bikini : 16
- barn_spider -> wolf_spider : 15
- potpie -> bagel : 12
- bedlington_terrier -> miniature_poodle : 11

**Training time:**

- Average epoch time: 0h 6m 49s
- Total training time: 11h 29m 33s

**Plots:**

<div align="center">

| Loss                                 | Accuracy                                     | Epoch Time                                       |
| ------------------------------------ | -------------------------------------------- | ------------------------------------------------ |
| ![Loss Plot](outputs/plots/loss.png) | ![Accuracy Plot](outputs/plots/accuracy.png) | ![Epoch Time Plot](outputs/plots/epoch_time.png) |

<br>

Sample images of most confused pairs (left two images form one pair, right two images form another pair):<br>
<img src="outputs/plots/most_confused_pairs_samples.png" width="400px">

</div>

## Running

Prepare dataset:

```bash
python dataset.py
```

### Local Computer

1. Train model:

```bash
python train.py
```

2. Run inference:

```bash
python inference.py
```

### Kaggle Notebook

- Or run the provided notebook in Kaggle or Colab (update paths if needed).
- Create a dataset called `imagenet` and upload the processed train/test data created from `dataset.py` or in kaggle notebook.
- Create a dataset named `VGG16 Config` with the two config yaml files. Or simply upload it if on Colab.
- When inferencing, initialize a model named `VGG16` and upload the corresponding model checkpoint parameters.

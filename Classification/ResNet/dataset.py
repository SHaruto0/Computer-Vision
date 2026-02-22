import os
import random
import shutil
import kagglehub
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import torchvision.transforms as T
from torch.utils.data import Dataset

from utils import BASE_PATH, DATA_CFG

def download_data(data_dir):
    data_dir = Path(BASE_PATH) / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    download_path = kagglehub.dataset_download("gpiosenka/sports-classification")

    print("Path to dataset files:", download_path)

    print(f"Moving data into {data_dir} ...")
    for content_path in Path(download_path).iterdir():
        shutil.move(content_path, data_dir)
    print("Moving complete!")
    return data_dir

def build_transforms(image_size=224, train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=DATA_CFG["mean"], std=DATA_CFG["std"])
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=DATA_CFG["mean"], std=DATA_CFG["std"])
        ])

class SportsDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = BASE_PATH / Path(root)
        self.transform = transform

        self.df = pd.read_csv(self.root / "sports.csv")
        self.samples = self.df[self.df["data set"] == split]
        self.samples = self.samples[self.samples["filepaths"].str.endswith(".jpg")].reset_index(drop=True)

        self.classes = self.samples["labels"].unique().tolist()

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        row = self.samples.iloc[index]
        img_path = self.root / row["filepaths"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["class id"])
        return image, label

if __name__ == "__main__":
    download_data(DATA_CFG['root'])
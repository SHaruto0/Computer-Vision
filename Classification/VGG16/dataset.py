import os
import random
import shutil
import kagglehub
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import torchvision.transforms as T
from torch.utils.data import Dataset

from utils import BASE_PATH, DATA_CFG

def download_data(data_dir):
    data_dir = Path(BASE_PATH) / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    download_path = kagglehub.dataset_download("dimensi0n/imagenet-256")

    print("Path to dataset files:", download_path)

    print(f"Moving data into {data_dir} ...")
    shutil.move(download_path, data_dir)
    print("Moving complete!")
    return data_dir

def process_data(download):
    if download:
        data_path = download_data(DATA_CFG['root'])
    else:
        data_path = Path(BASE_PATH) / DATA_CFG['root']

    data_path = Path(data_path)
    train_path = Path(data_path) / "train"
    test_path = Path(data_path) / "test"
    print(f"{train_path}")
    print(f"{test_path}")

    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    for class_dir in tqdm(list(data_path.iterdir())):
        print()
        if class_dir == train_path or class_dir == test_path: continue
        if class_dir.is_dir():
            img_paths = []
            class_name = class_dir.name
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    img_paths.append(img_path)

            random.shuffle(img_paths)

            split_idx = int(len(img_paths) * DATA_CFG["split_ratio"])
            train_imgs = img_paths[:split_idx]
            test_imgs   = img_paths[split_idx:]

            (train_path / class_name).mkdir(parents=True, exist_ok=True)
            (test_path / class_name).mkdir(parents=True, exist_ok=True)

            for img in train_imgs:
                shutil.copy(img, train_path / class_name / img.name)

            for img in test_imgs:
                shutil.copy(img, test_path / class_name / img.name)

            shutil.rmtree(class_dir)
            print(f"{class_name}: {len(train_imgs)} train, {len(test_imgs)} val")

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

class ImageNetDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = BASE_PATH / Path(root) / split
        self.transform = transform

        # Scan class folders
        self.classes = sorted([d.name for i, d in enumerate(self.root.iterdir()) if d.is_dir() and i % (1000 // DATA_CFG.get("num_classes", 1000)) == 0])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Build list of (image_path, label)
        self.samples = []
        for cls_name in self.classes:
            cls_folder = self.root / cls_name
            for img_path in cls_folder.iterdir():
                if img_path.suffix.lower() == ".jpg":
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    process_data(download=True)
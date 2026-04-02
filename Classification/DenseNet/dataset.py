import shutil
import kagglehub
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.asyncio import tqdm

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from utils import BASE_PATH, DATA_CFG

def download_data(data_dir):
    data_dir = Path(BASE_PATH) / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    download_path = kagglehub.dataset_download("gpiosenka/butterfly-images40-species")

    print("Path to dataset files:", download_path)

    print(f"Moving data into {data_dir} ...")
    for content_path in Path(download_path).iterdir():
        shutil.move(content_path, data_dir)
    print("Moving complete!")
    return data_dir

def calculate_mean_std():
    transform = T.Compose([
        T.ToTensor()
    ])
    
    train_datasets = ButterflyDataset(
            root=DATA_CFG["root"], 
            split="train",
            transform=transform)
    
    loader = DataLoader(train_datasets, batch_size=64, shuffle=False)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0
    
    for images, _ in tqdm(loader):
        b, c, h, w = images.shape
        num_pixels = b * h * w
    
        mean += images.sum(dim=[0, 2, 3])
        std += (images ** 2).sum(dim=[0, 2, 3])
        total_pixels += num_pixels
    
    mean /= total_pixels
    std = torch.sqrt(std / total_pixels - mean ** 2)
    
    print("Mean:", mean)
    print("Std:", std)

def build_transforms(image_size=224, train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean=DATA_CFG["mean"], std=DATA_CFG["std"])
        ])
    else:
        return T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=DATA_CFG["mean"], std=DATA_CFG["std"])
        ])    
    
class ButterflyDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = BASE_PATH / Path(root)
        self.data_dir = self.root / split
        self.transform = transform

        self.df = pd.read_csv(self.root / "butterflies and moths.csv")

        data = self.df[self.df["data set"] == split]
        _unique_id_and_classes = data[["class id", "labels"]].drop_duplicates().reset_index(drop=True)
        self.class_to_idx = {cls_name: id for id, cls_name in _unique_id_and_classes.itertuples(index=False)}
        self.idx_to_class = {id: cls_name for cls_name, id in self.class_to_idx.items()}

        self.samples = []
        for _, row in data.iterrows():
            filename = row["filepaths"]
            label_name = row["labels"]
            label_idx = self.class_to_idx[label_name]
            filepath = self.root / filename
            self.samples.append((filepath, label_idx))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        filepath, label = self.samples[index]
        image = Image.open(filepath).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
if __name__ == "__main__":
    # download_data(DATA_CFG["root"])
    dataset = ButterflyDataset(root=DATA_CFG["root"], split="train", transform=build_transforms(DATA_CFG["image_size"], train=True))
    print("Number of samples:", len(dataset))
    print("Number of classes:", len(dataset.classes))
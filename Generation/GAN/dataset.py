import shutil
import kagglehub
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.asyncio import tqdm

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from utils import BASE_PATH
from configs.data import DATA_CFG

def download_data(data_dir):
    data_dir = Path(BASE_PATH) / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    download_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

    print("Path to dataset files:", download_path)

    print(f"Moving data into {data_dir} ...")
    for content_path in Path(download_path + "/img_align_celeba").iterdir():
        shutil.move(content_path, data_dir)
    print("Moving complete!")
    return data_dir

def calculate_mean_std():
    transform = T.Compose([
        T.ToTensor()
    ])
    
    train_datasets = CelebADataset(
            root=DATA_CFG["root"], 
            transform=transform)
    
    loader = DataLoader(train_datasets, batch_size=128, shuffle=False)
    
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

def build_transforms(image_size=64):
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=DATA_CFG["mean"], std=DATA_CFG["std"])
    ])

class CelebADataset(Dataset):
    def __init__(self, root, transform=None):
        super(CelebADataset, self).__init__()
        self.root = BASE_PATH / root / "img_align_celeba"
        self.transform = transform

        self.samples = []
        for img_path in self.root.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                self.samples.append(img_path)
        
        self.samples.sort() 
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        return image, 0

if __name__ == "__main__":
    # download_data(DATA_CFG["root"])
    dataset = CelebADataset(root=DATA_CFG["root"], transform=build_transforms(image_size=DATA_CFG["image_size"]))
    print("Number of samples:", len(dataset))
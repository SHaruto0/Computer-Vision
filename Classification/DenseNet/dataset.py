import shutil
import kagglehub
import pandas as pd
from PIL import Image
from pathlib import Path

import torchvision.transforms as T
from torch.utils.data import Dataset

from utils import BASE_PATH, DATA_CFG

def download_data(data_dir):
    data_dir = Path(BASE_PATH) / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    download_path = kagglehub.dataset_download("phucthaiv02/butterfly-image-classification")

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
    
class ButterflyDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = BASE_PATH / Path(root)
        self.data_dir = self.root / split
        self.transform = transform

        self.filename = "Training_set.csv" if split == "train" else "Testing_set.csv"
        self.df = pd.read_csv(self.root / self.filename)

        self.classes = sorted(self.df["label"].unique().tolist())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        for row in self.df.itertuples():
            self.samples.append((self.data_dir / row.filename, row.label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        filepath, label = self.samples[index]
        image = Image.open(filepath).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, self.class_to_idx[label]
    
if __name__ == "__main__":
    # download_data(DATA_CFG["root"])
    dataset = ButterflyDataset(root=DATA_CFG["root"], split="train", transform=build_transforms(DATA_CFG["image_size"], train=True))
    print("Number of samples:", len(dataset))
    print("Number of classes:", len(dataset.classes))
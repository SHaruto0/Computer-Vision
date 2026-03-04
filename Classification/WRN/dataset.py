import shutil
import kagglehub
from PIL import Image
from pathlib import Path

import torchvision.transforms as T
from torch.utils.data import Dataset

from utils import BASE_PATH, DATA_CFG

def download_data(data_dir):
    data_dir = Path(BASE_PATH) / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    download_path = kagglehub.dataset_download("melikechan/cifar100")

    print("Path to dataset files:", download_path)

    print(f"Moving data into {data_dir} ...")
    for content_path in Path(download_path + "/cifar100").iterdir():
        print(f"Moving {content_path} to {data_dir} ...")
        shutil.move(content_path, data_dir)
    print("Moving complete!")
    return data_dir

def build_transforms(image_size=32, train=True):
    if train:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=DATA_CFG["mean"], std=DATA_CFG["std"])
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=DATA_CFG["mean"], std=DATA_CFG["std"])
        ])

class CIFAR100(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = BASE_PATH / Path(root)
        self.transform = transform

        self.data_dir = self.root / split
        self.data_path = []
        self.class_to_idx = {class_dir.name: idx for idx, class_dir in enumerate(self.data_dir.iterdir()) if class_dir.is_dir()}

        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.iterdir():
                    if img_path.suffix in [".jpg", ".png"]:
                        self.data_path.append((img_path, self.class_to_idx[class_dir.name]))

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        img_path, label = self.data_path[index]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(label)
        return image, label

if __name__ == "__main__":
    download_data(DATA_CFG['root'])
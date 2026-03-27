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

    download_path = kagglehub.dataset_download("sshikamaru/fruit-recognition")

    print("Path to dataset files:", download_path)

    print(f"Moving data into {data_dir} ...")
    for content_path in Path(download_path).iterdir():
        shutil.move(content_path, data_dir)
    print("Moving complete!")
    return data_dir

def build_transforms(image_size=100, train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
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
        self.train_dir = self.root / "train/train"
        self.test_dir = self.root / "test/test"
        self.transform = transform

        self.classes = sorted([d.name for d in self.train_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        if split == "train":
            for cls_name in self.train_dir.iterdir():
                cls_folder = self.train_dir / cls_name
                for img_path in cls_folder.iterdir():
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        self.samples.append((img_path, self.class_to_idx[cls_name.name]))
        else:
            self.df = pd.read_csv(self.root / "sampleSubmission.csv")
            for _, row in self.df.iterrows():
                img_path = self.test_dir / f"{int(row['id']):04d}.jpg"
                self.samples.append((img_path, self.class_to_idx[row['label']]))

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
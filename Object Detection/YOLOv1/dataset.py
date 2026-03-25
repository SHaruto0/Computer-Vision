import shutil
import kagglehub
from pathlib import Path

from utils import BASE_PATH, DATA_CFG

def download_data(data_dir):
    data_dir = Path(BASE_PATH) / data_dir / "2007"
    data_dir.mkdir(parents=True, exist_ok=True)

    download_path = kagglehub.dataset_download("zaraks/pascal-voc-2007")

    print("Path to dataset files:", download_path)

    print(f"Moving data into {data_dir} ...")
    for content_path in Path(download_path + "/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007").iterdir():
        print(f"Moving {content_path} to {data_dir} ...")
        shutil.move(content_path, data_dir)
    print("Moving complete!")
    return data_dir

if __name__ == "__main__":
    download_data(DATA_CFG["root"])
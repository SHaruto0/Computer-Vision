import yaml
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent

def load_yaml(path):
    path = BASE_PATH / path
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

DATA_CFG = load_yaml("configs/data.yaml")
DENSENET_CFG = load_yaml("configs/densenet.yaml")
from models.mobilenet import MobileNetV4ConvS
from models.resnet import ResNet50, ResNet101, ResNet152
from models.densenet import DenseNet121, DenseNet169, DenseNet201

from configs.mobilenet import MOBILENET_CONFIG
from configs.resnet import RESNET_CONFIG
from configs.densenet import DENSENET_CONFIG

MODEL_REGISTRY = {
    "conv-s":      (lambda nc: MobileNetV4ConvS(num_classes=nc), MOBILENET_CONFIG),
    "resnet50":    (lambda nc: ResNet50(num_classes=nc),         RESNET_CONFIG),
    "resnet101":   (lambda nc: ResNet101(num_classes=nc),        RESNET_CONFIG),
    "resnet152":   (lambda nc: ResNet152(num_classes=nc),        RESNET_CONFIG),
    "densenet121": (lambda nc: DenseNet121(num_classes=nc),      DENSENET_CONFIG),
    "densenet169": (lambda nc: DenseNet169(num_classes=nc),      DENSENET_CONFIG),
    "densenet201": (lambda nc: DenseNet201(num_classes=nc),      DENSENET_CONFIG),
}


def build_model(model_name, num_classes, device="cpu"):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Options: {list(MODEL_REGISTRY)}")
    build_fn, _ = MODEL_REGISTRY[model_name]
    return build_fn(num_classes).to(device)
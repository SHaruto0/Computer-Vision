import torch
import torch.nn as nn

from .uib import UIB
from .fusedIB import FusedIB
from .convBNReLU import ConvBNReLU

class MobileNetV4ConvS(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.3):
        super(MobileNetV4ConvS, self).__init__()
        self.UIB_CFG = [
            (5, 5, 192,  96, 2),
            (0, 3, 192,  96, 1),
            (0, 3, 192,  96, 1),
            (0, 3, 192,  96, 1),
            (0, 3, 192,  96, 1),
            (3, 0, 384,  96, 1),
            (3, 3, 576, 128, 2),
            (5, 5, 512, 128, 1),
            (0, 5, 512, 128, 1),
            (0, 5, 384, 128, 1),
            (0, 3, 512, 128, 1),
            (0, 3, 512, 128, 1),
        ]

        layers = [
            ConvBNReLU(3, 32, k=3, stride=2),
            FusedIB(32, 32, 32, stride=2),
            FusedIB(32, 96, 64, stride=2),
        ]

        in_channel = 64
        for k1, k2, exp_channel, out_channel, stride in self.UIB_CFG:
            layers.append(UIB(in_channel, exp_channel, out_channel, k1, k2, stride))
            in_channel = out_channel

        layers.append(ConvBNReLU(in_channel, 960, k=1))
        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(960, 1280, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(1280, num_classes, kernel_size=1),
        )   

        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x.flatten(1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


def test():
    model = MobileNetV4ConvS(num_classes=1000)
    print(sum(p.numel() for p in model.parameters()) / 1e6)  # ~3.8
    print(model(torch.randn(2, 3, 224, 224)).shape)

if __name__ == "__main__":
    test()
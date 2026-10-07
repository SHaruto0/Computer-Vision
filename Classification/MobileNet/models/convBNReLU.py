import torch
import torch.nn as nn

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, k=3, stride=1, groups=1, activation=True):
        layers = [
            nn.Conv2d(in_channel, out_channel, k, stride, padding=k//2,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channel)
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
            
        super(ConvBNReLU, self).__init__(*layers)
import torch
import torch.nn as nn

from .convBNReLU import ConvBNReLU

class FusedIB(nn.Module):
    def __init__(self, in_channel, exp_channel, out_channel, stride):
        super(FusedIB, self).__init__()
        self.expansion = ConvBNReLU(in_channel, exp_channel, k=3, stride=stride)
        self.project = ConvBNReLU(exp_channel, out_channel, k=1, activation=False)
        self.use_skip = stride == 1 and in_channel == out_channel

    def forward(self, x):
        out = self.project(self.expansion(x))
        return out + x if self.use_skip else out

def test():
    block = FusedIB(32, 96, 64, 2)
    x = torch.randn(1, 32, 56, 56)
    print(block(x).shape)

if __name__ == "__main__":
    test()
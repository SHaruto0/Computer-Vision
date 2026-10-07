import torch
import torch.nn as nn

from .convBNReLU import ConvBNReLU

class UIB(nn.Module):
    def __init__(self, in_channel, exp_channel, out_channel, k1=0, k2=0, stride=1):
        super(UIB, self).__init__()

        # stride goes on middle DW if present, else start DW
        s1 = stride if (k1 and not k2) else 1
        s2 = stride if k2 else 1

        self.start_dw = (ConvBNReLU(in_channel, in_channel, k=k1, stride=s1,
                                    groups=in_channel, activation=False)
                         if k1 else nn.Identity())
        self.expand = ConvBNReLU(in_channel, exp_channel, k=1)
        self.middle_dw = (ConvBNReLU(exp_channel, exp_channel, k=k2, stride=s2,
                                     groups=exp_channel)
                          if k2 else nn.Identity())
        self.project = ConvBNReLU(exp_channel, out_channel, k=1, activation=False)

        self.use_skip = stride == 1 and in_channel == out_channel

    def forward(self, x):
        out = self.start_dw(x)
        out = self.expand(out)
        out = self.middle_dw(out)
        out = self.project(out)
        return out + x if self.use_skip else out
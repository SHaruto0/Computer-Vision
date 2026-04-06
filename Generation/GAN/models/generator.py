import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64):
        super(Generator, self).__init__()
        
        self.net = nn.Sequential(
            # input is (latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(inplace=True),
            # state size: (feature_maps*8, 4, 4)
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(inplace=True),
            # state size: (feature_maps*4, 8, 8)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),
            # state size: (feature_maps*2, 16, 16)
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True),
            # state size: (feature_maps, 32, 32)
            nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # output size: (img_channels, 64, 64)
        )

    def forward(self, x):
        return self.net(x)

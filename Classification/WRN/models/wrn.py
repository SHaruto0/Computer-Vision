import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, out_channels, k, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels*k, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels*k)
        self.conv2 = nn.Conv2d(out_channels*k, out_channels*k, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels*k)
        self.relu = nn.ReLU(inplace=True)

        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x
    
class WRN(nn.Module):
    def __init__(self, block, k, N, image_channels, num_classes):
        super(WRN, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Number of blocks in each layer is determined by N, where N = 6n + 4
        n = (N - 4) // 6

        # WRN Layers
        self.layer1 = self._make_layer(block, k, n, out_channels=16, stride=1)
        self.layer2 = self._make_layer(block, k, n, out_channels=32, stride=2)
        self.layer3 = self._make_layer(block, k, n, out_channels=64, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64*k, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, k, n, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * k:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*k, kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm2d(out_channels*k))
        
        layers.append(block(self.in_channels, out_channels, k, identity_downsample, stride))
        self.in_channels = out_channels * k

        for i in range(n - 1):
            layers.append(block(self.in_channels, out_channels, k))
        
        return nn.Sequential(*layers)
    
def WRN28_10(img_channels=3, num_classes=100):
    return WRN(block, 10, 28, img_channels, num_classes)
    

def test():
    net = WRN28_10()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.shape)

if __name__ == "__main__":
    test()
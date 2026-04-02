import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels=4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        identity = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        out = torch.cat((identity, x), dim=1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate=32):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.growth_rate = growth_rate

        layers = []
        for _ in range(num_layers):
            layers.append(DenseLayer(in_channels, self.growth_rate))
            in_channels += self.growth_rate

        self.dense_block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.dense_block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compression_factor=0.5):
        super(TransitionLayer, self).__init__()
        out_channels = int(in_channels * compression_factor)

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x
    
class DenseNet(nn.Module):
    def __init__(self, image_channels, layers, growth_rate=32, compression_factor=0.5, num_classes=100):
        super(DenseNet, self).__init__()
        self.features = []
        self.features.append(
            nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        )
        self.features.append(nn.BatchNorm2d(64))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        num_channels = 64
        for i, num_layers in enumerate(layers):
            dense_block = DenseBlock(num_layers, num_channels, growth_rate=growth_rate)
            self.features.append(dense_block)
            num_channels += num_layers * growth_rate

            if i != len(layers) - 1:
                transition_layer = TransitionLayer(num_channels, compression_factor=compression_factor)
                self.features.append(transition_layer)
                num_channels = int(num_channels * compression_factor)
        
        self.features.append(nn.BatchNorm2d(num_channels))
        self.features.append(nn.AdaptiveAvgPool2d((1,1)))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def DenseNet121(img_channels, num_classes):
    return DenseNet(image_channels=img_channels, layers=[6, 12, 24, 16], growth_rate=32, compression_factor=0.5, num_classes=num_classes)

def DenseNet169(img_channels, num_classes):
    return DenseNet(image_channels=img_channels, layers=[6, 12, 32, 32], growth_rate=32, compression_factor=0.5, num_classes=num_classes)

def DenseNet201(img_channels, num_classes):
    return DenseNet(image_channels=img_channels, layers=[6, 12, 48, 32], growth_rate=32, compression_factor=0.5, num_classes=num_classes)

def DenseNet264(img_channels, num_classes):
    return DenseNet(image_channels=img_channels, layers=[6, 12, 64, 48], growth_rate=32, compression_factor=0.5, num_classes=num_classes)

def test():
    model = DenseNet121(img_channels=3, num_classes=100)
    x = torch.randn(2, 3, 100, 100)
    out = model(x)
    print(out.shape)

if __name__ == "__main__":
    test()
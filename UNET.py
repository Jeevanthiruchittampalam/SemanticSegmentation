import torch
import torch.nn as nn

class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(TwoConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.up(x))
        return x

class UNet_2D(nn.Module):
    def __init__(self, num_classes=24):  # Set default to 24 classes
        super(UNet_2D, self).__init__()
        self.TCB1 = TwoConvBlock(3, 64, 64)
        self.TCB2 = TwoConvBlock(64, 128, 128)
        self.TCB3 = TwoConvBlock(128, 256, 256)
        self.TCB4 = TwoConvBlock(256, 512, 512)
        self.TCB5 = TwoConvBlock(512, 1024, 1024)
        self.TCB6 = TwoConvBlock(1024, 512, 512)
        self.TCB7 = TwoConvBlock(512, 256, 256)
        self.TCB8 = TwoConvBlock(256, 128, 128)
        self.TCB9 = TwoConvBlock(128, 64, 64)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        
        self.UC1 = UpConv(1024, 512) 
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4 = UpConv(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # Updated for 24 classes

    def forward(self, x):
        x1 = self.TCB1(x)
        x = self.maxpool(x1)

        x2 = self.TCB2(x)
        x = self.maxpool(x2)

        x3 = self.TCB3(x)
        x = self.maxpool(x3)

        x4 = self.TCB4(x)
        x = self.maxpool(x4)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim=1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.TCB9(x)

        x = self.final_conv(x)
        return x

"""Small ResNet for 28x28 images."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False), nn.ReLU(),
            nn.Linear(ch // reduction, ch, bias=False), nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        return x * self.fc(w).view(b, c, 1, 1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True, dropout=0.15):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + self.shortcut(x))

class ResNetSmall(nn.Module):
    def __init__(self, channels=(64, 128, 256), blocks_per_stage=2, num_classes=10, use_se=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]), nn.ReLU(),
        )
        layers = []
        in_ch = channels[0]
        for i, ch in enumerate(channels):
            stride = 1 if i == 0 else 2
            for j in range(blocks_per_stage):
                s = stride if j == 0 else 1
                layers.append(ResBlock(in_ch, ch, stride=s, use_se=use_se))
                in_ch = ch
        self.stages = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

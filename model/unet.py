"""
U-Net Deep Learning Model for SAR Multimodal Flood Segmentation.
Accepts 7 input channels: [pre_VV, pre_VH, post_VV, post_VH, diff_VV, slope, HAND].
Outputs 3 classes: [0: background, 1: permanent_water, 2: flood_water].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class FloodUNet(nn.Module):
    """
    Multimodal 7-channel SAR U-Net for Himalayan Flood and Water mapping.
    Matches Kuro Siwo multi-task specifications.
    """
    def __init__(self, in_channels: int = 7, num_classes: int = 3, base_c: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c, base_c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c * 2, base_c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c * 4, base_c * 8))

        self.up1 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_c * 8, base_c * 4)

        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_c * 4, base_c * 2)

        self.up3 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_c * 2, base_c)

        self.outc = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        u1 = self.up1(x4)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv3(u3)

        logits = self.outc(u3)
        return logits

def load_flood_model(weights_path: str = None, device: str = "cpu") -> FloodUNet:
    """
    Factory function to initialize and load weights for the 7-channel Flood U-Net.
    """
    model = FloodUNet(in_channels=7, num_classes=3)
    if weights_path and torch.cuda.is_available() and device == "cuda":
        model = model.to("cuda")
    else:
        model = model.to(device)

    # If pretrained weights file exists, load it; otherwise return ready initialized model
    import os
    if weights_path and os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state)
        except Exception as e:
            print(f"Warning: could not load weights from {weights_path}: {e}")
    model.eval()
    return model

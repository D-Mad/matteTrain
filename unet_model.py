import torch
import torch.nn as nn
from unet_parts import DoubleConv, Down, Up, OutConv

class UNet9C(nn.Module):
    def __init__(self, n_channels_in=9, n_classes_out=1, bilinear=True):
        super(UNet9C, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_classes_out = n_classes_out
        self.bilinear = bilinear

        # ---- Encoder ----
        # This is the important modification: n_channels_in=9
        self.inc = DoubleConv(self.n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # ---- Decoder ----
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # ---- Output Layer ----
        self.outc = OutConv(64, self.n_classes_out)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path and skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        
        # Apply output activation
        # For alpha matting, we need values in [0, 1]
        # Using clamp instead of sigmoid based on FBA_Matting recommendation
        return torch.clamp(logits, 0, 1)

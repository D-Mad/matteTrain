import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Khối dư cơ bản: (Conv -> BN -> ReLU) * 2 + kết nối dư.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection nếu số kênh đầu vào và đầu ra khác nhau
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ResUNetDown(nn.Module):
    """
    Phần đi xuống của ResUNet: MaxPool -> ResidualBlock.
    """
    def __init__(self, in_channels, out_channels):
        super(ResUNetDown, self).__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)

class ResUNetUp(nn.Module):
    """

    Phần đi lên của ResUNet: Upsample -> Concat -> ResidualBlock.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(ResUNetUp, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        # in_channels là tổng số kênh từ skip connection và lớp dưới
        self.conv = ResidualBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 là từ lớp dưới (upsampled), x2 là từ skip connection
        x1 = self.up(x1)

        # Đảm bảo kích thước không gian khớp nhau (quan trọng cho inference)
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ResUNet(nn.Module):
    """
    Kiến trúc Residual U-Net cho Alpha Matting.
    """
    def __init__(self, n_channels_in=9, n_classes_out=1, bilinear=True):
        super(ResUNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_classes_out = n_classes_out
        self.bilinear = bilinear

        # Encoder
        self.inc = ResidualBlock(n_channels_in, 64)
        self.down1 = ResUNetDown(64, 128)
        self.down2 = ResUNetDown(128, 256)
        self.down3 = ResUNetDown(256, 512)
        self.down4 = ResUNetDown(512, 512) # Bottleneck

        # Decoder
        # Số kênh đầu vào của mỗi lớp Up là:
        # kênh từ skip connection + kênh từ lớp dưới
        self.up1 = ResUNetUp(512 + 512, 256, bilinear)
        self.up2 = ResUNetUp(256 + 256, 128, bilinear)
        self.up3 = ResUNetUp(128 + 128, 64, bilinear)
        self.up4 = ResUNetUp(64 + 64, 64, bilinear)
        
        # Output
        self.outc = OutConv(64, n_classes_out)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        
        # Lớp đầu ra tuyến tính với kẹp giá trị, theo khuyến nghị
        return torch.clamp(logits, 0, 1)

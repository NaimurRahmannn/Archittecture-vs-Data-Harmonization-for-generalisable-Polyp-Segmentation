import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DecodeBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class M1ConvNeXtUNet(nn.Module):
    def __init__(self, encoder_name="convnext_tiny", pretrained=True):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for M1ConvNeXtUNet. Install it with: pip install timm"
            ) from exc

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        channels = self.encoder.feature_info.channels()
        if len(channels) != 4:
            raise ValueError(f"Expected 4 feature stages, got {len(channels)} from {encoder_name}")

        c1, c2, c3, c4 = channels

        # Extra bottleneck downsample enables 4 decode stages with 4 skip connections.
        self.bottleneck = ConvBlock(c4, c4)

        self.dec4 = DecodeBlock(c4, c4, c3)
        self.dec3 = DecodeBlock(c3, c3, c2)
        self.dec2 = DecodeBlock(c2, c2, c1)
        self.dec1 = DecodeBlock(c1, c1, 64)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        f1, f2, f3, f4 = self.encoder(x)

        x = F.max_pool2d(f4, kernel_size=2, stride=2)
        x = self.bottleneck(x)

        x = self.dec4(x, f4)
        x = self.dec3(x, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)
        x = self.final_up(x)
        return self.out_conv(x)

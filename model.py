"""
U-Net model for binary segmentation (Oil Spill Detection).
Standard encoder-decoder architecture with skip connections.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two consecutive Conv2d -> BatchNorm -> ReLU blocks."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """Encoder path: ConvBlock + MaxPool at each level."""

    def __init__(self, in_channels, features):
        super().__init__()
        self.stages = nn.ModuleList()
        self.pools = nn.ModuleList()

        for f in features:
            self.stages.append(ConvBlock(in_channels, f))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = f

    def forward(self, x):
        skip_connections = []
        for stage, pool in zip(self.stages, self.pools):
            x = stage(x)
            skip_connections.append(x)
            x = pool(x)
        return x, skip_connections


class Decoder(nn.Module):
    """Decoder path: Upsample + concat skip + ConvBlock at each level."""

    def __init__(self, features):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.stages = nn.ModuleList()

        # Decoder goes from bottleneck back through encoder levels in reverse
        # features = [64, 128, 256, 512], bottleneck = 1024
        # Level 0: upconv 1024->512, concat with skip(512) = 1024, conv 1024->512
        # Level 1: upconv 512->256,  concat with skip(256) = 512,  conv 512->256
        # Level 2: upconv 256->128,  concat with skip(128) = 256,  conv 256->128
        # Level 3: upconv 128->64,   concat with skip(64)  = 128,  conv 128->64
        reversed_features = list(reversed(features))
        in_ch = reversed_features[0] * 2  # bottleneck channels (1024)
        for out_ch in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            # After concat with skip: out_ch (from upconv) + out_ch (from skip) = 2 * out_ch
            self.stages.append(ConvBlock(out_ch * 2, out_ch))
            in_ch = out_ch  # output of this level feeds into the next upconv

    def forward(self, x, skip_connections):
        for upconv, stage, skip in zip(self.upconvs, self.stages, reversed(skip_connections)):
            x = upconv(x)
            # Handle potential size mismatch from odd dimensions
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = stage(x)
        return x


class UNet(nn.Module):
    """
    U-Net for binary segmentation.
    
    Architecture:
        Encoder: 1 -> 64 -> 128 -> 256 -> 512
        Bottleneck: 512 -> 1024
        Decoder: 1024 -> 512 -> 256 -> 128 -> 64
        Output: 64 -> 1 (raw logits, no sigmoid — use BCEWithLogitsLoss)
    """

    def __init__(self, in_channels=1, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.encoder = Encoder(in_channels, features)
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        self.decoder = Decoder(features)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        return self.final_conv(x)


if __name__ == "__main__":
    # Verify model architecture
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 512, 512)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

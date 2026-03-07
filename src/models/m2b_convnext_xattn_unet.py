import math

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


class CrossAttentionPreserveFusion(nn.Module):
    def __init__(self, decoder_ch, encoder_ch, attn_dim=64, max_tokens=1024):
        super().__init__()
        self.attn_dim = int(attn_dim)
        self.max_tokens = int(max_tokens)
        self.scale = self.attn_dim ** -0.5

        self.q_proj = nn.Conv2d(decoder_ch, self.attn_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(encoder_ch, self.attn_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(encoder_ch, self.attn_dim, kernel_size=1, bias=False)
        self.attn_out_proj = nn.Conv2d(self.attn_dim, decoder_ch, kernel_size=1, bias=False)

        # Direct encoder path to preserve information when attention is uncertain.
        self.skip_proj = nn.Conv2d(encoder_ch, decoder_ch, kernel_size=1, bias=False)

        self.q_norm = nn.LayerNorm(self.attn_dim)
        self.k_norm = nn.LayerNorm(self.attn_dim)

        self.post_norm = nn.BatchNorm2d(decoder_ch)
        self.alpha_logit = nn.Parameter(torch.tensor(-2.0))
        self.beta_logit = nn.Parameter(torch.tensor(2.0))

    def _maybe_pool(self, x):
        _, _, h, w = x.shape
        tokens = h * w
        if tokens <= self.max_tokens:
            return x, 1
        scale = int(math.ceil(math.sqrt(tokens / float(self.max_tokens))))
        return F.avg_pool2d(x, kernel_size=scale, stride=scale), scale

    def forward(self, dec_feat, enc_feat):
        if dec_feat.shape[-2:] != enc_feat.shape[-2:]:
            enc_feat = F.interpolate(enc_feat, size=dec_feat.shape[-2:], mode="bilinear", align_corners=False)

        dec_small, scale_d = self._maybe_pool(dec_feat)
        enc_small, scale_e = self._maybe_pool(enc_feat)
        if dec_small.shape[-2:] != enc_small.shape[-2:]:
            target_hw = (
                min(dec_small.shape[-2], enc_small.shape[-2]),
                min(dec_small.shape[-1], enc_small.shape[-1]),
            )
            dec_small = F.interpolate(dec_small, size=target_hw, mode="bilinear", align_corners=False)
            enc_small = F.interpolate(enc_small, size=target_hw, mode="bilinear", align_corners=False)

        q = self.q_proj(dec_small)
        k = self.k_proj(enc_small)
        v = self.v_proj(enc_small)

        b, c, h, w = q.shape
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        attn_feat = self.attn_out_proj(out)

        if scale_d > 1 or scale_e > 1:
            attn_feat = F.interpolate(attn_feat, size=dec_feat.shape[-2:], mode="bilinear", align_corners=False)

        skip_feat = self.skip_proj(enc_feat)
        alpha = torch.sigmoid(self.alpha_logit)
        beta = torch.sigmoid(self.beta_logit)
        fused = dec_feat + alpha * attn_feat + beta * skip_feat
        return self.post_norm(fused)


class DecodeBlockXAttnPreserve(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, attn_dim=64, max_tokens=1024):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.fusion = CrossAttentionPreserveFusion(out_ch, skip_ch, attn_dim=attn_dim, max_tokens=max_tokens)
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fusion(x, skip)
        return self.conv(x)


class M2bConvNeXtXAttnUNet(nn.Module):
    def __init__(self, encoder_name="convnext_tiny", pretrained=True, attn_dim=64, max_tokens=1024):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for M2bConvNeXtXAttnUNet. Install it with: pip install timm"
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

        self.bottleneck = ConvBlock(c4, c4)
        self.dec4 = DecodeBlockXAttnPreserve(c4, c4, c3, attn_dim=attn_dim, max_tokens=max_tokens)
        self.dec3 = DecodeBlockXAttnPreserve(c3, c3, c2, attn_dim=attn_dim, max_tokens=max_tokens)
        self.dec2 = DecodeBlockXAttnPreserve(c2, c2, c1, attn_dim=attn_dim, max_tokens=max_tokens)
        self.dec1 = DecodeBlockXAttnPreserve(c1, c1, 64, attn_dim=attn_dim, max_tokens=max_tokens)

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

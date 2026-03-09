import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


def load_simple_yaml(path):
    config = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.lower() in {"true", "false"}:
                config[key] = value.lower() == "true"
                continue
            try:
                config[key] = int(value)
                continue
            except ValueError:
                pass
            try:
                config[key] = float(value)
                continue
            except ValueError:
                pass
            config[key] = value
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_manifest(manifest_csv):
    with Path(manifest_csv).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_manifest(rows, split=None, test_group=None):
    filtered = []
    for row in rows:
        if split is not None and row["split"] != split:
            continue
        if test_group is not None and row["test_group"] != test_group:
            continue
        filtered.append(row)
    return filtered


def dice_bce_loss(logits, targets, eps=1e-6):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()
    return bce + dice_loss


def compute_metrics(tp, fp, fn, eps=1e-8):
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f2 = (5.0 * precision * recall + eps) / (4.0 * precision + recall + eps)
    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "F2": float(f2),
    }


class ManifestSegDataset(Dataset):
    def __init__(
        self,
        rows,
        data_root,
        image_size=256,
        augment=False,
        rotate_limit=20.0,
        brightness_range=0.2,
        contrast_range=0.2,
        gamma_range=0.2,
    ):
        self.rows = rows
        self.data_root = Path(data_root)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.rotate_limit = float(rotate_limit)
        self.brightness_range = float(brightness_range)
        self.contrast_range = float(contrast_range)
        self.gamma_range = float(gamma_range)

    def __len__(self):
        return len(self.rows)

    def _apply_augment(self, image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        angle = random.uniform(-self.rotate_limit, self.rotate_limit)
        image = image.rotate(angle, resample=Image.BILINEAR)
        mask = mask.rotate(angle, resample=Image.NEAREST)

        if self.brightness_range > 0:
            brightness_factor = random.uniform(1.0 - self.brightness_range, 1.0 + self.brightness_range)
            image = ImageEnhance.Brightness(image).enhance(brightness_factor)

        if self.contrast_range > 0:
            contrast_factor = random.uniform(1.0 - self.contrast_range, 1.0 + self.contrast_range)
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        if self.gamma_range > 0:
            gamma = random.uniform(1.0 - self.gamma_range, 1.0 + self.gamma_range)
            image_np = np.asarray(image, dtype=np.float32) / 255.0
            image_np = np.power(np.clip(image_np, 0.0, 1.0), gamma)
            image = Image.fromarray(np.clip(image_np * 255.0, 0, 255).astype(np.uint8))

        return image, mask

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self.data_root / row["image"]
        mask_path = self.data_root / row["mask"]

        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)
        if self.augment:
            image, mask = self._apply_augment(image, mask)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        return image_tensor, mask_tensor


class DoubleConv(nn.Module):
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


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_ch=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)


def build_model_from_config(cfg):
    model_name = str(cfg.get("model_name", "m0_unet")).lower()
    if model_name in {"m0_unet", "unet"}:
        return UNet(in_channels=3, out_channels=1, base_ch=cfg.get("base_ch", 32))
    if model_name in {"m1_convnext_unet", "m1_convnext_base"}:
        from src.models.m1_convnext_unet import M1ConvNeXtUNet

        return M1ConvNeXtUNet(
            encoder_name=cfg.get("encoder_name", "convnext_tiny"),
            pretrained=cfg.get("pretrained", True),
        )
    if model_name in {"m2_convnext_xattn_unet", "m2_convnext_xattn"}:
        from src.models.m2_convnext_xattn_unet import M2ConvNeXtXAttnUNet

        return M2ConvNeXtXAttnUNet(
            encoder_name=cfg.get("encoder_name", "convnext_tiny"),
            pretrained=cfg.get("pretrained", True),
            attn_dim=cfg.get("attn_dim", 64),
            max_tokens=cfg.get("max_tokens", 1024),
        )
    if model_name in {"m2b_convnext_xattn_unet", "m2b_convnext_xattn"}:
        from src.models.m2b_convnext_xattn_unet import M2bConvNeXtXAttnUNet

        return M2bConvNeXtXAttnUNet(
            encoder_name=cfg.get("encoder_name", "convnext_tiny"),
            pretrained=cfg.get("pretrained", True),
            attn_dim=cfg.get("attn_dim", 64),
            max_tokens=cfg.get("max_tokens", 1024),
        )
    raise ValueError(f"Unsupported model_name: {cfg.get('model_name')}")


@torch.no_grad()
def evaluate_model(model, loader, device, threshold=0.5):
    model.eval()
    tp = 0.0
    fp = 0.0
    fn = 0.0
    pred_area_sum = 0.0
    gt_area_sum = 0.0
    num_images = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        tp += (preds * masks).sum().item()
        fp += (preds * (1.0 - masks)).sum().item()
        fn += ((1.0 - preds) * masks).sum().item()
        pred_area_sum += preds.mean(dim=(1, 2, 3)).sum().item()
        gt_area_sum += masks.mean(dim=(1, 2, 3)).sum().item()
        num_images += preds.size(0)

    metrics = compute_metrics(tp, fp, fn)
    mean_pred_area = pred_area_sum / max(1, num_images)
    mean_gt_area = gt_area_sum / max(1, num_images)
    metrics.update(
        {
            "mean_pred_area": float(mean_pred_area),
            "mean_gt_area": float(mean_gt_area),
            "area_ratio": float(mean_pred_area / (mean_gt_area + 1e-8)),
            "area_diff": float(mean_pred_area - mean_gt_area),
            "mean_pred_area_per_image": float(mean_pred_area),
            "mean_gt_area_per_image": float(mean_gt_area),
        }
    )
    return metrics


def save_json(path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

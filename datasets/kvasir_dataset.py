from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_KVASIR_ROOT = Path(r"E:\Thesis_Dataset\kvasir-seg\Kvasir-SEG")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _is_supported_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def _scan_named_files(directory: Path) -> Dict[str, Path]:
    return {path.stem: path for path in sorted(directory.iterdir()) if _is_supported_file(path)}


def _resolve_pair_path(path_value: str, root_dir: Optional[Path]) -> Path:
    path = Path(path_value)
    if path.is_absolute() or root_dir is None:
        return path
    return root_dir / path


def discover_kvasir_rows(
    root_dir: Path | str,
    split: str = "test",
    dataset: str = "kvasir",
    source: str = "external_test",
) -> List[Dict[str, str]]:
    root_dir = Path(root_dir)
    image_dir = root_dir / "images"
    mask_dir = root_dir / "masks"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing Kvasir image directory: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing Kvasir mask directory: {mask_dir}")

    image_map = _scan_named_files(image_dir)
    mask_map = _scan_named_files(mask_dir)

    missing_masks = sorted(set(image_map) - set(mask_map))
    missing_images = sorted(set(mask_map) - set(image_map))
    if missing_masks or missing_images:
        raise FileNotFoundError(
            "Kvasir image/mask pairing mismatch: "
            f"missing_masks={len(missing_masks)} missing_images={len(missing_images)} "
            f"sample_missing_masks={missing_masks[:5]} sample_missing_images={missing_images[:5]}"
        )

    rows: List[Dict[str, str]] = []
    for stem in sorted(image_map):
        rows.append(
            {
                "image_path": (Path("images") / image_map[stem].name).as_posix(),
                "mask_path": (Path("masks") / mask_map[stem].name).as_posix(),
                "split": split,
                "dataset": dataset,
                "source": source,
            }
        )
    return rows


def read_kvasir_manifest(manifest_csv: Path | str, split: Optional[str] = "test") -> List[Dict[str, str]]:
    manifest_csv = Path(manifest_csv)
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if split is None:
        return rows
    filtered = []
    for row in rows:
        row_split = row.get("split", split)
        if row_split == split:
            filtered.append(row)
    return filtered


class KvasirSegDataset(Dataset):
    def __init__(
        self,
        root_dir: Optional[Path | str] = None,
        rows: Optional[Sequence[Dict[str, str]]] = None,
        manifest_csv: Optional[Path | str] = None,
        image_size: int = 256,
        imagenet_norm: bool = False,
        split: str = "test",
    ) -> None:
        self.root_dir = Path(root_dir).resolve() if root_dir is not None else None
        if rows is None:
            if manifest_csv is not None:
                rows = read_kvasir_manifest(manifest_csv, split=split)
            else:
                if self.root_dir is None:
                    raise ValueError("KvasirSegDataset requires either rows, manifest_csv, or root_dir.")
                rows = discover_kvasir_rows(self.root_dir, split=split)

        self.rows = list(rows)
        self.image_size = int(image_size)
        self.imagenet_norm = bool(imagenet_norm)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_rel_path = row.get("image_path") or row.get("image")
        mask_rel_path = row.get("mask_path") or row.get("mask")
        if not image_rel_path or not mask_rel_path:
            raise KeyError(f"Kvasir row is missing image/mask paths: {row}")

        image_path = _resolve_pair_path(image_rel_path, self.root_dir)
        mask_path = _resolve_pair_path(mask_rel_path, self.root_dir)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        orig_width, orig_height = image.size
        orig_mask_np = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        if self.imagenet_norm:
            image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD

        mask_np = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        metadata = {
            "index": int(index),
            "filename": image_path.name,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_rel_path": Path(image_rel_path).as_posix(),
            "mask_rel_path": Path(mask_rel_path).as_posix(),
            "split": row.get("split", "test"),
            "dataset": row.get("dataset", "kvasir"),
            "source": row.get("source", "external_test"),
            "orig_height": int(orig_height),
            "orig_width": int(orig_width),
            "resized_height": self.image_size,
            "resized_width": self.image_size,
            "mask_area_fraction": float(orig_mask_np.mean()),
        }
        return image_tensor, mask_tensor, metadata

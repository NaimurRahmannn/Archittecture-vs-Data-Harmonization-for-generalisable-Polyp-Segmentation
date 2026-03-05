import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description="Create sanity overlays from manifest groups.")
    parser.add_argument("--manifest", default=r"E:\Thesis_Code\splits\manifest.csv")
    parser.add_argument(
        "--data_root",
        default=r"E:\Thesis_Dataset\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3",
    )
    parser.add_argument("--out_dir", default=r"E:\Thesis_Code\debug_overlays")
    parser.add_argument("--samples_per_group", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_manifest(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def group_rows(rows):
    return {
        "train": [row for row in rows if row["split"] == "train"],
        "data2": [row for row in rows if row["split"] == "test" and row["test_group"] == "data2"],
        "data3": [row for row in rows if row["split"] == "test" and row["test_group"] == "data3"],
        "data4": [row for row in rows if row["split"] == "test" and row["test_group"] == "data4"],
    }


def get_seq_id(image_rel):
    match = re.search(r"sequenceData/positive/(seq\d+)/", image_rel.replace("\\", "/"))
    if match:
        return match.group(1)
    return "-"


def make_overlay(image_path, mask_path, label):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image_np = np.asarray(image, dtype=np.float32)
    mask_np = np.asarray(mask, dtype=np.uint8) > 127

    overlay_np = image_np.copy()
    alpha = 0.45
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    overlay_np[mask_np] = (1.0 - alpha) * overlay_np[mask_np] + alpha * red
    overlay = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8))

    drawer = ImageDraw.Draw(overlay)
    drawer.rectangle([(5, 5), (5 + 12 * len(label), 30)], fill=(0, 0, 0))
    drawer.text((10, 10), label, fill=(255, 255, 255))
    return overlay


def main():
    args = parse_args()
    rows = read_manifest(args.manifest)
    grouped = group_rows(rows)
    rng = random.Random(args.seed)

    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for group_name, items in grouped.items():
        if not items:
            print(f"{group_name}: 0 rows")
            continue

        picked = items[:]
        rng.shuffle(picked)
        picked = picked[: min(args.samples_per_group, len(picked))]

        group_dir = out_root / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in enumerate(picked):
            image_path = data_root / row["image"]
            mask_path = data_root / row["mask"]
            seq_id = get_seq_id(row["image"])
            label = f"split={row['split']} group={group_name} centre={row['center']} seq_id={seq_id}"

            overlay = make_overlay(image_path, mask_path, label)
            stem = Path(row["image"]).stem
            out_path = group_dir / f"{idx:02d}_{stem}.jpg"
            overlay.save(out_path, quality=95)

        print(f"{group_name}: wrote {len(picked)} overlays to {group_dir}")


if __name__ == "__main__":
    main()

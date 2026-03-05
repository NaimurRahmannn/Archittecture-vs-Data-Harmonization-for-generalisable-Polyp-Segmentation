import argparse
import csv
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Build train/val manifest for PolypGen single-frame data (excluding NBI).")
    p.add_argument("--root", required=True, help="Dataset root folder (PolypGen2021_MultiCenterData_v3)")
    p.add_argument("--out_csv", required=True, help="Output CSV path")
    p.add_argument("--out_json", required=True, help="Output JSON summary path")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    return p.parse_args()


def resolve_mask(masks_dir: Path, img_path: Path) -> Path | None:
    mask_path = masks_dir / img_path.name
    if mask_path.exists():
        return mask_path

    mask_path = masks_dir / f"{img_path.stem}_mask{img_path.suffix}"
    if mask_path.exists():
        return mask_path

    return None


def collect_samples(root: Path):
    samples = []
    missing_masks = 0
    excluded_nbi = 0

    data_dirs = sorted([p for p in root.glob("data_C*") if p.is_dir()])
    for data_dir in data_dirs:
        center = data_dir.name.split("_")[-1]
        images_dir = data_dir / f"images_{center}"
        masks_dir = data_dir / f"masks_{center}"
        if not images_dir.exists():
            continue

        for img_path in sorted(images_dir.glob("*.jpg")):
            if "nbi" in img_path.name.lower():
                excluded_nbi += 1
                continue
            mask_path = resolve_mask(masks_dir, img_path)
            if mask_path is None:
                missing_masks += 1
                continue
            samples.append(
                {
                    "image": img_path,
                    "mask": mask_path,
                    "center": center,
                }
            )

    return samples, excluded_nbi, missing_masks


def split_samples(samples, val_ratio, seed):
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    val_size = int(len(samples) * val_ratio)
    val_set = set(indices[:val_size])

    records = []
    for i, sample in enumerate(samples):
        split = "val" if i in val_set else "train"
        records.append({**sample, "split": split})
    return records


def write_csv(records, root: Path, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "mask", "center", "split"])
        for r in records:
            image_rel = r["image"].relative_to(root).as_posix()
            mask_rel = r["mask"].relative_to(root).as_posix()
            writer.writerow([image_rel, mask_rel, r["center"], r["split"]])


def write_summary(records, root: Path, out_json: Path, val_ratio, seed, excluded_nbi, missing_masks):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    total = len(records)
    train = sum(1 for r in records if r["split"] == "train")
    val = total - train
    per_center = {}
    for r in records:
        per_center.setdefault(r["center"], {"train": 0, "val": 0, "total": 0})
        per_center[r["center"]]["total"] += 1
        per_center[r["center"]][r["split"]] += 1

    summary = {
        "dataset_root": str(root),
        "total": total,
        "train": train,
        "val": val,
        "val_ratio": val_ratio,
        "seed": seed,
        "excluded_nbi": excluded_nbi,
        "missing_masks": missing_masks,
        "per_center": per_center,
    }

    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    samples, excluded_nbi, missing_masks = collect_samples(root)
    if not samples:
        raise SystemExit("No samples found. Check dataset root and folder structure.")

    records = split_samples(samples, args.val_ratio, args.seed)
    write_csv(records, root, Path(args.out_csv))
    write_summary(records, root, Path(args.out_json), args.val_ratio, args.seed, excluded_nbi, missing_masks)


if __name__ == "__main__":
    main()

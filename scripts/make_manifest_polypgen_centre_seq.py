import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a deterministic PolypGen manifest for centre + sequence generalisation (no NBI)."
    )
    parser.add_argument("--root", required=True, help="Dataset root folder")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--out_json", required=True, help="Output JSON summary path")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Validation split ratio for C1..C5")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for stratified validation split")
    return parser.parse_args()


def is_nbi(path: Path) -> bool:
    return "nbi" in path.name.lower()


def to_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def resolve_single_mask(masks_dir: Path, image_path: Path):
    candidates = [
        masks_dir / image_path.name,
        masks_dir / f"{image_path.stem}_mask{image_path.suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidates[-1]
    return None, candidates[-1]


def is_c6_sequence_name(name: str) -> bool:
    if "_C6_" in name:
        return True
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", Path(name).stem) if token]
    return any(token.upper() == "C6" for token in tokens)


def collect_single_frames(root: Path):
    train_candidates = []
    test_rows = []
    missing_cases = []

    for center_idx in range(1, 7):
        center = f"C{center_idx}"
        data_dir = root / f"data_{center}"
        images_dir = data_dir / f"images_{center}"
        masks_dir = data_dir / f"masks_{center}"
        if not images_dir.exists():
            continue

        for image_path in sorted(images_dir.glob("*.jpg")):
            if is_nbi(image_path):
                continue

            mask_path, expected_mask = resolve_single_mask(masks_dir, image_path)
            if mask_path is None:
                missing_cases.append(
                    {
                        "image": to_rel(image_path, root),
                        "expected_mask": to_rel(expected_mask, root),
                    }
                )
                continue

            row = {
                "image": to_rel(image_path, root),
                "mask": to_rel(mask_path, root),
                "center": center,
                "split": "train" if center_idx <= 5 else "test",
                "test_group": "" if center_idx <= 5 else "data2",
            }
            if center_idx <= 5:
                train_candidates.append(row)
            else:
                test_rows.append(row)

    return train_candidates, test_rows, missing_cases


def collect_sequence_frames(root: Path):
    rows = []
    missing = []
    seq_root = root / "sequenceData" / "positive"
    if not seq_root.exists():
        return rows, missing

    for seq_dir in sorted(path for path in seq_root.iterdir() if path.is_dir()):
        images_dirs = sorted(path for path in seq_dir.iterdir() if path.is_dir() and path.name.startswith("images_"))
        masks_dirs = sorted(path for path in seq_dir.iterdir() if path.is_dir() and path.name.startswith("masks_"))
        if not images_dirs or not masks_dirs:
            continue

        images_dir = images_dirs[0]
        masks_dir = masks_dirs[0]

        for image_path in sorted(images_dir.glob("*.jpg")):
            if is_nbi(image_path):
                continue

            mask_path = masks_dir / f"{image_path.stem}_mask{image_path.suffix}"
            if not mask_path.exists():
                missing.append(
                    {
                        "image": to_rel(image_path, root),
                        "expected_mask": to_rel(mask_path, root),
                    }
                )
                continue

            if is_c6_sequence_name(image_path.name):
                center = "C6"
                test_group = "data4"
            else:
                center = "CUNK"
                test_group = "data3"

            rows.append(
                {
                    "image": to_rel(image_path, root),
                    "mask": to_rel(mask_path, root),
                    "center": center,
                    "split": "test",
                    "test_group": test_group,
                }
            )

    return rows, missing


def assign_validation(rows, val_ratio: float, seed: int):
    rng = random.Random(seed)
    by_center = defaultdict(list)
    for row in rows:
        by_center[row["center"]].append(row)

    assigned = []
    for center in sorted(by_center):
        group = sorted(by_center[center], key=lambda item: item["image"])
        rng.shuffle(group)
        val_count = int(len(group) * val_ratio)
        val_images = {row["image"] for row in group[:val_count]}
        for row in sorted(group, key=lambda item: item["image"]):
            row_copy = dict(row)
            row_copy["split"] = "val" if row_copy["image"] in val_images else "train"
            assigned.append(row_copy)

    assigned.sort(key=lambda item: item["image"])
    return assigned


def write_manifest(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "mask", "center", "split", "test_group"])
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows, missing_masks_single, missing_masks_seq):
    split_counts = Counter()
    test_group_counts = Counter()
    centre_counts = Counter()

    for row in rows:
        split_counts[row["split"]] += 1
        centre_counts[row["center"]] += 1
        if row["test_group"]:
            test_group_counts[row["test_group"]] += 1

    return {
        "total_rows": len(rows),
        "missing_masks_single": missing_masks_single,
        "missing_masks_seq": missing_masks_seq,
        "split_counts": dict(sorted(split_counts.items())),
        "test_group_counts": dict(sorted(test_group_counts.items())),
        "centre_counts": dict(sorted(centre_counts.items())),
    }


def write_summary(summary, out_json: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

    train_candidates, single_test_rows, missing_single = collect_single_frames(root)
    train_val_rows = assign_validation(train_candidates, args.val_ratio, args.seed)
    sequence_rows, missing_seq = collect_sequence_frames(root)

    all_rows = train_val_rows + single_test_rows + sequence_rows
    all_rows.sort(key=lambda item: item["image"])

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    write_manifest(all_rows, out_csv)

    summary = build_summary(
        all_rows,
        missing_masks_single=len(missing_single),
        missing_masks_seq=len(missing_seq),
    )
    write_summary(summary, out_json)

    if missing_single:
        first_missing = missing_single[0]
        print("Missing single-frame mask:")
        print(json.dumps(first_missing, indent=2))
    else:
        print("Missing single-frame mask: null")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

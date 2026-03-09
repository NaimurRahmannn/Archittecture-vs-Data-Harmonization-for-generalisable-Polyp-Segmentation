import argparse
import csv
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create SeqVal split from data3 test rows.")
    parser.add_argument("--in_manifest", default=r"E:\Thesis_Code\splits\manifest.csv")
    parser.add_argument("--out_manifest", default=r"E:\Thesis_Code\splits\manifest_seqval.csv")
    parser.add_argument("--out_summary", default=r"E:\Thesis_Code\splits\seqval_summary.json")
    parser.add_argument("--seqval_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    in_manifest = Path(args.in_manifest)
    out_manifest = Path(args.out_manifest)
    out_summary = Path(args.out_summary)

    if not in_manifest.exists():
        raise SystemExit(f"Input manifest not found: {in_manifest}")

    with in_manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames or "split" not in fieldnames or "test_group" not in fieldnames:
        raise SystemExit("Manifest must contain columns: split, test_group")

    candidate_indices = [
        idx
        for idx, row in enumerate(rows)
        if row.get("split") == "test" and row.get("test_group") == "data3"
    ]

    rng = random.Random(args.seed)
    seqval_count = int(len(candidate_indices) * args.seqval_ratio)
    seqval_indices = set(rng.sample(candidate_indices, seqval_count))

    for idx in seqval_indices:
        rows[idx]["split"] = "seqval"

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {
        "train": 0,
        "val": 0,
        "seqval": 0,
        "test_data2": 0,
        "test_data3_remaining": 0,
        "test_data4": 0,
    }
    for row in rows:
        split = row.get("split")
        group = row.get("test_group")
        if split == "train":
            counts["train"] += 1
        elif split == "val":
            counts["val"] += 1
        elif split == "seqval":
            counts["seqval"] += 1
        elif split == "test" and group == "data2":
            counts["test_data2"] += 1
        elif split == "test" and group == "data3":
            counts["test_data3_remaining"] += 1
        elif split == "test" and group == "data4":
            counts["test_data4"] += 1

    summary = {
        "input_manifest": str(in_manifest),
        "output_manifest": str(out_manifest),
        "seqval_ratio": args.seqval_ratio,
        "seed": args.seed,
        "data3_candidates": len(candidate_indices),
        "seqval_selected": seqval_count,
        **counts,
    }

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

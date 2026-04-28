from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.kvasir_dataset import DEFAULT_KVASIR_ROOT, discover_kvasir_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Build a Kvasir-SEG external-test manifest.")
    parser.add_argument("--kvasir_root", default=str(DEFAULT_KVASIR_ROOT))
    parser.add_argument("--output", default=str(ROOT / "kvasir_manifest.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    rows = discover_kvasir_rows(args.kvasir_root, split="test", dataset="kvasir", source="external_test")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "mask_path", "split", "dataset", "source"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote={out_path} rows={len(rows)} kvasir_root={Path(args.kvasir_root).resolve()}")


if __name__ == "__main__":
    main()

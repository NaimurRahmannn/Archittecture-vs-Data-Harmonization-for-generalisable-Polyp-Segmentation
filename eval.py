import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pipeline_common import (
    ManifestSegDataset,
    build_model_from_config,
    evaluate_model,
    filter_manifest,
    load_simple_yaml,
    read_manifest,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate M0 U-Net checkpoint by test group.")
    parser.add_argument("--ckpt", required=True, help="Path to best.pt")
    parser.add_argument("--group", required=True, choices=["data2", "data3", "data4"])
    parser.add_argument("--config", default=None, help="Optional config path to override checkpoint config")
    parser.add_argument("--epoch_tag", default=None, help="Optional epoch tag, e.g. epoch10")
    parser.add_argument("--model_tag", default=None, help="Optional model tag for output filename, e.g. m2b")
    parser.add_argument("--out_name", default=None, help="Optional explicit output filename")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt).resolve()
    checkpoint = torch.load(ckpt_path, map_location=device)

    if args.config:
        cfg = load_simple_yaml(args.config)
    else:
        cfg = checkpoint.get("config", {})

    model_name = cfg.get("model_name", "unknown")
    ckpt_mtime = datetime.fromtimestamp(ckpt_path.stat().st_mtime).isoformat(sep=" ", timespec="seconds")
    print(f"model_name={model_name}")
    print(f"ckpt={ckpt_path}")
    print(f"ckpt_mtime={ckpt_mtime}")

    required = ["manifest_csv", "data_root"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise SystemExit(f"Missing config keys: {missing}. Pass --config or use checkpoint saved by train.py")

    rows = read_manifest(cfg["manifest_csv"])
    group_rows = filter_manifest(rows, split="test", test_group=args.group)
    if not group_rows:
        raise SystemExit(f"No rows found for group={args.group}")

    dataset = ManifestSegDataset(group_rows, cfg["data_root"], image_size=cfg.get("image_size", 256))
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])

    metrics = evaluate_model(model, loader, device)
    payload = {"group": args.group, "num_samples": len(group_rows), **metrics}
    print(payload)

    run_dir = ckpt_path.parent
    if args.out_name:
        out_path = run_dir / args.out_name
    elif args.epoch_tag:
        resolved_model_tag = args.model_tag or str(model_name).lower().split("_")[0]
        out_path = run_dir / f"metrics_{resolved_model_tag}_{args.epoch_tag}_{args.group}.json"
    else:
        out_path = run_dir / f"metrics_{args.group}.json"
    save_json(out_path, payload)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()

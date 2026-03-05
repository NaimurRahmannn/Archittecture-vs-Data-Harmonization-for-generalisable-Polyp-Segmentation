import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pipeline_common import (
    ManifestSegDataset,
    UNet,
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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.ckpt, map_location=device)

    if args.config:
        cfg = load_simple_yaml(args.config)
    else:
        cfg = checkpoint.get("config", {})

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

    model = UNet(in_channels=3, out_channels=1, base_ch=cfg.get("base_ch", 32)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    metrics = evaluate_model(model, loader, device)
    payload = {"group": args.group, "num_samples": len(group_rows), **metrics}
    print(payload)

    run_dir = Path(args.ckpt).resolve().parent
    out_path = run_dir / f"metrics_{args.group}.json"
    save_json(out_path, payload)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()

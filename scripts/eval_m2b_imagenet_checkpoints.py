import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    parser = argparse.ArgumentParser(description="Evaluate M2b ImageNet checkpoints on data2/data3/data4.")
    parser.add_argument("--config", required=True, help="Path to seed config yaml.")
    return parser.parse_args()


def build_loader(rows, cfg):
    dataset = ManifestSegDataset(
        rows,
        cfg["data_root"],
        image_size=cfg.get("image_size", 256),
        imagenet_norm=cfg.get("imagenet_norm", False),
        augment=False,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )


def main():
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    run_dir = Path(cfg["run_dir"])
    rows = read_manifest(cfg["manifest_csv"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_specs = [
        ("best_by_valdice", run_dir / "best_by_valdice.pt"),
        ("best_by_seqvalf2", run_dir / "best_by_seqvalf2.pt"),
        ("best_by_seqvalf2_constrained", run_dir / "best_by_seqvalf2_constrained.pt"),
    ]
    groups = ("data2", "data3", "data4")

    for strategy, ckpt_path in ckpt_specs:
        if not ckpt_path.exists():
            print(f"skip={strategy} missing_ckpt={ckpt_path}")
            continue

        checkpoint = torch.load(ckpt_path, map_location=device)
        model = build_model_from_config(cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
        ckpt_epoch = checkpoint.get("epoch")
        print(f"eval strategy={strategy} epoch={ckpt_epoch} ckpt={ckpt_path}")

        for group in groups:
            group_rows = filter_manifest(rows, split="test", test_group=group)
            if not group_rows:
                print(f"skip strategy={strategy} group={group} no_rows")
                continue
            loader = build_loader(group_rows, cfg)
            metrics = evaluate_model(model, loader, device, threshold=0.5)
            payload = {
                "strategy": strategy,
                "ckpt": str(ckpt_path),
                "ckpt_epoch": ckpt_epoch,
                "group": group,
                "num_samples": len(group_rows),
                **metrics,
            }
            out_path = run_dir / f"metrics_{strategy}_{group}.json"
            save_json(out_path, payload)
            print(f"saved={out_path}")


if __name__ == "__main__":
    main()

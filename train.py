import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pipeline_common import (
    ManifestSegDataset,
    UNet,
    dice_bce_loss,
    evaluate_model,
    filter_manifest,
    load_simple_yaml,
    read_manifest,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal M0 U-Net training pipeline.")
    parser.add_argument("--config", default="configs/m0_unet.yaml")
    return parser.parse_args()


def build_loader(rows, cfg, shuffle):
    dataset = ManifestSegDataset(
        rows,
        cfg["data_root"],
        image_size=cfg.get("image_size", 256),
        augment=shuffle and cfg.get("augment", True),
        rotate_limit=cfg.get("rotate_limit", 20.0),
        brightness_range=cfg.get("brightness_range", 0.2),
        contrast_range=cfg.get("contrast_range", 0.2),
        gamma_range=cfg.get("gamma_range", 0.2),
    )
    return DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 8),
        shuffle=shuffle,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )


def main():
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    set_seed(cfg.get("seed", 0))

    manifest_rows = read_manifest(cfg["manifest_csv"])
    train_rows = filter_manifest(manifest_rows, split="train")
    val_rows = filter_manifest(manifest_rows, split="val")

    if not train_rows or not val_rows:
        raise SystemExit("Train/val rows are empty. Check manifest and config paths.")

    train_loader = build_loader(train_rows, cfg, shuffle=True)
    val_loader = build_loader(val_rows, cfg, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1, base_ch=cfg.get("base_ch", 32)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    epochs = int(cfg.get("epochs", 20))

    run_dir = Path(cfg.get("run_dir", "runs/m0_unet"))
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = run_dir / "best.pt"
    best_val_dice = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = dice_bce_loss(logits, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate_model(model, val_loader, device)
        val_dice = val_metrics["Dice"]
        print(f"epoch={epoch} train_loss={train_loss:.5f} val_dice={val_dice:.5f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "best_val_dice": best_val_dice,
                    "epoch": epoch,
                },
                best_ckpt,
            )
            print(f"saved_best={best_ckpt} val_dice={best_val_dice:.5f}")

    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    for group in ("data2", "data3", "data4"):
        group_rows = filter_manifest(manifest_rows, split="test", test_group=group)
        if not group_rows:
            print(f"{group}: skipped (no rows)")
            continue
        group_loader = build_loader(group_rows, cfg, shuffle=False)
        metrics = evaluate_model(model, group_loader, device)
        metrics_payload = {"group": group, "num_samples": len(group_rows), **metrics}
        out_path = run_dir / f"metrics_{group}.json"
        save_json(out_path, metrics_payload)
        print(f"{group}: {metrics_payload}")


if __name__ == "__main__":
    main()

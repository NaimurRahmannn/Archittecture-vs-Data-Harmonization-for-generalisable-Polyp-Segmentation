import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pipeline_common import (
    ManifestSegDataset,
    build_model_from_config,
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


def collect_alpha_beta_stats(model):
    alpha_values = []
    beta_values = []
    for name, param in model.named_parameters():
        if name.endswith("alpha_logit"):
            alpha_values.append(torch.sigmoid(param.detach()).item())
        elif name.endswith("beta_logit"):
            beta_values.append(torch.sigmoid(param.detach()).item())
    if not alpha_values and not beta_values:
        return None
    alpha_mean = sum(alpha_values) / max(1, len(alpha_values))
    beta_mean = sum(beta_values) / max(1, len(beta_values))
    return alpha_mean, beta_mean


def parse_snapshot_eval_epochs(cfg):
    raw = cfg.get("snapshot_eval_epochs", "10,25,50")
    if isinstance(raw, (int, float)):
        return {int(raw)}
    if isinstance(raw, str):
        values = []
        for item in raw.split(","):
            token = item.strip()
            if token:
                values.append(int(token))
        return set(values)
    return {10, 25, 50}


def build_model_tag(cfg):
    model_name = str(cfg.get("model_name", "model")).lower().strip()
    return model_name.split("_")[0] if model_name else "model"


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
    model = build_model_from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    epochs = int(cfg.get("epochs", 50))

    run_dir = Path(cfg.get("run_dir", "runs/m0_unet"))
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = run_dir / "best.pt"
    best_val_dice = -1.0
    snapshot_eval_epochs = parse_snapshot_eval_epochs(cfg)
    model_tag = build_model_tag(cfg)

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
        gate_stats = collect_alpha_beta_stats(model)
        if gate_stats is None:
            print(f"epoch={epoch} train_loss={train_loss:.5f} val_dice={val_dice:.5f}")
        else:
            alpha_mean, beta_mean = gate_stats
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} val_dice={val_dice:.5f} "
                f"alpha_mean={alpha_mean:.4f} beta_mean={beta_mean:.4f}"
            )

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

        if epoch in snapshot_eval_epochs:
            for group in ("data2", "data3", "data4"):
                group_rows = filter_manifest(manifest_rows, split="test", test_group=group)
                if not group_rows:
                    continue
                group_loader = build_loader(group_rows, cfg, shuffle=False)
                metrics = evaluate_model(model, group_loader, device)
                metrics_payload = {"epoch": epoch, "group": group, "num_samples": len(group_rows), **metrics}
                out_path = run_dir / f"metrics_{model_tag}_epoch{epoch}_{group}.json"
                save_json(out_path, metrics_payload)
                print(f"saved_snapshot_metrics={out_path}")

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

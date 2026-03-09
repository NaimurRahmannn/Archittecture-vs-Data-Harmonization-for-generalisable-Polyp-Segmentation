import argparse
import csv
import math
import shutil
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


def save_strategy_checkpoint(path, model, cfg, epoch, val_dice, seqval_f2, strategy):
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "val_dice": val_dice,
            "seqval_f2": seqval_f2,
            "selection_strategy": strategy,
        },
        path,
    )


def evaluate_groups(model, manifest_rows, cfg, device, threshold=0.5):
    out = {}
    for group in ("data2", "data3", "data4"):
        group_rows = filter_manifest(manifest_rows, split="test", test_group=group)
        if not group_rows:
            continue
        group_loader = build_loader(group_rows, cfg, shuffle=False)
        out[group] = {"num_samples": len(group_rows), **evaluate_model(model, group_loader, device, threshold=threshold)}
    return out


def update_checkpoint_selection_csv(csv_path, model_name, strategy_rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "strategy",
        "ckpt_path",
        "data2_Dice",
        "data2_F2",
        "data2_area_ratio",
        "data3_Dice",
        "data3_F2",
        "data3_area_ratio",
        "data4_Dice",
        "data4_F2",
        "data4_area_ratio",
        "data4_minus_data3_F2",
        "data4_minus_data3_area_ratio",
    ]

    existing = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    key_pairs = {(row["model_name"], row["strategy"]) for row in strategy_rows}
    existing = [row for row in existing if (row.get("model_name"), row.get("strategy")) not in key_pairs]
    merged = existing + strategy_rows

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)


def write_seqval_epoch_log(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "ckpt_path", "seqval_f2", "seqval_dice", "seqval_area_ratio", "seqval_area_diff"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pick_constrained_seqval_candidate(epoch_rows):
    if not epoch_rows:
        return None

    def by_f2_desc(items):
        return sorted(items, key=lambda r: float(r["seqval_f2"]), reverse=True)

    strict = [r for r in epoch_rows if 0.85 <= float(r["seqval_area_ratio"]) <= 1.15]
    if strict:
        return by_f2_desc(strict)[0]

    relaxed = [r for r in epoch_rows if 0.80 <= float(r["seqval_area_ratio"]) <= 1.20]
    if relaxed:
        return by_f2_desc(relaxed)[0]

    top5 = by_f2_desc(epoch_rows)[:5]
    return min(top5, key=lambda r: abs(float(r["seqval_area_ratio"]) - 1.0))


def sweep_thresholds(model, loader, device, thresholds):
    best_dice = None
    best_f2 = None
    for t in thresholds:
        metrics = evaluate_model(model, loader, device, threshold=t)
        row = {"threshold": t, **metrics}
        if best_dice is None or row["Dice"] > best_dice["Dice"]:
            best_dice = row
        if best_f2 is None or row["F2"] > best_f2["F2"]:
            best_f2 = row
    return best_dice, best_f2


def update_checkpoint_selection_csv_v2(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "strategy",
        "threshold_mode",
        "threshold_value",
        "ckpt_path",
        "data2_Dice",
        "data2_F2",
        "data2_area_ratio",
        "data3_Dice",
        "data3_F2",
        "data3_area_ratio",
        "data4_Dice",
        "data4_F2",
        "data4_area_ratio",
        "gap_F2",
        "gap_area_ratio",
    ]

    existing = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    key_pairs = {(r["model_name"], r["strategy"], r["threshold_mode"]) for r in rows}
    existing = [
        r for r in existing if (r.get("model_name"), r.get("strategy"), r.get("threshold_mode")) not in key_pairs
    ]
    merged = existing + rows

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

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
    seqval_rows = filter_manifest(manifest_rows, split="seqval")

    if not train_rows or not val_rows:
        raise SystemExit("Train/val rows are empty. Check manifest and config paths.")

    train_loader = build_loader(train_rows, cfg, shuffle=True)
    val_loader = build_loader(val_rows, cfg, shuffle=False)
    seqval_loader = build_loader(seqval_rows, cfg, shuffle=False) if seqval_rows else None

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
    best_by_valdice_ckpt = run_dir / "best_by_valdice.pt"
    best_by_seqvalf2_ckpt = run_dir / "best_by_seqvalf2.pt"
    best_by_seqvalf2_constrained_ckpt = run_dir / "best_by_seqvalf2_constrained.pt"
    epoch_ckpt_dir = run_dir / "epoch_ckpts"
    epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_dice = -1.0
    best_seqval_f2 = -1.0
    snapshot_eval_epochs = parse_snapshot_eval_epochs(cfg)
    model_tag = build_model_tag(cfg)
    seqval_epoch_rows = []

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
        seqval_f2 = float("nan")
        seqval_dice = float("nan")
        seqval_area_ratio = float("nan")
        seqval_area_diff = float("nan")
        if seqval_loader is not None:
            seqval_metrics = evaluate_model(model, seqval_loader, device)
            seqval_f2 = seqval_metrics["F2"]
            seqval_dice = seqval_metrics["Dice"]
            seqval_area_ratio = seqval_metrics["area_ratio"]
            seqval_area_diff = seqval_metrics["area_diff"]

            epoch_ckpt_path = epoch_ckpt_dir / f"epoch_{epoch:03d}.pt"
            save_strategy_checkpoint(epoch_ckpt_path, model, cfg, epoch, val_dice, seqval_f2, "seqval_candidate")
            seqval_epoch_rows.append(
                {
                    "epoch": epoch,
                    "ckpt_path": str(epoch_ckpt_path),
                    "seqval_f2": seqval_f2,
                    "seqval_dice": seqval_dice,
                    "seqval_area_ratio": seqval_area_ratio,
                    "seqval_area_diff": seqval_area_diff,
                }
            )

        gate_stats = collect_alpha_beta_stats(model)
        if gate_stats is None:
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} val_dice={val_dice:.5f} "
                f"seqval_f2={seqval_f2:.5f}"
            )
        else:
            alpha_mean, beta_mean = gate_stats
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} val_dice={val_dice:.5f} seqval_f2={seqval_f2:.5f} "
                f"alpha_mean={alpha_mean:.4f} beta_mean={beta_mean:.4f}"
            )

        saved_ckpts = []
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_strategy_checkpoint(best_by_valdice_ckpt, model, cfg, epoch, val_dice, seqval_f2, "valdice")
            save_strategy_checkpoint(best_ckpt, model, cfg, epoch, val_dice, seqval_f2, "valdice")
            saved_ckpts.append(f"valdice->{best_by_valdice_ckpt}")

        if seqval_loader is not None and seqval_f2 > best_seqval_f2:
            best_seqval_f2 = seqval_f2
            save_strategy_checkpoint(best_by_seqvalf2_ckpt, model, cfg, epoch, val_dice, seqval_f2, "seqvalf2")
            saved_ckpts.append(f"seqvalf2->{best_by_seqvalf2_ckpt}")

        if saved_ckpts:
            print(f"saved_checkpoints={'; '.join(saved_ckpts)}")

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

    model_name = cfg.get("model_name", "unknown")
    seqval_epoch_log_csv = Path(r"E:\Thesis_Code\results\tables\m2b_seqval_epoch_log.csv")
    calibrated_threshold = 0.5
    calibrated_threshold_dice = 0.5
    constrained_candidate = None

    if seqval_loader is not None and seqval_epoch_rows:
        write_seqval_epoch_log(seqval_epoch_log_csv, seqval_epoch_rows)
        print(f"saved_seqval_epoch_log={seqval_epoch_log_csv}")

        constrained_candidate = pick_constrained_seqval_candidate(seqval_epoch_rows)
        if constrained_candidate is not None:
            shutil.copy2(constrained_candidate["ckpt_path"], best_by_seqvalf2_constrained_ckpt)
            print(
                "saved_constrained_checkpoint="
                f"{best_by_seqvalf2_constrained_ckpt} (epoch={constrained_candidate['epoch']}, "
                f"seqval_f2={constrained_candidate['seqval_f2']:.5f}, "
                f"seqval_area_ratio={constrained_candidate['seqval_area_ratio']:.5f})"
            )

        base_for_calibration = best_by_seqvalf2_ckpt if best_by_seqvalf2_ckpt.exists() else best_by_valdice_ckpt
        checkpoint = torch.load(base_for_calibration, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        thresholds = [round(v / 100.0, 2) for v in range(5, 96, 5)]
        best_dice_thresh_row, best_f2_thresh_row = sweep_thresholds(model, seqval_loader, device, thresholds)
        calibrated_threshold = float(best_f2_thresh_row["threshold"])
        calibrated_threshold_dice = float(best_dice_thresh_row["threshold"])
        print(
            f"threshold_calibration: best_f2_t={calibrated_threshold:.2f}, "
            f"best_dice_t={calibrated_threshold_dice:.2f}"
        )

    strategy_specs = [("valdice", best_by_valdice_ckpt)]
    if seqval_loader is not None and best_by_seqvalf2_ckpt.exists():
        strategy_specs.append(("seqvalf2", best_by_seqvalf2_ckpt))
    if seqval_loader is not None and best_by_seqvalf2_constrained_ckpt.exists():
        strategy_specs.append(("seqvalf2_constrained", best_by_seqvalf2_constrained_ckpt))

    comparison_rows = []
    summary_rows = []
    for strategy, ckpt_path in strategy_specs:
        if not ckpt_path.exists():
            continue
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        threshold_modes = [("default0.5", 0.5)]
        if seqval_loader is not None:
            threshold_modes.append(("calibrated", calibrated_threshold))

        for threshold_mode, threshold_value in threshold_modes:
            group_metrics = evaluate_groups(model, manifest_rows, cfg, device, threshold=threshold_value)

            mode_key = threshold_mode.replace(".", "p")
            for group, metrics in group_metrics.items():
                out_path = run_dir / f"metrics_{strategy}_{mode_key}_{group}.json"
                save_json(
                    out_path,
                    {
                        "strategy": strategy,
                        "threshold_mode": threshold_mode,
                        "threshold_value": threshold_value,
                        "group": group,
                        **metrics,
                    },
                )
                print(f"saved_strategy_metrics={out_path}")

            if not all(g in group_metrics for g in ("data2", "data3", "data4")):
                continue

            data2 = group_metrics["data2"]
            data3 = group_metrics["data3"]
            data4 = group_metrics["data4"]
            row = {
                "model_name": model_name,
                "strategy": strategy,
                "threshold_mode": threshold_mode,
                "threshold_value": threshold_value,
                "ckpt_path": str(ckpt_path),
                "data2_Dice": data2["Dice"],
                "data2_F2": data2["F2"],
                "data2_area_ratio": data2["area_ratio"],
                "data3_Dice": data3["Dice"],
                "data3_F2": data3["F2"],
                "data3_area_ratio": data3["area_ratio"],
                "data4_Dice": data4["Dice"],
                "data4_F2": data4["F2"],
                "data4_area_ratio": data4["area_ratio"],
                "gap_F2": data4["F2"] - data3["F2"],
                "gap_area_ratio": data4["area_ratio"] - data3["area_ratio"],
            }
            comparison_rows.append(row)
            summary_rows.append(row)

    comparison_csv_v2 = Path(r"E:\Thesis_Code\results\tables\checkpoint_selection_comparison_v2.csv")
    if comparison_rows:
        update_checkpoint_selection_csv_v2(comparison_csv_v2, comparison_rows)
        print(f"saved_comparison_csv_v2={comparison_csv_v2}")

    strategy_epoch_map = {}
    for strategy, ckpt_path in strategy_specs:
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            strategy_epoch_map[strategy] = ckpt.get("epoch")

    print(f"chosen_epochs={strategy_epoch_map}")
    print(f"chosen_threshold_f2={calibrated_threshold:.2f}, chosen_threshold_dice={calibrated_threshold_dice:.2f}")

    baseline = next(
        (
            r
            for r in summary_rows
            if r["strategy"] == "seqvalf2" and r["threshold_mode"] == "default0.5"
        ),
        None,
    )
    constrained = next(
        (
            r
            for r in summary_rows
            if r["strategy"] == "seqvalf2_constrained" and r["threshold_mode"] == "default0.5"
        ),
        None,
    )
    if baseline and constrained:
        base_dist = abs(float(baseline["data4_area_ratio"]) - 1.0)
        cons_dist = abs(float(constrained["data4_area_ratio"]) - 1.0)
        reduced = cons_dist < base_dist
        maintained = float(constrained["data4_F2"]) >= float(baseline["data4_F2"])
        print(
            "overseg_reduction="
            f"{reduced} (|area_ratio-1|: {base_dist:.4f} -> {cons_dist:.4f}), "
            f"data4_F2_maintained={maintained} ({baseline['data4_F2']:.4f} -> {constrained['data4_F2']:.4f})"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import heapq
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.kvasir_dataset import DEFAULT_KVASIR_ROOT, KvasirSegDataset, discover_kvasir_rows, read_kvasir_manifest
from pipeline_common import build_model_from_config, compute_metrics, load_simple_yaml, save_json, set_seed

DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7)
MODEL_ALIASES = {
    "m0": "m0_unet",
    "unet": "m0_unet",
    "m0_unet": "m0_unet",
    "m1": "m1_convnext_base",
    "m1_convnext_base": "m1_convnext_base",
    "convnext": "m1_convnext_base",
    "convnext_base": "m1_convnext_base",
    "m2": "m2_convnext_xattn",
    "m2_convnext_xattn": "m2_convnext_xattn",
    "m2b": "m2b_convnext_xattn",
    "m2b_convnext_xattn": "m2b_convnext_xattn",
    "deeplabv3": "m3_deeplabv3plus",
    "deeplabv3plus": "m3_deeplabv3plus",
    "m3": "m3_deeplabv3plus",
    "m3_deeplabv3plus": "m3_deeplabv3plus",
}
SIZE_GROUPS = ("small", "medium", "large")


@dataclass
class MetricAccumulator:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    pred_area_sum: float = 0.0
    gt_area_sum: float = 0.0
    num_images: int = 0

    def update(self, tp: float, fp: float, fn: float, pred_area: float, gt_area: float) -> None:
        self.tp += float(tp)
        self.fp += float(fp)
        self.fn += float(fn)
        self.pred_area_sum += float(pred_area)
        self.gt_area_sum += float(gt_area)
        self.num_images += 1

    def finalize(self) -> Dict[str, float | int | None]:
        if self.num_images == 0:
            return {
                "num_samples": 0,
                "Dice": None,
                "IoU": None,
                "Precision": None,
                "Recall": None,
                "F2": None,
                "mean_pred_area": None,
                "mean_gt_area": None,
                "area_ratio": None,
                "area_diff": None,
                "mean_pred_area_per_image": None,
                "mean_gt_area_per_image": None,
            }

        metrics = compute_metrics(self.tp, self.fp, self.fn)
        mean_pred_area = self.pred_area_sum / self.num_images
        mean_gt_area = self.gt_area_sum / self.num_images
        metrics.update(
            {
                "num_samples": self.num_images,
                "mean_pred_area": float(mean_pred_area),
                "mean_gt_area": float(mean_gt_area),
                "area_ratio": float(mean_pred_area / (mean_gt_area + 1e-8)),
                "area_diff": float(mean_pred_area - mean_gt_area),
                "mean_pred_area_per_image": float(mean_pred_area),
                "mean_gt_area_per_image": float(mean_gt_area),
            }
        )
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PolypGen/EndoCV checkpoint on Kvasir-SEG only.")
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument("--model", required=True, help="Model tag, e.g. M0, M1, M2, M2b, M3.")
    parser.add_argument("--config", default=None, help="Optional config yaml to override checkpoint config.")
    parser.add_argument("--manifest", default=str(ROOT / "kvasir_manifest.csv"))
    parser.add_argument("--kvasir_root", default=str(DEFAULT_KVASIR_ROOT))
    parser.add_argument("--img_size", type=int, choices=[256, 352, 512], default=256)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--report_threshold", type=float, default=0.5)
    parser.add_argument("--top_k_worst", type=int, default=20)
    parser.add_argument("--small_thresh", type=float, default=0.05, help="Small if GT area fraction < threshold.")
    parser.add_argument("--medium_thresh", type=float, default=0.15, help="Medium if GT area fraction < threshold.")
    parser.add_argument("--output_dir", default=str(ROOT / "outputs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cpu or cuda:0.")

    parser.set_defaults(imagenet_norm=None)
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument("--imagenet_norm", dest="imagenet_norm", action="store_true")
    norm_group.add_argument("--no_imagenet_norm", dest="imagenet_norm", action="store_false")
    return parser.parse_args()


def threshold_key(value: float) -> str:
    return f"{value:.2f}"


def normalize_thresholds(values: Sequence[float], report_threshold: float) -> tuple[list[float], float]:
    merged = sorted({round(float(value), 4) for value in list(values) + [report_threshold]})
    if not merged:
        raise ValueError("At least one threshold is required.")
    for value in merged:
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1: {value}")
    report_threshold = round(float(report_threshold), 4)
    return merged, report_threshold


def resolve_model_name(model_arg: str, fallback: str | None = None) -> str:
    token = str(model_arg or fallback or "").strip().lower().replace("-", "_")
    token = token.replace("+", "plus")
    if token in MODEL_ALIASES:
        return MODEL_ALIASES[token]
    if fallback:
        return str(fallback)
    raise ValueError(f"Unsupported model alias: {model_arg}")


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
    if isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ValueError("Checkpoint does not contain a supported state dict.")


def maybe_strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    if keys and all(key.startswith("module.") for key in keys):
        return {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def resolve_eval_config(checkpoint_payload: dict, args) -> Dict[str, object]:
    if args.config:
        cfg = load_simple_yaml(args.config)
    else:
        cfg = dict(checkpoint_payload.get("config", {}))

    cfg["model_name"] = resolve_model_name(args.model, cfg.get("model_name"))
    cfg["image_size"] = int(args.img_size)
    cfg["batch_size"] = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 8))
    cfg["num_workers"] = int(args.num_workers if args.num_workers is not None else cfg.get("num_workers", 0))
    cfg["imagenet_norm"] = bool(cfg.get("imagenet_norm", False) if args.imagenet_norm is None else args.imagenet_norm)
    cfg["pretrained"] = False
    return cfg


def build_rows(manifest_path: Path, kvasir_root: Path) -> tuple[list[dict[str, str]], str]:
    if manifest_path.exists():
        rows = read_kvasir_manifest(manifest_path, split="test")
        if rows:
            return rows, "manifest"
    rows = discover_kvasir_rows(kvasir_root, split="test", dataset="kvasir", source="external_test")
    return rows, "folder_scan"


def build_loader(rows: Sequence[dict[str, str]], cfg: Dict[str, object], kvasir_root: Path) -> DataLoader:
    dataset = KvasirSegDataset(
        root_dir=kvasir_root,
        rows=rows,
        image_size=int(cfg["image_size"]),
        imagenet_norm=bool(cfg.get("imagenet_norm", False)),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def extract_logits(model_output):
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, dict):
        for key in ("out", "logits", "mask"):
            if key in model_output and torch.is_tensor(model_output[key]):
                return model_output[key]
    if isinstance(model_output, (list, tuple)):
        for item in model_output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Unsupported model output type: {type(model_output)}")


def collated_to_list(values) -> List:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    return list(values)


def size_group_for_area(area_fraction: float, small_thresh: float, medium_thresh: float) -> str:
    if area_fraction < small_thresh:
        return "small"
    if area_fraction < medium_thresh:
        return "medium"
    return "large"


def maybe_push_worst_case(heap: list, record: dict, pred_mask: np.ndarray, top_k: int, counter: int) -> None:
    if top_k <= 0:
        return
    payload = dict(record)
    payload["pred_mask"] = pred_mask.astype(np.uint8)
    entry = (-float(record["Dice"]), counter, payload)
    if len(heap) < top_k:
        heapq.heappush(heap, entry)
    else:
        current_best_of_worst = -heap[0][0]
        if float(record["Dice"]) < current_best_of_worst:
            heapq.heapreplace(heap, entry)


def overlay_mask(image_np: np.ndarray, mask_bool: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> Image.Image:
    out = image_np.astype(np.float32).copy()
    color_np = np.array(color, dtype=np.float32)
    out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * color_np
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def build_error_overlay(image_np: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray, alpha: float = 0.50) -> Image.Image:
    out = image_np.astype(np.float32).copy()
    tp_mask = pred_mask & gt_mask
    fp_mask = pred_mask & ~gt_mask
    fn_mask = ~pred_mask & gt_mask

    out[tp_mask] = (1.0 - alpha) * out[tp_mask] + alpha * np.array([0.0, 255.0, 0.0], dtype=np.float32)
    out[fp_mask] = (1.0 - alpha) * out[fp_mask] + alpha * np.array([255.0, 0.0, 0.0], dtype=np.float32)
    out[fn_mask] = (1.0 - alpha) * out[fn_mask] + alpha * np.array([0.0, 0.0, 255.0], dtype=np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def make_visual_panel(record: dict, img_size: int) -> Image.Image:
    image = Image.open(record["image_path"]).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    gt_mask = Image.open(record["mask_path"]).convert("L").resize((img_size, img_size), Image.NEAREST)
    image_np = np.asarray(image, dtype=np.uint8)
    gt_mask_np = np.asarray(gt_mask, dtype=np.uint8) > 127
    pred_mask_np = np.asarray(record["pred_mask"], dtype=np.uint8) > 0

    input_panel = image
    gt_panel = overlay_mask(image_np, gt_mask_np, color=(0, 255, 0))
    pred_panel = overlay_mask(image_np, pred_mask_np, color=(255, 0, 0))
    error_panel = build_error_overlay(image_np, pred_mask_np, gt_mask_np)

    panels = [
        ("Input", input_panel),
        ("GT", gt_panel),
        ("Prediction", pred_panel),
        ("Error", error_panel),
    ]

    gap = 6
    header_h = 28
    footer_h = 34
    panel_w, panel_h = panels[0][1].size
    canvas_w = len(panels) * panel_w + (len(panels) - 1) * gap
    canvas_h = header_h + panel_h + footer_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    for idx, (title, panel) in enumerate(panels):
        x0 = idx * (panel_w + gap)
        canvas.paste(panel, (x0, header_h))
        draw.text((x0 + 8, 7), title, fill=(255, 255, 255))

    footer = (
        f"{record['filename']} | Dice={record['Dice']:.4f} | IoU={record['IoU']:.4f} "
        f"| F2={record['F2']:.4f} | size={record['size_group']}"
    )
    draw.text((8, header_h + panel_h + 9), footer, fill=(240, 240, 240))
    return canvas


def export_worst_case_visuals(worst_cases: Sequence[dict], visuals_dir: Path, img_size: int) -> List[dict]:
    visuals_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, record in enumerate(worst_cases, start=1):
        panel = make_visual_panel(record, img_size=img_size)
        stem = Path(record["filename"]).stem
        out_path = visuals_dir / f"{rank:02d}_{stem}.png"
        panel.save(out_path)

        row = {key: value for key, value in record.items() if key != "pred_mask"}
        row["rank"] = rank
        row["visual_path"] = str(out_path)
        rows.append(row)

    csv_path = visuals_dir / "worst_cases.csv"
    if rows:
        fieldnames = [
            "rank",
            "filename",
            "size_group",
            "mask_area_fraction_original",
            "Dice",
            "IoU",
            "Precision",
            "Recall",
            "F2",
            "pred_area",
            "gt_area",
            "area_ratio",
            "image_path",
            "mask_path",
            "visual_path",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def evaluate_dataset(
    model,
    loader: DataLoader,
    device: torch.device,
    thresholds: Sequence[float],
    report_threshold: float,
    small_thresh: float,
    medium_thresh: float,
    top_k_worst: int,
) -> tuple[dict[str, dict[str, dict]], list[dict]]:
    accumulators = {
        threshold: {"overall": MetricAccumulator(), "small": MetricAccumulator(), "medium": MetricAccumulator(), "large": MetricAccumulator()}
        for threshold in thresholds
    }
    worst_heap: list = []
    sample_counter = 0

    model.eval()
    with torch.no_grad():
        for images, masks, metadata in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = extract_logits(model(images))
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            probs = torch.sigmoid(logits)
            gt_masks = (masks > 0.5).float()

            filenames = collated_to_list(metadata["filename"])
            image_paths = collated_to_list(metadata["image_path"])
            mask_paths = collated_to_list(metadata["mask_path"])
            gt_area_original = [float(x) for x in collated_to_list(metadata["mask_area_fraction"])]
            size_groups = [size_group_for_area(area, small_thresh, medium_thresh) for area in gt_area_original]

            for threshold in thresholds:
                preds = (probs > threshold).float()
                tp_values = (preds * gt_masks).sum(dim=(1, 2, 3)).detach().cpu().tolist()
                fp_values = (preds * (1.0 - gt_masks)).sum(dim=(1, 2, 3)).detach().cpu().tolist()
                fn_values = ((1.0 - preds) * gt_masks).sum(dim=(1, 2, 3)).detach().cpu().tolist()
                pred_areas = preds.mean(dim=(1, 2, 3)).detach().cpu().tolist()
                gt_areas = gt_masks.mean(dim=(1, 2, 3)).detach().cpu().tolist()

                bucket = accumulators[threshold]
                for idx in range(len(tp_values)):
                    bucket["overall"].update(tp_values[idx], fp_values[idx], fn_values[idx], pred_areas[idx], gt_areas[idx])
                    bucket[size_groups[idx]].update(tp_values[idx], fp_values[idx], fn_values[idx], pred_areas[idx], gt_areas[idx])

                    if threshold == report_threshold:
                        image_metrics = compute_metrics(tp_values[idx], fp_values[idx], fn_values[idx])
                        record = {
                            "filename": filenames[idx],
                            "image_path": image_paths[idx],
                            "mask_path": mask_paths[idx],
                            "size_group": size_groups[idx],
                            "mask_area_fraction_original": gt_area_original[idx],
                            "pred_area": float(pred_areas[idx]),
                            "gt_area": float(gt_areas[idx]),
                            "area_ratio": float(pred_areas[idx] / (gt_areas[idx] + 1e-8)),
                            **image_metrics,
                        }
                        pred_mask = preds[idx, 0].detach().cpu().numpy().astype(np.uint8)
                        maybe_push_worst_case(worst_heap, record, pred_mask, top_k_worst, sample_counter)
                        sample_counter += 1

    finalized = {
        threshold_key(threshold): {subset: metric.finalize() for subset, metric in bucket.items()}
        for threshold, bucket in accumulators.items()
    }
    worst_cases = [entry[2] for entry in sorted(worst_heap, key=lambda item: (item[2]["Dice"], item[2]["filename"]))]
    return finalized, worst_cases


def write_threshold_results_csv(path: Path, threshold_metrics: dict[str, dict]) -> None:
    fieldnames = [
        "threshold",
        "num_samples",
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "F2",
        "mean_pred_area",
        "mean_gt_area",
        "area_ratio",
        "area_diff",
        "mean_pred_area_per_image",
        "mean_gt_area_per_image",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for threshold, metrics in threshold_metrics.items():
            writer.writerow({"threshold": threshold, **metrics})


def write_metrics_csv(path: Path, metrics_by_threshold: dict[str, dict[str, dict]]) -> None:
    fieldnames = [
        "threshold",
        "subset",
        "num_samples",
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "F2",
        "mean_pred_area",
        "mean_gt_area",
        "area_ratio",
        "area_diff",
        "mean_pred_area_per_image",
        "mean_gt_area_per_image",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for threshold, bucket in metrics_by_threshold.items():
            for subset, metrics in bucket.items():
                writer.writerow({"threshold": threshold, "subset": subset, **metrics})


def main():
    args = parse_args()
    if args.small_thresh >= args.medium_thresh:
        raise SystemExit("--small_thresh must be smaller than --medium_thresh.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    ckpt_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    kvasir_root = Path(args.kvasir_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    visuals_dir = output_dir / "kvasir_visuals"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = resolve_eval_config(checkpoint, args)
    state_dict = maybe_strip_module_prefix(extract_state_dict(checkpoint))

    rows, row_source = build_rows(manifest_path, kvasir_root)
    if not rows:
        raise SystemExit("No Kvasir rows found for evaluation.")

    thresholds, report_threshold = normalize_thresholds(args.thresholds, args.report_threshold)
    loader = build_loader(rows, cfg, kvasir_root)
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(state_dict, strict=True)

    metrics_by_threshold, worst_cases = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        thresholds=thresholds,
        report_threshold=report_threshold,
        small_thresh=float(args.small_thresh),
        medium_thresh=float(args.medium_thresh),
        top_k_worst=int(args.top_k_worst),
    )

    threshold_metrics = {threshold: bucket["overall"] for threshold, bucket in metrics_by_threshold.items()}
    selected_metrics = metrics_by_threshold[threshold_key(report_threshold)]
    worst_case_rows = export_worst_case_visuals(worst_cases, visuals_dir=visuals_dir, img_size=int(cfg["image_size"]))

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_alias": args.model,
        "model_name": cfg["model_name"],
        "img_size": int(cfg["image_size"]),
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg["num_workers"]),
        "imagenet_norm": bool(cfg["imagenet_norm"]),
        "device": str(device),
        "kvasir_root": str(kvasir_root),
        "manifest": str(manifest_path) if manifest_path.exists() else None,
        "row_source": row_source,
        "num_samples": len(rows),
        "thresholds": thresholds,
        "report_threshold": report_threshold,
        "size_thresholds": {
            "small_lt": float(args.small_thresh),
            "medium_lt": float(args.medium_thresh),
            "large_ge": float(args.medium_thresh),
        },
        "overall_by_threshold": threshold_metrics,
        "size_group_metrics_by_threshold": {
            threshold: {subset: metrics for subset, metrics in bucket.items() if subset != "overall"}
            for threshold, bucket in metrics_by_threshold.items()
        },
        "selected_threshold_metrics": selected_metrics,
        "worst_cases": worst_case_rows,
    }

    metrics_json_path = output_dir / "kvasir_metrics.json"
    metrics_csv_path = output_dir / "kvasir_metrics.csv"
    threshold_csv_path = output_dir / "kvasir_threshold_results.csv"
    save_json(metrics_json_path, payload)
    write_metrics_csv(metrics_csv_path, metrics_by_threshold)
    write_threshold_results_csv(threshold_csv_path, threshold_metrics)

    print(f"checkpoint={ckpt_path}")
    print(f"model_name={cfg['model_name']}")
    print(f"num_samples={len(rows)} img_size={cfg['image_size']} imagenet_norm={cfg['imagenet_norm']}")
    for threshold, metrics in threshold_metrics.items():
        print(
            f"threshold={threshold} Dice={metrics['Dice']:.4f} IoU={metrics['IoU']:.4f} "
            f"Precision={metrics['Precision']:.4f} Recall={metrics['Recall']:.4f} F2={metrics['F2']:.4f}"
        )
    print(f"saved_json={metrics_json_path}")
    print(f"saved_csv={metrics_csv_path}")
    print(f"saved_threshold_csv={threshold_csv_path}")
    print(f"saved_visuals={visuals_dir}")


if __name__ == "__main__":
    main()

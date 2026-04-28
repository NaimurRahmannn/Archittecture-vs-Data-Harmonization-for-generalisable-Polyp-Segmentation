from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = (
    "Dice",
    "IoU",
    "Precision",
    "Recall",
    "F2",
    "mean_pred_area",
    "mean_gt_area",
    "area_ratio",
    "area_diff",
)
SUBSETS = ("overall", "small", "medium", "large")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed Kvasir evaluation outputs into paper-style mean/std summaries."
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seed list. Default: 0,1,2",
    )
    parser.add_argument(
        "--dir_template",
        default="outputs/kvasir_m2b_seed{seed}_valdice",
        help=(
            "Per-seed output directory template. Use {seed} as the placeholder. "
            "Example: outputs/kvasir_m2b_seed{seed}_seqval_constrained"
        ),
    )
    parser.add_argument(
        "--metrics_filename",
        default="kvasir_metrics.json",
        help="Metrics JSON filename inside each seed directory.",
    )
    parser.add_argument(
        "--label",
        default="m2b_kvasir",
        help="Label written into the aggregated outputs.",
    )
    parser.add_argument(
        "--out_csv",
        default="outputs/kvasir_mean_std.csv",
        help="Output CSV path for selected-threshold subset metrics.",
    )
    parser.add_argument(
        "--out_threshold_csv",
        default="outputs/kvasir_threshold_mean_std.csv",
        help="Output CSV path for overall threshold-sweep aggregation.",
    )
    parser.add_argument(
        "--out_json",
        default="outputs/kvasir_mean_std.json",
        help="Output JSON path for the full aggregated payload.",
    )
    return parser.parse_args()


def parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_std(values: list[float]) -> tuple[float, float]:
    mean_value = statistics.mean(values)
    std_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return float(mean_value), float(std_value)


def collect_runs(seeds: list[str], dir_template: str, metrics_filename: str) -> tuple[list[dict], list[str]]:
    runs = []
    missing = []
    for seed in seeds:
        run_dir = Path(dir_template.format(seed=seed))
        metrics_path = run_dir / metrics_filename
        if not metrics_path.exists():
            missing.append(str(metrics_path))
            continue
        payload = load_json(metrics_path)
        runs.append(
            {
                "seed": seed,
                "run_dir": str(run_dir.resolve()),
                "metrics_path": str(metrics_path.resolve()),
                "payload": payload,
            }
        )
    return runs, missing


def aggregate_selected_threshold(runs: list[dict]) -> list[dict]:
    rows = []
    first_payload = runs[0]["payload"]
    report_threshold = first_payload["report_threshold"]

    for subset in SUBSETS:
        metric_lists = {metric: [] for metric in METRICS}
        seeds_used = []
        for run in runs:
            payload = run["payload"]
            selected = payload["selected_threshold_metrics"]
            if subset not in selected:
                continue
            seeds_used.append(str(run["seed"]))
            for metric in METRICS:
                metric_lists[metric].append(float(selected[subset][metric]))

        if not seeds_used:
            continue

        row = {
            "subset": subset,
            "report_threshold": report_threshold,
            "n_seeds": len(seeds_used),
            "seeds_used": ",".join(seeds_used),
        }
        for metric in METRICS:
            mean_value, std_value = mean_std(metric_lists[metric])
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_mean_std"] = f"{mean_value:.4f}+-{std_value:.4f}"
        rows.append(row)

    return rows


def aggregate_threshold_sweep(runs: list[dict]) -> list[dict]:
    first_payload = runs[0]["payload"]
    threshold_keys = list(first_payload["overall_by_threshold"].keys())
    rows = []

    for threshold in threshold_keys:
        metric_lists = {metric: [] for metric in METRICS}
        seeds_used = []
        for run in runs:
            payload = run["payload"]
            overall_by_threshold = payload["overall_by_threshold"]
            if threshold not in overall_by_threshold:
                continue
            seeds_used.append(str(run["seed"]))
            for metric in METRICS:
                metric_lists[metric].append(float(overall_by_threshold[threshold][metric]))

        if not seeds_used:
            continue

        row = {
            "threshold": threshold,
            "subset": "overall",
            "n_seeds": len(seeds_used),
            "seeds_used": ",".join(seeds_used),
        }
        for metric in METRICS:
            mean_value, std_value = mean_std(metric_lists[metric])
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_mean_std"] = f"{mean_value:.4f}+-{std_value:.4f}"
        rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict], leading_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = []
    for metric in METRICS:
        metric_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_mean_std"])
    fieldnames = leading_fields + metric_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main():
    args = parse_args()
    seeds = parse_list(args.seeds)
    if not seeds:
        raise SystemExit("No seeds provided.")

    runs, missing = collect_runs(seeds, args.dir_template, args.metrics_filename)
    if not runs:
        raise SystemExit(f"No metrics JSON files found. Missing examples: {missing[:3]}")

    selected_rows = aggregate_selected_threshold(runs)
    threshold_rows = aggregate_threshold_sweep(runs)
    first_payload = runs[0]["payload"]

    summary_payload = {
        "label": args.label,
        "dir_template": args.dir_template,
        "metrics_filename": args.metrics_filename,
        "n_runs": len(runs),
        "requested_seeds": seeds,
        "used_seeds": [run["seed"] for run in runs],
        "missing_metrics_paths": missing,
        "report_threshold": first_payload["report_threshold"],
        "img_size": first_payload.get("img_size"),
        "model_name": first_payload.get("model_name"),
        "model_alias": first_payload.get("model_alias"),
        "selected_threshold_summary": selected_rows,
        "threshold_sweep_summary": threshold_rows,
        "source_runs": [
            {
                "seed": run["seed"],
                "run_dir": run["run_dir"],
                "metrics_path": run["metrics_path"],
                "checkpoint": run["payload"].get("checkpoint"),
            }
            for run in runs
        ],
    }

    write_csv(
        Path(args.out_csv),
        selected_rows,
        leading_fields=["subset", "report_threshold", "n_seeds", "seeds_used"],
    )
    write_csv(
        Path(args.out_threshold_csv),
        threshold_rows,
        leading_fields=["threshold", "subset", "n_seeds", "seeds_used"],
    )
    save_json(Path(args.out_json), summary_payload)

    print(f"saved_selected_csv={Path(args.out_csv).resolve()}")
    print(f"saved_threshold_csv={Path(args.out_threshold_csv).resolve()}")
    print(f"saved_json={Path(args.out_json).resolve()}")
    print(f"used_seeds={','.join(summary_payload['used_seeds'])}")
    for row in selected_rows:
        print(
            f"subset={row['subset']} "
            f"Dice={row['Dice_mean_std']} "
            f"F2={row['F2_mean_std']} "
            f"area_ratio={row['area_ratio_mean_std']}"
        )


if __name__ == "__main__":
    main()

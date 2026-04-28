import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate M2b ImageNet multi-seed metrics to mean/std table.")
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seed list (default: 0,1,2).",
    )
    parser.add_argument(
        "--strategies",
        default="best_by_valdice,best_by_seqvalf2_constrained",
        help="Comma-separated checkpoint strategies to include.",
    )
    parser.add_argument(
        "--runs_root",
        default="E:/Thesis_Code/runs",
        help="Root folder that contains m2b_imagenet_seed*/ folders.",
    )
    parser.add_argument(
        "--run_prefix",
        default="m2b_imagenet_seed",
        help="Run directory prefix. Per-seed path is <runs_root>/<run_prefix><seed>.",
    )
    parser.add_argument(
        "--model_name",
        default="m2b_convnext_xattn_imagenetnorm",
        help="Model name value to write in the output CSV.",
    )
    parser.add_argument(
        "--out_csv",
        default="E:/Thesis_Code/results/tables/m2b_imagenet_mean_std.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def parse_list(raw):
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values):
    if not values:
        return 0.0, 0.0
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if len(values) > 1 else 0.0
    return float(mean_val), float(std_val)


def main():
    args = parse_args()
    seeds = [int(x) for x in parse_list(args.seeds)]
    strategies = parse_list(args.strategies)
    groups = ("data2", "data3", "data4")
    runs_root = Path(args.runs_root)
    run_prefix = args.run_prefix
    model_name = args.model_name

    rows = []
    for strategy in strategies:
        for group in groups:
            dice_values = []
            f2_values = []
            used_seeds = []
            for seed in seeds:
                run_dir = runs_root / f"{run_prefix}{seed}"
                metrics_path = run_dir / f"metrics_{strategy}_{group}.json"
                if not metrics_path.exists():
                    continue
                payload = load_json(metrics_path)
                dice_values.append(float(payload["Dice"]))
                f2_values.append(float(payload["F2"]))
                used_seeds.append(str(seed))

            if not used_seeds:
                continue

            dice_mean, dice_std = mean_std(dice_values)
            f2_mean, f2_std = mean_std(f2_values)
            rows.append(
                {
                    "model_name": model_name,
                    "strategy": strategy,
                    "group": group,
                    "n_seeds": len(used_seeds),
                    "seeds_used": ",".join(used_seeds),
                    "Dice_mean": dice_mean,
                    "Dice_std": dice_std,
                    "Dice_mean_std": f"{dice_mean:.4f}+-{dice_std:.4f}",
                    "F2_mean": f2_mean,
                    "F2_std": f2_std,
                    "F2_mean_std": f"{f2_mean:.4f}+-{f2_std:.4f}",
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "strategy",
        "group",
        "n_seeds",
        "seeds_used",
        "Dice_mean",
        "Dice_std",
        "Dice_mean_std",
        "F2_mean",
        "F2_std",
        "F2_mean_std",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved={out_csv}")
    for row in rows:
        print(
            f"{row['strategy']} {row['group']}: "
            f"Dice {row['Dice_mean_std']}, F2 {row['F2_mean_std']} "
            f"(n={row['n_seeds']}, seeds={row['seeds_used']})"
        )


if __name__ == "__main__":
    main()

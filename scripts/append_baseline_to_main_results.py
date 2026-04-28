import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Append one model's group metrics to main results CSV.")
    parser.add_argument("--model_name", default="DeepLabV3+")
    parser.add_argument("--run_dir", default=r"E:\Thesis_Code\runs\m3_deeplabv3plus")
    parser.add_argument("--csv_path", default=r"E:\Thesis_Code\results\tables\main_results_with_baseline.csv")
    return parser.parse_args()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    csv_path = Path(args.csv_path)

    data2 = read_json(run_dir / "metrics_data2.json")
    data3 = read_json(run_dir / "metrics_data3.json")
    data4 = read_json(run_dir / "metrics_data4.json")
    metrics_map = {"data2": data2, "data3": data3, "data4": data4}

    drop_f2 = float(data4["F2"]) - float(data3["F2"])
    drop_dice = float(data4["Dice"]) - float(data3["Dice"])
    area_ratio_gap = float(data4["area_ratio"]) - float(data3["area_ratio"])

    new_rows = []
    for group in ("data2", "data3", "data4"):
        g = metrics_map[group]
        row = {
            "model": args.model_name,
            "group": group,
            "Dice": g["Dice"],
            "IoU": g["IoU"],
            "Precision": g["Precision"],
            "Recall": g["Recall"],
            "F2": g["F2"],
            "mean_pred_area": g["mean_pred_area"],
            "mean_gt_area": g["mean_gt_area"],
            "area_ratio": g["area_ratio"],
            "area_diff": g["area_diff"],
            "mean_pred_area_per_image": g.get("mean_pred_area_per_image", g["mean_pred_area"]),
            "mean_gt_area_per_image": g.get("mean_gt_area_per_image", g["mean_gt_area"]),
            "drop_F2_data4_minus_data3": drop_f2,
            "drop_Dice_data4_minus_data3": drop_dice,
            "area_ratio_gap_data4_minus_data3": area_ratio_gap,
        }
        new_rows.append(row)

    fieldnames = [
        "model",
        "group",
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
        "drop_F2_data4_minus_data3",
        "drop_Dice_data4_minus_data3",
        "area_ratio_gap_data4_minus_data3",
    ]

    existing = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        existing = [r for r in existing if not (r.get("model") == args.model_name and r.get("group") in metrics_map)]

    merged = existing + new_rows
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    print(f"saved={csv_path}")


if __name__ == "__main__":
    main()

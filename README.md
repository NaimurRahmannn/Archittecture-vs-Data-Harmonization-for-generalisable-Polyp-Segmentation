# Architecture vs Data-Harmonization for Generalisable Polyp Segmentation

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-Polyp%20Segmentation-0A7E8C)
![Evaluation](https://img.shields.io/badge/Evaluation-Centre%20%2B%20Sequence%20Shift-1F6FEB)
![Status](https://img.shields.io/badge/Status-Research%20Code-orange)

Codebase for a thesis study on robust polyp segmentation under centre shift and sequence shift.

Primary question:
Does model architecture (especially cross-attention skip fusion) improve out-of-distribution generalisation more than data harmonization-oriented selection strategies such as SeqVal?

Quick links: [Model Variants](#model-variants) | [Dataset Protocol](#dataset-and-evaluation-protocol) | [Setup](#setup) | [Quickstart](#quickstart) | [Metrics](#reported-metrics)

## Table of Contents

- [Model Variants](#model-variants)
- [Dataset and Evaluation Protocol](#dataset-and-evaluation-protocol)
- [Split Definition](#split-definition)
- [Optional SeqVal](#optional-seqval)
- [Setup](#setup)
- [Quickstart](#quickstart)
- [Reported Metrics](#reported-metrics)
- [Notes](#notes)

## Model Variants

| Model | Description |
|---|---|
| M0 | U-Net baseline |
| M1 | ConvNeXt-Tiny U-Net with concatenation skip fusion |
| M2 | Naive cross-attention skip fusion |
| M2b (key) | Gated cross-attention plus preserved skip path with learnable alpha and beta |
| Baseline | DeepLabV3 (ResNet50, torchvision) |

## Dataset and Evaluation Protocol

PolypGen / EndoCV-style setup.

Expected dataset root (example):

```text
E:/Thesis_Dataset/PolypGen2021_MultiCenterData_v3/PolypGen2021_MultiCenterData_v3
```

### Split Definition

| Group | Shift Type | Definition |
|---|---|---|
| Train/Val | In-domain | Single frames from C1-C5 (Val is a hold-out from the training pool) |
| data2 | Centre shift | C6 single frames |
| data3 | Sequence shift | Positive sequence frames (non-C6 / centre-unknown) |
| data4 | Centre plus sequence shift | C6 positive sequence frames |

### Optional SeqVal

Hold out 10% from data3 (never used for training) for robustness-aware checkpoint selection.

## Setup

```bash
pip install torch torchvision timm pillow numpy
```

Use the CUDA build of PyTorch if running on GPU.

## Quickstart

### 1. Build the base manifest

```bash
python scripts/make_manifest_polypgen_centre_seq.py \
	--root "E:/.../PolypGen2021_MultiCenterData_v3" \
	--out_csv "splits/manifest.csv" \
	--out_json "splits/summary.json" \
	--val_ratio 0.10 \
	--seed 0
```

### 2. (Optional) Add SeqVal partition

```bash
python scripts/create_seqval_manifest.py \
	--in_manifest "splits/manifest.csv" \
	--out_manifest "splits/manifest_seqval.csv" \
	--seqval_ratio 0.10 \
	--seed 0
```

### 3. Train

Edit one config in configs/ and set:

- data_root
- manifest_csv
- run_dir

Then run:

```bash
python train.py --config configs/<your_config>.yaml
```

### 4. Evaluate on data2, data3, data4

```bash
python eval.py --ckpt runs/<run_name>/best_by_valdice.pt --group data2
python eval.py --ckpt runs/<run_name>/best_by_valdice.pt --group data3
python eval.py --ckpt runs/<run_name>/best_by_valdice.pt --group data4
```

If SeqVal is enabled, also compare these checkpoints:

- best_by_seqvalf2.pt
- best_by_seqvalf2_constrained.pt

## Reported Metrics

All metrics are computed per test group at threshold 0.5:

- Dice
- IoU
- Precision
- Recall
- F2
- area_ratio = mean_pred_area / mean_gt_area

The area_ratio metric is included to diagnose segmentation size bias.

## Notes

- This repository does not include the dataset. Please follow the dataset license and terms of use.
- Large checkpoint files (.pt) should not be committed directly to GitHub; use external storage or Git LFS.

## Kvasir External Test Workflow

This repository also supports Kvasir-SEG as an external test-only dataset for cross-dataset domain generalization evaluation.

The Kvasir integration is additive only:

- Training remains on PolypGen / EndoCV checkpoints only.
- Kvasir-SEG is used only at evaluation time.
- Existing training and evaluation entry points are unchanged.

### Added Modules

- `datasets/kvasir_dataset.py`
	Loads Kvasir-SEG from the standard `images/` and `masks/` folders, resizes image and mask together, supports optional ImageNet normalization, and returns `(image_tensor, mask_tensor, metadata_dict)`.
- `scripts/build_kvasir_manifest.py`
	Builds `kvasir_manifest.csv` with `image_path`, `mask_path`, `split`, `dataset`, and `source`.
- `evaluation/evaluate_kvasir.py`
	Loads an existing trained checkpoint and evaluates it on Kvasir-SEG only.

### Build The Kvasir Manifest

```powershell
.\.venv\Scripts\python.exe scripts\build_kvasir_manifest.py
```

Optional overrides:

```powershell
.\.venv\Scripts\python.exe scripts\build_kvasir_manifest.py --kvasir_root E:\Thesis_Dataset\kvasir-seg\Kvasir-SEG --output E:\Thesis_Code\kvasir_manifest.csv
```

### Evaluate A Trained Checkpoint On Kvasir

Example with M2b:

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate_kvasir.py --checkpoint runs\m2b_imagenet_seed0\best.pt --model M2b
```

Example with explicit config override:

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate_kvasir.py --checkpoint runs\m2b_imagenet_seed0\best.pt --model M2b --config configs\m2b_imagenet_seed0.yaml
```

### Resolution Ablation

Supported evaluation resolutions:

- `256`
- `352`
- `512`

Examples:

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate_kvasir.py --checkpoint runs\m2b_imagenet_seed0\best.pt --model M2b --img_size 256
.\.venv\Scripts\python.exe evaluation\evaluate_kvasir.py --checkpoint runs\m2b_imagenet_seed0\best.pt --model M2b --img_size 352
.\.venv\Scripts\python.exe evaluation\evaluate_kvasir.py --checkpoint runs\m2b_imagenet_seed0\best.pt --model M2b --img_size 512
```

### Metrics And Analysis

The Kvasir evaluator reports:

- `Dice`
- `IoU`
- `Precision`
- `Recall`
- `F2`
- `mean_pred_area`
- `mean_gt_area`
- `area_ratio`

It also includes:

- threshold sensitivity at `0.3`, `0.5`, and `0.7`
- worst-case qualitative overlays
- size-stratified results for `small`, `medium`, and `large` polyps based on GT mask area fraction

### Output Files

Running `evaluation/evaluate_kvasir.py` writes:

- `outputs/kvasir_metrics.json`
- `outputs/kvasir_metrics.csv`
- `outputs/kvasir_threshold_results.csv`
- `outputs/kvasir_visuals/`

The visual directory contains the top worst Dice cases and a `worst_cases.csv` index.

### Multi-Seed Aggregation

To summarise Kvasir results across seeds in paper-style `mean +- std` format, run:

```powershell
.\.venv\Scripts\python.exe scripts\aggregate_kvasir_mean_std.py --seeds 0,1,2 --dir_template 'outputs/kvasir_m2b_seed{seed}_valdice' --label m2b_kvasir_valdice --out_csv outputs\kvasir_m2b_valdice_mean_std.csv --out_threshold_csv outputs\kvasir_m2b_valdice_threshold_mean_std.csv --out_json outputs\kvasir_m2b_valdice_mean_std.json
```

For the robustness-analysis checkpoint choice:

```powershell
.\.venv\Scripts\python.exe scripts\aggregate_kvasir_mean_std.py --seeds 0,1,2 --dir_template 'outputs/kvasir_m2b_seed{seed}_seqval_constrained' --label m2b_kvasir_seqval_constrained --out_csv outputs\kvasir_m2b_seqval_constrained_mean_std.csv --out_threshold_csv outputs\kvasir_m2b_seqval_constrained_threshold_mean_std.csv --out_json outputs\kvasir_m2b_seqval_constrained_mean_std.json
```

### Integration Notes

- The evaluator reuses the existing checkpoint format with `model_state` and embedded `config`.
- It reuses the existing model factory in `pipeline_common.py`.
- Use `--config` if a checkpoint does not contain the config values you want to evaluate with.
- Use `--imagenet_norm` or `--no_imagenet_norm` to override normalization behavior when needed.
- No retraining is performed on Kvasir-SEG.

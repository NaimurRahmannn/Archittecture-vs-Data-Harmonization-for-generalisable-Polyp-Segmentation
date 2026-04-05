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



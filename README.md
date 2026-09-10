# Fabric/Textile Defect Inspection System (CNN)

ENEL4AI H2 — Artificial Intelligence — 2026
Project 3: Fabric/Textile Defect Inspection System

Group members: Hussain S., Fataki I., Hlangu L., Dube N.K.

## Overview

A prototype neural-network-based system for detecting weaving flaws, holes,
oil spots, and thread defects in fabric images, built with a CNN using
transfer learning (ResNet18).

The project originally planned to use the AITEX fabric defect dataset, but
switched to the TILDA fabric defect dataset (5-class, 64x64 patch version)
after determining a team member's laptop could not handle AITEX's larger
strip images. This is documented in `Task2_Report.docx`, along with the
class-imbalance handling this required.

## Repository structure

```
Models/            trained model weights
  resnet18_baseline.pth   v1: frozen backbone, final layer only, 10 epochs
  resnet18_v2.pth         v2: layer4 fine-tuned + more augmentation, 15 epochs
logs/               training logs, evaluation results, confusion matrices
scripts/            all code used to produce the above, in run order
Task2_Report.docx   full write-up: dataset, methodology, results, discussion
split_summary.csv   final train/test class counts
```

## Results summary

| Metric | v1 (baseline) | v2 (improved) |
|---|---|---|
| Overall test accuracy | 81.5% | 89.8% |
| Macro-average precision | 0.413 | 0.621 |
| Macro-average recall | 0.607 | 0.737 |
| Macro-average F1-score | 0.460 | 0.654 |

Full per-class breakdown and confusion matrices are in `logs/` and in
`Task2_Report.docx`.

## Reproducing these results

1. Set up the environment:
   ```
   pip install -r scripts/requirements.txt
   ```
2. Place the raw TILDA data (`DataSet/`) and the existing `processed_data/test/`
   split alongside these scripts, then run, in order:
   ```
   python scripts/build_train_split.py
   python scripts/train_resnet18.py          # produces Models/resnet18_baseline.pth
   python scripts/train_resnet18_v2.py        # produces Models/resnet18_v2.pth
   python scripts/evaluate_resnet18.py
   python scripts/evaluate_resnet18_v2.py
   python scripts/make_training_curves.py
   ```

All scripts use a fixed random seed (42) for reproducibility.

## Status

- [x] Dataset acquisition, cleaning, and class-imbalance handling
- [x] Baseline model trained and evaluated
- [x] Improved model trained and evaluated
- [ ] Working prototype (simulated fabric feed + dashboard + defect log) — in progress

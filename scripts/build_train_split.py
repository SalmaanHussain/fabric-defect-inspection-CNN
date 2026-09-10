"""
Rebuilds processed_data/train/ so it has all 5 classes (it was previously
missing hole, objects, oil spot, and thread error).

Rule: a train image is any raw DataSet image that is NOT already used in
processed_data/test/ (verified as a true subset before writing this script),
so there is no overlap between train and test.

Imbalance fix: the "good" class is capped at MAX_GOOD images instead of
using all ~19,700 leftover images, cutting the imbalance from ~69:1 down to
~10:1 against the rarest class, and reducing total training data size.

Run with a fixed random seed so the split is reproducible (same result every
time this script is run), which matters for the report.
"""
import os
import random
import shutil
import csv

random.seed(42)

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "DataSet")
TEST_DIR = os.path.join(ROOT, "processed_data", "test")
TRAIN_DIR = os.path.join(ROOT, "processed_data", "train")

CLASSES = ["good", "hole", "objects", "oil spot", "thread error"]
MAX_GOOD = 3000

summary_rows = []

for cls in CLASSES:
    raw_dir = os.path.join(RAW_DIR, cls)
    test_dir = os.path.join(TEST_DIR, cls)
    train_dir = os.path.join(TRAIN_DIR, cls)

    # wipe any pre-existing train files for this class so the result is a
    # clean, fully-reproducible split rather than a mix of old + new files
    if os.path.isdir(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir, exist_ok=True)

    raw_files = set(os.listdir(raw_dir))
    test_files = set(os.listdir(test_dir))

    # safety check: test must be a genuine subset of raw before we proceed
    assert test_files.issubset(raw_files), f"test files not a subset of raw for class '{cls}'"

    candidates = sorted(raw_files - test_files)  # everything not already in test

    if cls == "good":
        random.shuffle(candidates)
        chosen = candidates[:MAX_GOOD]
    else:
        chosen = candidates  # keep every rare-class image, can't afford to lose any

    copied = 0
    for fname in chosen:
        src = os.path.join(raw_dir, fname)
        dst = os.path.join(train_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        copied += 1

    summary_rows.append({
        "class": cls,
        "raw_total": len(raw_files),
        "test_count": len(test_files),
        "available_for_train": len(candidates),
        "train_count": copied,
    })

print(f"{'class':14s} {'raw':>7s} {'test':>6s} {'avail':>7s} {'train':>7s}")
for r in summary_rows:
    print(f"{r['class']:14s} {r['raw_total']:7d} {r['test_count']:6d} {r['available_for_train']:7d} {r['train_count']:7d}")

with open(os.path.join(ROOT, "processed_data", "split_summary.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["class", "raw_total", "test_count", "available_for_train", "train_count"])
    writer.writeheader()
    writer.writerows(summary_rows)

print("\nDone. Summary saved to processed_data/split_summary.csv")

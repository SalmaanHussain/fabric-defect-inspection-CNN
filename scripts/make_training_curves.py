"""Plots train/val accuracy and loss curves for v1 vs v2 from the real training logs."""
import csv
import os
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(ROOT, "logs")


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows


v1 = load(os.path.join(LOGS_DIR, "training_log.csv"))
v2 = load(os.path.join(LOGS_DIR, "training_log_v2.csv"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.plot([r["epoch"] for r in v1], [r["train_acc"] for r in v1], "o-", color="tab:blue", label="v1 train")
ax.plot([r["epoch"] for r in v1], [r["val_acc"] for r in v1], "o--", color="tab:blue", alpha=0.5, label="v1 val")
ax.plot([r["epoch"] for r in v2], [r["train_acc"] for r in v2], "s-", color="tab:orange", label="v2 train")
ax.plot([r["epoch"] for r in v2], [r["val_acc"] for r in v2], "s--", color="tab:orange", alpha=0.5, label="v2 val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Training / Validation Accuracy")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot([r["epoch"] for r in v1], [r["train_loss"] for r in v1], "o-", color="tab:blue", label="v1 train")
ax.plot([r["epoch"] for r in v1], [r["val_loss"] for r in v1], "o--", color="tab:blue", alpha=0.5, label="v1 val")
ax.plot([r["epoch"] for r in v2], [r["train_loss"] for r in v2], "s-", color="tab:orange", label="v2 train")
ax.plot([r["epoch"] for r in v2], [r["val_loss"] for r in v2], "s--", color="tab:orange", alpha=0.5, label="v2 val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training / Validation Loss")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
out_path = os.path.join(LOGS_DIR, "training_curves_v1_vs_v2.png")
fig.savefig(out_path, dpi=150)
print("Saved:", out_path)

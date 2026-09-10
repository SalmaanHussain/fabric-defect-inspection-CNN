"""
Evaluates the trained baseline ResNet18 model (Models/resnet18_baseline.pth)
on the held-out processed_data/test set (never used during training or
validation), and produces the metrics Task 2 asks for: accuracy, precision,
recall, F1-score, and a confusion matrix.

Outputs:
  - logs/evaluation_report.txt   (accuracy + per-class precision/recall/F1)
  - logs/confusion_matrix.png    (visual confusion matrix)
  - logs/test_predictions.csv    (per-image prediction, for spot-checking)
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(ROOT, "processed_data", "test")
MODEL_PATH = os.path.join(ROOT, "Models", "resnet18_baseline.pth")
LOGS_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_set = datasets.ImageFolder(TEST_DIR, transform=eval_transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
class_names = test_set.classes
print(f"Test set: {len(test_set)} images across {len(class_names)} classes: {class_names}")

# rebuild the exact same architecture used in training, then load the
# trained weights (frozen backbone + a 5-class final layer)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

all_preds, all_labels, all_paths = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

# recover file paths in the same order ImageFolder iterated them
all_paths = [p for p, _ in test_set.samples]

acc = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
cm = confusion_matrix(all_labels, all_preds)

print(f"\nOverall test accuracy: {acc:.4f}\n")
print(report)

# save text report
report_path = os.path.join(LOGS_DIR, "evaluation_report.txt")
with open(report_path, "w") as f:
    f.write(f"Test set size: {len(test_set)} images\n")
    f.write(f"Classes: {class_names}\n\n")
    f.write(f"Overall accuracy: {acc:.4f}\n\n")
    f.write("Per-class precision / recall / F1-score:\n")
    f.write(report)
    f.write("\nConfusion matrix (rows = true class, columns = predicted class):\n")
    f.write(f"{'':14s}" + "".join(f"{c[:10]:>12s}" for c in class_names) + "\n")
    for i, row in enumerate(cm):
        f.write(f"{class_names[i]:14s}" + "".join(f"{v:12d}" for v in row) + "\n")

# save confusion matrix as an image
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title(f"Confusion Matrix — ResNet18 Baseline (test acc={acc:.3f})")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
cm_path = os.path.join(LOGS_DIR, "confusion_matrix.png")
fig.savefig(cm_path, dpi=150)

# save per-image predictions for spot-checking / error analysis
import csv
pred_csv_path = os.path.join(LOGS_DIR, "test_predictions.csv")
with open(pred_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "true_label", "predicted_label", "correct"])
    for path, true_idx, pred_idx in zip(all_paths, all_labels, all_preds):
        writer.writerow([path, class_names[true_idx], class_names[pred_idx], true_idx == pred_idx])

print(f"\nSaved: {report_path}")
print(f"Saved: {cm_path}")
print(f"Saved: {pred_csv_path}")

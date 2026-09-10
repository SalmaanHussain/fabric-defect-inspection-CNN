"""
Evaluates the improved v2 model (Models/resnet18_v2.pth) on the same
held-out processed_data/test set used for the v1 baseline, so the two can
be compared directly. Must use the same 128x128 image size that v2 was
trained with (v1 used 224x224).

Outputs:
  - logs/evaluation_report_v2.txt
  - logs/confusion_matrix_v2.png
  - logs/test_predictions_v2.csv
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import csv

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(ROOT, "processed_data", "test")
MODEL_PATH = os.path.join(ROOT, "Models", "resnet18_v2.pth")
LOGS_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = 128  # must match training size used for resnet18_v2.pth
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

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

all_paths = [p for p, _ in test_set.samples]

acc = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
cm = confusion_matrix(all_labels, all_preds)

print(f"\nOverall test accuracy: {acc:.4f}\n")
print(report)

report_path = os.path.join(LOGS_DIR, "evaluation_report_v2.txt")
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

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title(f"Confusion Matrix — ResNet18 v2 (test acc={acc:.3f})")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
cm_path = os.path.join(LOGS_DIR, "confusion_matrix_v2.png")
fig.savefig(cm_path, dpi=150)

pred_csv_path = os.path.join(LOGS_DIR, "test_predictions_v2.csv")
with open(pred_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "true_label", "predicted_label", "correct"])
    for path, true_idx, pred_idx in zip(all_paths, all_labels, all_preds):
        writer.writerow([path, class_names[true_idx], class_names[pred_idx], true_idx == pred_idx])

print(f"\nSaved: {report_path}")
print(f"Saved: {cm_path}")
print(f"Saved: {pred_csv_path}")

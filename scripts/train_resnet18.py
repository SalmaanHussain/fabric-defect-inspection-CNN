"""
Trains a baseline ResNet18 fabric-defect classifier on the TILDA data
(processed_data/train, 5 classes: good, hole, objects, oil spot, thread error).

Approach (transfer learning, matching the project brief):
  - Start from a ResNet18 pretrained on ImageNet.
  - Freeze the pretrained backbone and only train a new final layer for our
    5 classes. This is the standard, fastest baseline approach for a small
    dataset and a first submission.
  - Use class-weighted loss to correct the remaining ~10.5:1 imbalance left
    after capping "good" during the Phase A data fix.
  - Carve 15% off the train folder as a validation set (stratified), so we
    can watch for overfitting during training. The processed_data/test
    folder is NOT touched here — it stays held out for the separate
    evaluation script.

Outputs:
  - Models/resnet18_baseline.pth   (best checkpoint, by validation accuracy)
  - logs/training_log.csv          (per-epoch train/val loss + accuracy)
"""
import os
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(ROOT, "processed_data", "train")
MODELS_DIR = os.path.join(ROOT, "Models")
LOGS_DIR = os.path.join(ROOT, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

SEED = 42
VAL_FRACTION = 0.15
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
IMG_SIZE = 224  # ResNet18 pretrained weights expect ImageNet-scale input

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# two ImageFolder instances over the same directory (same sorted file order),
# one with training-time augmentation, one without, so the val subset never
# gets the random-flip augmentation meant for training data only
full_train_aug = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
full_train_plain = datasets.ImageFolder(TRAIN_DIR, transform=eval_transform)

class_names = full_train_aug.classes
print(f"Classes ({len(class_names)}): {class_names}")

targets = [label for _, label in full_train_aug.samples]
train_idx, val_idx = train_test_split(
    range(len(targets)), test_size=VAL_FRACTION, stratify=targets, random_state=SEED
)

train_set = Subset(full_train_aug, train_idx)
val_set = Subset(full_train_plain, val_idx)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# class weights: inverse of class frequency in the training split, so rare
# classes (hole, oil spot, thread error) count for more in the loss
counts_per_class = [0] * len(class_names)
for idx in train_idx:
    counts_per_class[targets[idx]] += 1
print("Training images per class:", dict(zip(class_names, counts_per_class)))

total = sum(counts_per_class)
class_weights = torch.tensor(
    [total / (len(class_names) * c) for c in counts_per_class], dtype=torch.float32
).to(device)
print("Class weights (loss):", dict(zip(class_names, [round(w, 3) for w in class_weights.tolist()])))

# model: pretrained ResNet18, backbone frozen, new final layer for 5 classes
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # new layer -> trainable by default
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)


def run_epoch(loader, train_mode):
    model.train() if train_mode else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            n += images.size(0)
    return total_loss / n, correct / n


log_rows = []
best_val_acc = 0.0
best_path = os.path.join(MODELS_DIR, "resnet18_baseline.pth")

print(f"\nStarting training: {EPOCHS} epochs, {len(train_set)} train / {len(val_set)} val images\n")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = run_epoch(train_loader, train_mode=True)
    val_loss, val_acc = run_epoch(val_loader, train_mode=False)
    dt = time.time() - t0

    print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
          f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt:.1f}s")

    log_rows.append({
        "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
        "val_loss": val_loss, "val_acc": val_acc, "seconds": round(dt, 1),
    })

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_path)
        print(f"  -> new best val_acc={val_acc:.4f}, saved to {best_path}")

log_path = os.path.join(LOGS_DIR, "training_log.csv")
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])
    writer.writeheader()
    writer.writerows(log_rows)

print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
print(f"Best model saved to: {best_path}")
print(f"Training log saved to: {log_path}")

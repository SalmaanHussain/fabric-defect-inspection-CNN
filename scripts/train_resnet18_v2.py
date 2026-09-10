"""
Improved version of train_resnet18.py — same data and same class-weighted-loss
approach, with three changes aimed at raising accuracy on the rare defect
classes (see logs/training_log.csv from the v1 baseline for comparison):

  1. Unfreeze the last ResNet block (layer4) instead of only the final
     layer, so the pretrained features can adapt to fabric texture rather
     than staying fixed at generic-photo features. Trained with a smaller
     learning rate than the new final layer, since it's fine-tuning
     already-good weights rather than learning from scratch.
  2. Train at 128x128 instead of 224x224. Our raw patches are only 64x64,
     so 224x224 added computation without adding real information; this
     mainly buys back the training-time budget spent on change #1.
  3. Add rotation + brightness/contrast jitter to the training augmentation
     (on top of the existing horizontal flip), to give the model more
     varied examples of the rare classes without needing more raw images.

Outputs (kept separate from the v1 baseline so both can be compared in the
report):
  - Models/resnet18_v2.pth
  - logs/training_log_v2.csv
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
EPOCHS = 15
HEAD_LR = 1e-3       # learning rate for the new final layer (learning from scratch)
BACKBONE_LR = 1e-4   # smaller learning rate for the unfrozen pretrained block (fine-tuning)
IMG_SIZE = 128        # change #2: smaller than v1's 224, closer to the native 64x64 patch size

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# change #3: rotation + colour jitter added on top of the horizontal flip
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

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

counts_per_class = [0] * len(class_names)
for idx in train_idx:
    counts_per_class[targets[idx]] += 1
print("Training images per class:", dict(zip(class_names, counts_per_class)))

total = sum(counts_per_class)
class_weights = torch.tensor(
    [total / (len(class_names) * c) for c in counts_per_class], dtype=torch.float32
).to(device)
print("Class weights (loss):", dict(zip(class_names, [round(w, 3) for w in class_weights.tolist()])))

# change #1: pretrained ResNet18, only layer4 (last block) + fc unfrozen
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # trainable by default
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam([
    {"params": model.fc.parameters(), "lr": HEAD_LR},
    {"params": model.layer4.parameters(), "lr": BACKBONE_LR},
])

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total_params:,}")


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
best_path = os.path.join(MODELS_DIR, "resnet18_v2.pth")

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

log_path = os.path.join(LOGS_DIR, "training_log_v2.csv")
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])
    writer.writeheader()
    writer.writerows(log_rows)

print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
print(f"Best model saved to: {best_path}")
print(f"Training log saved to: {log_path}")

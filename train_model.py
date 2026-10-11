"""
=============================================================
 Driver Drowsiness Detection - Model Training Script
=============================================================
 This script trains a CNN (Convolutional Neural Network) model
 to classify images into 4 categories:
   1. Closed  -- Eyes are closed
   2. Open    -- Eyes are open
   3. yawn    -- Driver is yawning
   4. no_yawn -- Driver is not yawning

 Dataset Structure Expected:
   archive (1)/dataset_new/
   +-- train/
   |   +-- Closed/    (617 images)
   |   +-- Open/      (617 images)
   |   +-- no_yawn/   (616 images)
   |   +-- yawn/      (617 images)
   +-- test/
       +-- Closed/    (109 images)
       +-- Open/      (109 images)
       +-- no_yawn/   (109 images)
       +-- yawn/      (106 images)

 Output:
   - driver_drowsiness_model.pth   (trained model weights)
   - class_labels.json             (class name mapping)
   - training_results.png          (accuracy/loss plots)

 Usage:
   python train_model.py
=============================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (avoids Tkinter crash)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =====================================================
# 1. CONFIGURATION -- Set paths and parameters
# =====================================================

# Dataset paths (relative to this script's location)
DATASET_DIR = os.path.join("archive (1)", "dataset_new")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Image and model parameters
IMG_SIZE = 64           # All images resized to 64x64 pixels (fast on CPU)
BATCH_SIZE = 16         # Number of images per training batch
EPOCHS = 30             # Maximum training cycles
LEARNING_RATE = 0.001   # Adam optimizer learning rate
NUM_CLASSES = 4         # Closed, Open, yawn, no_yawn

# Where to save the trained model
MODEL_PATH = "driver_drowsiness_model.pth"
LABELS_PATH = "class_labels.json"

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# 2. VALIDATE DATASET -- Make sure folders exist
# =====================================================

print("=" * 60)
print("  DRIVER DROWSINESS DETECTION -- MODEL TRAINING")
print("=" * 60)
print(f"\n   Device: {DEVICE}")

# Check that dataset directories actually exist before proceeding
if not os.path.isdir(TRAIN_DIR):
    print(f"\n   ERROR: Training directory not found: {TRAIN_DIR}")
    print("   Make sure the dataset folder 'archive (1)/dataset_new/' exists.")
    sys.exit(1)

if not os.path.isdir(TEST_DIR):
    print(f"\n   ERROR: Test directory not found: {TEST_DIR}")
    print("   Make sure the dataset folder 'archive (1)/dataset_new/' exists.")
    sys.exit(1)

# Verify all 4 class folders exist in train and test
expected_classes = ['Closed', 'Open', 'no_yawn', 'yawn']
for cls in expected_classes:
    train_cls_path = os.path.join(TRAIN_DIR, cls)
    test_cls_path = os.path.join(TEST_DIR, cls)
    if not os.path.isdir(train_cls_path):
        print(f"\n   ERROR: Missing training class folder: {train_cls_path}")
        sys.exit(1)
    if not os.path.isdir(test_cls_path):
        print(f"\n   ERROR: Missing test class folder: {test_cls_path}")
        sys.exit(1)

print("   Dataset directories verified successfully.")


# =====================================================
# 3. DATA PREPARATION -- Load and augment images
# =====================================================

# --- Training data transforms (with augmentation) ---
# Creates random variations of each image so the model
# can generalize better and avoid overfitting.
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),       # Resize all images
    transforms.RandomHorizontalFlip(),             # Randomly flip horizontally
    transforms.RandomRotation(20),                 # Randomly rotate +/-20 deg
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1                    # Random color changes
    ),
    transforms.RandomAffine(
        degrees=0, translate=(0.1, 0.1),
        shear=10                                   # Random shift and shear
    ),
    transforms.ToTensor(),                         # Convert to tensor (0-1)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],                # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

# --- Test data transforms (no augmentation) ---
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load datasets from folders
print("\n[1/6] Loading training data...")
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
print(f"       Found {len(train_dataset)} training images")

print("[2/6] Loading test data...")
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
print(f"       Found {len(test_dataset)} test images")

# Create data loaders (batches images for training)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# Get class names (sorted alphabetically by ImageFolder)
class_names = train_dataset.classes  # ['Closed', 'Open', 'no_yawn', 'yawn']
print(f"\n   Class labels : {class_names}")
print(f"   Class mapping: {train_dataset.class_to_idx}")

# Sanity check
if len(train_dataset) == 0:
    print("\n   ERROR: No training images found!")
    sys.exit(1)


# =====================================================
# 4. BUILD THE CNN MODEL
# =====================================================

print("\n[3/6] Building CNN model...")


class DrowsinessCNN(nn.Module):
    """
    Custom CNN for drowsiness detection.
    4 convolutional blocks followed by fully connected layers.
    """

    def __init__(self, num_classes=4):
        super(DrowsinessCNN, self).__init__()

        # ---- Convolutional Block 1 ----
        # 32 filters of 3x3, learns basic edges and textures
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)       # 145 -> 72
        )

        # ---- Convolutional Block 2 ----
        # 64 filters, learns more complex patterns
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)       # 72 -> 36
        )

        # ---- Convolutional Block 3 ----
        # 128 filters, learns high-level features
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)       # 36 -> 18
        )

        # ---- Convolutional Block 4 ----
        # 256 filters, captures fine details
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # ---- Classification Layers ----
        # AdaptiveAvgPool2d makes it work with any input size
        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # Always outputs 2x2
        flat_size = 256 * 2 * 2  # = 1024

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),         # 50% dropout to prevent overfitting
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),         # 30% dropout
            nn.Linear(128, num_classes)  # Output: raw scores for 4 classes
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# Create model and move to device (GPU/CPU)
model = DrowsinessCNN(num_classes=NUM_CLASSES).to(DEVICE)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total parameters    : {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")


# =====================================================
# 5. COMPILE -- Set loss function and optimizer
# =====================================================

print("\n[4/6] Setting up optimizer...")

criterion = nn.CrossEntropyLoss()    # Loss function for multi-class
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler -- reduce LR when accuracy plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5
)

print("   Optimizer : Adam")
print(f"   LR        : {LEARNING_RATE}")
print("   Loss      : CrossEntropyLoss")


# =====================================================
# 6. TRAINING AND EVALUATION FUNCTIONS
# =====================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch and return avg loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f"      Batch {batch_idx+1}/{len(loader)}", flush=True)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Evaluate the model on a dataset and return avg loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed for evaluation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# =====================================================
# 7. TRAIN THE MODEL
# =====================================================

print(f"\n[5/6] Starting training...")
print(f"   Epochs     : {EPOCHS} (max)")
print(f"   Batch size : {BATCH_SIZE}")
print(f"   Image size : {IMG_SIZE}x{IMG_SIZE}")
print("-" * 60)

# Track history for plotting
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

best_val_acc = 0.0
patience_counter = 0
PATIENCE = 5  # Early stopping patience

for epoch in range(EPOCHS):
    # Train for one epoch
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE
    )

    # Evaluate on test set
    val_loss, val_acc = evaluate(
        model, test_loader, criterion, DEVICE
    )

    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Update learning rate based on val accuracy
    scheduler.step(val_acc)

    # Print epoch results
    print(f"   Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Save best model (checkpoint)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"           -> Saved best model (Val Acc: {val_acc*100:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\n   Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
        break

# Load best weights back
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))


# =====================================================
# 8. FINAL EVALUATION
# =====================================================

print("\n" + "=" * 60)
print("  [6/6] MODEL EVALUATION")
print("=" * 60)

test_loss, test_accuracy = evaluate(model, test_loader, criterion, DEVICE)
print(f"\n   Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"   Test Loss     : {test_loss:.4f}")

# Save class labels mapping as JSON (needed by detect_drowsiness.py)
class_info = {
    'class_names': class_names,
    'class_to_idx': train_dataset.class_to_idx,
    'img_size': IMG_SIZE
}
with open(LABELS_PATH, 'w') as f:
    json.dump(class_info, f, indent=2)

print(f"\n   Model saved as : {MODEL_PATH}")
print(f"   Labels saved as: {LABELS_PATH}")


# =====================================================
# 9. PLOT TRAINING HISTORY
# =====================================================

print("\n   Generating training plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Accuracy plot ---
ax1.plot(history['train_acc'],
         label='Training Accuracy', color='#2196F3', linewidth=2)
ax1.plot(history['val_acc'],
         label='Validation Accuracy', color='#FF5722', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# --- Loss plot ---
ax2.plot(history['train_loss'],
         label='Training Loss', color='#2196F3', linewidth=2)
ax2.plot(history['val_loss'],
         label='Validation Loss', color='#FF5722', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
print("   Plot saved: training_results.png")

# Try to display plot (may fail on headless systems, that's OK)
try:
    plt.show()
except Exception:
    pass

# =====================================================
# DONE!
# =====================================================
print("\n" + "=" * 60)
print("  TRAINING COMPLETE!")
print("=" * 60)
print(f"  Model saved   : {MODEL_PATH}")
print(f"  Labels saved  : {LABELS_PATH}")
print(f"  Plot saved    : training_results.png")
print(f"  Best Val Acc  : {best_val_acc * 100:.2f}%")
print(f"  Test Accuracy : {test_accuracy * 100:.2f}%")
print("=" * 60)
print("\nNext step: Run  python detect_drowsiness.py  to start real-time detection.")

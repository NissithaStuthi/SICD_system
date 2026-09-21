import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import BiTemporalPatchClassificationDataset
from src.model import PatchClassifierCNN, ResNetPatchClassifier

print("=" * 65)
print("  MODULE 5: CONVOLUTIONAL NEURAL NETWORKS & PATCH CLASSIFICATION")
print("=" * 65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware Compute Device: {device}")

# 1. Prepare Patch-Level Dataset
train_patch_ds = BiTemporalPatchClassificationDataset("data/dataset_split/train", patch_size=128, max_samples=15)
test_patch_ds = BiTemporalPatchClassificationDataset("data/dataset_split/test", patch_size=128, max_samples=8)

train_loader = DataLoader(train_patch_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_patch_ds, batch_size=16, shuffle=False)

print(f"\n[1] Patch Classification Dataset:")
print(f"    - Extracted Train Patches: {len(train_patch_ds)} samples")
print(f"    - Extracted Test Patches : {len(test_patch_ds)} samples")

# 2. Build Custom CNN Architecture
cnn_model = PatchClassifierCNN(in_channels=6, num_classes=1).to(device)
total_cnn_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)

print(f"\n[2] Custom CNN Architecture:")
print(f"    - Architecture           : Conv2D -> BN -> ReLU -> MaxPool (x4) -> AdaptivePool -> FC")
print(f"    - Trainable Parameters   : {total_cnn_params:,}")

# 3. Build Transfer Learning ResNet Backbone
resnet_model = ResNetPatchClassifier(in_channels=6, num_classes=1, pretrained=False).to(device)
total_resnet_params = sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)

print(f"\n[3] Transfer Learning Architecture (ResNet-18 Backbone):")
print(f"    - Architecture           : ResNet-18 6-Channel Modified Input + Custom Classifier Head")
print(f"    - Trainable Parameters   : {total_resnet_params:,}")

# 4. Fast Training of CNN Patch Classifier
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)

print("\n[4] Training Patch Classifier CNN (2 Epochs)...")
cnn_model.train()
for epoch in range(1, 3):
    running_loss = 0.0
    correct = 0
    total = 0
    for patches, labels in train_loader:
        patches, labels = patches.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = cnn_model(patches)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * patches.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / max(1, total)
    epoch_acc = (correct / max(1, total)) * 100.0
    print(f"    - Epoch {epoch}/2 | Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_acc:.1f}%")

# 5. Evaluate on Test Patches
cnn_model.eval()
test_correct = 0
test_total = 0
sample_patches_vis = []
sample_preds_vis = []
sample_labels_vis = []

with torch.no_grad():
    for patches, labels in test_loader:
        patches, labels = patches.to(device), labels.to(device)
        outputs = cnn_model(patches)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)
        
        if len(sample_patches_vis) < 4:
            sample_patches_vis.append(patches.cpu())
            sample_preds_vis.append(preds.cpu())
            sample_labels_vis.append(labels.cpu())

test_acc = (test_correct / max(1, test_total)) * 100.0
print(f"\n[5] Test Evaluation:")
print(f"    - Test Patch Classification Accuracy: {test_acc:.2f}%")

# 6. Save Visualization of Classified Patches
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
vis_p = sample_patches_vis[0]
vis_pred = sample_preds_vis[0]
vis_lbl = sample_labels_vis[0]

for i in range(min(4, vis_p.size(0))):
    p_b = vis_p[i, :3].permute(1, 2, 0).numpy()
    p_a = vis_p[i, 3:].permute(1, 2, 0).numpy()
    pred_str = "CHANGED" if vis_pred[i].item() == 1.0 else "UNCHANGED"
    gt_str = "CHANGED" if vis_lbl[i].item() == 1.0 else "UNCHANGED"
    match_str = "[CORRECT]" if pred_str == gt_str else "[MISMATCH]"
    
    axes[0, i].imshow(np.clip(p_b, 0, 1))
    axes[0, i].set_title(f"Patch {i+1} Date T1\nGT: {gt_str}", fontsize=10, fontweight="bold")
    axes[0, i].axis("off")
    
    axes[1, i].imshow(np.clip(p_a, 0, 1))
    axes[1, i].set_title(f"Patch {i+1} Date T2\nPred: {pred_str} {match_str}", fontsize=10, fontweight="bold", color="green" if pred_str == gt_str else "red")
    axes[1, i].axis("off")

plt.suptitle("Module 5: CNN Patch-Level Change Classification & Transfer Learning", fontsize=13, fontweight="bold")
plt.tight_layout()
out_fig = "module5_cnn_classification_result.png"
plt.savefig(out_fig, dpi=150)
plt.close()

print(f"\n[SUCCESS] CNN patch classification artifact saved to: '{out_fig}'")
print("=" * 65)

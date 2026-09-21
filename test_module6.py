import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from src.dataset import BiTemporalChangeDataset
from src.model import EarlyFusionUNet
from src.metrics import BCEDiceLoss, calculate_iou, calculate_metrics

print("=" * 65)
print("  MODULE 6: SEMANTIC SEGMENTATION (EARLY-FUSION U-NET)")
print("=" * 65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware Compute Device: {device}")

# 1. Load Bi-temporal Change Dataset
train_ds = BiTemporalChangeDataset("data/dataset_split/train", img_size=(256, 256))
test_ds = BiTemporalChangeDataset("data/dataset_split/test", img_size=(256, 256))

train_subset = Subset(train_ds, range(min(50, len(train_ds))))
test_subset = Subset(test_ds, range(min(20, len(test_ds))))

train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=4, shuffle=False)

# 2. Build Early-Fusion U-Net Architecture
unet_model = EarlyFusionUNet(in_channels=6, out_channels=1).to(device)
total_params = sum(p.numel() for p in unet_model.parameters() if p.requires_grad)

print(f"\n[1] Early Fusion U-Net Architecture:")
print(f"    - Input Channels         : 6 (Concatenated Date T1 [3ch] + Date T2 [3ch])")
print(f"    - Skip Connections       : 3 Multi-scale feature bridges")
print(f"    - Trainable Parameters   : {total_params:,}")

# 3. Training Loop
criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(unet_model.parameters(), lr=1e-3)

print("\n[2] Training U-Net for Change Segmentation (2 Epochs)...")
for epoch in range(1, 3):
    unet_model.train()
    running_loss, running_iou = 0.0, 0.0
    for before, after, masks in train_loader:
        before, after, masks = before.to(device), after.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = unet_model(before, after)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_iou += calculate_iou(outputs, masks)
        
    epoch_loss = running_loss / len(train_loader)
    epoch_iou = running_iou / len(train_loader)
    print(f"    - Epoch {epoch}/2 | Train Loss: {epoch_loss:.4f} | Train IoU: {epoch_iou:.4f}")

# 4. Evaluation on Test Set
unet_model.eval()
test_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
sample_vis = None

with torch.no_grad():
    for before, after, masks in test_loader:
        before, after, masks = before.to(device), after.to(device), masks.to(device)
        outputs = unet_model(before, after)
        
        m = calculate_metrics(outputs, masks)
        for k in test_metrics:
            test_metrics[k] += m[k]
            
        if sample_vis is None:
            probs = torch.sigmoid(outputs)
            sample_vis = (before.cpu(), after.cpu(), masks.cpu(), probs.cpu())

n_test = len(test_loader)
for k in test_metrics:
    test_metrics[k] /= n_test

print(f"\n[3] U-Net Semantic Segmentation Results:")
print(f"    - Mean Precision         : {test_metrics['precision']:.4f}")
print(f"    - Mean Recall            : {test_metrics['recall']:.4f}")
print(f"    - Mean F1-Score          : {test_metrics['f1']:.4f}")
print(f"    - Mean IoU (Jaccard)     : {test_metrics['iou']:.4f}")

# 5. Save Visualization
b_vis, a_vis, m_vis, p_vis = sample_vis
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

axes[0].imshow(b_vis[0].permute(1, 2, 0).numpy())
axes[0].set_title("1. Date T1 (Before)", fontweight="bold")
axes[0].axis("off")

axes[1].imshow(a_vis[0].permute(1, 2, 0).numpy())
axes[1].set_title("2. Date T2 (After)", fontweight="bold")
axes[1].axis("off")

axes[2].imshow(m_vis[0, 0].numpy(), cmap="gray")
axes[2].set_title("3. Ground Truth Mask", fontweight="bold")
axes[2].axis("off")

axes[3].imshow((p_vis[0, 0].numpy() > 0.5).astype(np.float32), cmap="gray")
axes[3].set_title(f"4. U-Net Pred Mask (IoU: {test_metrics['iou']:.3f})", fontweight="bold")
axes[3].axis("off")

plt.suptitle("Module 6: Semantic Segmentation with Early Fusion U-Net", fontsize=13, fontweight="bold")
plt.tight_layout()
out_fig = "module6_unet_segmentation_result.png"
plt.savefig(out_fig, dpi=150)
plt.close()

print(f"\n[SUCCESS] Semantic segmentation artifact saved to: '{out_fig}'")
print("=" * 65)

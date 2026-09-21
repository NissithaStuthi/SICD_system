import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from src.dataset import BiTemporalChangeDataset
from src.model import SiameseCNN
from src.metrics import BCEDiceLoss, calculate_iou, calculate_metrics

print("=" * 65)
print("  MODULE 7: SIAMESE NETWORK FOR CHANGE DETECTION")
print("=" * 65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware Compute Device: {device}")

# 1. Load Bi-temporal Change Dataset
train_ds = BiTemporalChangeDataset("data/dataset_split/train", img_size=(256, 256))
test_ds = BiTemporalChangeDataset("data/dataset_split/test", img_size=(256, 256))

train_subset = Subset(train_ds, range(min(24, len(train_ds))))
test_subset = Subset(test_ds, range(min(12, len(test_ds))))

train_loader = DataLoader(train_subset, batch_size=6, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=4, shuffle=False)

# 2. Build Siamese CNN Architecture
siamese_model = SiameseCNN(in_channels=3, out_channels=1).to(device)
total_params = sum(p.numel() for p in siamese_model.parameters() if p.requires_grad)

print(f"\n[1] Siamese CNN Architecture:")
print(f"    - Weight-Sharing Encoder : Shared 4-stage convolutional feature extractor")
print(f"    - Feature Differencing   : Deep metric difference |F_t1 - F_t2| (512-dim embedding)")
print(f"    - Decoder                : 3-stage transpose conv reconstructing change map")
print(f"    - Trainable Parameters   : {total_params:,}")

# 3. Training Loop
criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(siamese_model.parameters(), lr=1e-3)

print("\n[2] Training Siamese CNN with Feature Differencing (2 Epochs)...")
for epoch in range(1, 3):
    siamese_model.train()
    running_loss, running_iou = 0.0, 0.0
    for before, after, masks in train_loader:
        before, after, masks = before.to(device), after.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = siamese_model(before, after)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_iou += calculate_iou(outputs, masks)
        
    epoch_loss = running_loss / len(train_loader)
    epoch_iou = running_iou / len(train_loader)
    print(f"    - Epoch {epoch}/2 | Train Loss: {epoch_loss:.4f} | Train IoU: {epoch_iou:.4f}")

# 4. Evaluation on Test Set & Feature Extraction Inspection
siamese_model.eval()
test_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
sample_vis = None

with torch.no_grad():
    for before, after, masks in test_loader:
        before, after, masks = before.to(device), after.to(device), masks.to(device)
        outputs = siamese_model(before, after)
        
        m = calculate_metrics(outputs, masks)
        for k in test_metrics:
            test_metrics[k] += m[k]
            
        if sample_vis is None:
            # Extract intermediate feature difference
            f1 = siamese_model.extract_features(before[:1])
            f2 = siamese_model.extract_features(after[:1])
            diff_feat = torch.abs(f1 - f2).mean(dim=1).squeeze().cpu().numpy()
            
            probs = torch.sigmoid(outputs)
            sample_vis = (before.cpu(), after.cpu(), masks.cpu(), probs.cpu(), diff_feat)

n_test = len(test_loader)
for k in test_metrics:
    test_metrics[k] /= n_test

print(f"\n[3] Siamese Network Results:")
print(f"    - Mean Precision         : {test_metrics['precision']:.4f}")
print(f"    - Mean Recall            : {test_metrics['recall']:.4f}")
print(f"    - Mean F1-Score          : {test_metrics['f1']:.4f}")
print(f"    - Mean IoU (Jaccard)     : {test_metrics['iou']:.4f}")

# 5. Save Visualization
b_vis, a_vis, m_vis, p_vis, diff_feat_vis = sample_vis
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

axes[0].imshow(b_vis[0].permute(1, 2, 0).numpy())
axes[0].set_title("1. Date T1 (Before)", fontweight="bold")
axes[0].axis("off")

axes[1].imshow(a_vis[0].permute(1, 2, 0).numpy())
axes[1].set_title("2. Date T2 (After)", fontweight="bold")
axes[1].axis("off")

axes[2].imshow(diff_feat_vis, cmap="viridis")
axes[2].set_title("3. Siamese Deep Feature Diff", fontweight="bold")
axes[2].axis("off")

axes[3].imshow((p_vis[0, 0].numpy() > 0.5).astype(np.float32), cmap="gray")
axes[3].set_title(f"4. Siamese Pred Mask (IoU: {test_metrics['iou']:.3f})", fontweight="bold")
axes[3].axis("off")

axes[4].imshow(m_vis[0, 0].numpy(), cmap="gray")
axes[4].set_title("5. Ground Truth Mask", fontweight="bold")
axes[4].axis("off")

plt.suptitle("Module 7: Siamese Neural Network with Weight-Sharing & Feature Difference", fontsize=13, fontweight="bold")
plt.tight_layout()
out_fig = "module7_siamese_network_result.png"
plt.savefig(out_fig, dpi=150)
plt.close()

print(f"\n[SUCCESS] Siamese network artifact saved to: '{out_fig}'")
print("=" * 65)

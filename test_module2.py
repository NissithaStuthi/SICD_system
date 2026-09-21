import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import BiTemporalChangeDataset
from src.preprocessing import extract_tiles

print("=" * 65)
print("  MODULE 2: DATASET PREPARATION & PYTORCH PIPELINE")
print("=" * 65)

# 1. Dataset Directory Organization Check
base_dir = "data/dataset_split"
train_ds = BiTemporalChangeDataset(os.path.join(base_dir, "train"), img_size=(256, 256))
val_ds = BiTemporalChangeDataset(os.path.join(base_dir, "validation"), img_size=(256, 256))
test_ds = BiTemporalChangeDataset(os.path.join(base_dir, "test"), img_size=(256, 256))

print(f"\n[1] Change-Detection Dataset Splits:")
print(f"    - Train Set Pairs        : {len(train_ds)} image pairs")
print(f"    - Validation Set Pairs   : {len(val_ds)} image pairs")
print(f"    - Test Set Pairs         : {len(test_ds)} image pairs")
print(f"    - Total Dataset Volume   : {len(train_ds) + len(val_ds) + len(test_ds)} pairs")

# 2. PyTorch DataLoader Pipeline Check
batch_size = 4
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
b_batch, a_batch, m_batch = next(iter(train_loader))

print(f"\n[2] PyTorch DataLoader Tensor Pipeline:")
print(f"    - Batch Size             : {batch_size}")
print(f"    - Before Tensor Shape    : {b_batch.shape} (N, C, H, W)")
print(f"    - After Tensor Shape     : {a_batch.shape} (N, C, H, W)")
print(f"    - Ground Truth Mask Shape: {m_batch.shape} (N, 1, H, W)")
print(f"    - Tensor Value Ranges    : Before in [{b_batch.min():.2f}, {b_batch.max():.2f}], Mask in [{m_batch.min():.0f}, {m_batch.max():.0f}]")

# 3. Patch Extraction & Tiling for Large Satellite Imagery
sample_file = train_ds.filenames[0]
sample_b_path = os.path.join(base_dir, "train/before", sample_file)
sample_img = cv2.cvtColor(cv2.imread(sample_b_path), cv2.COLOR_BGR2RGB)

# Simulate tiling on a 256x256 image extracting 128x128 patches
tile_size = 128
tiles, positions = extract_tiles(sample_img, tile_size=tile_size, stride=tile_size)

print(f"\n[3] Large Image Tiling & Patch Extraction:")
print(f"    - Source Image Resolution: {sample_img.shape[0]}x{sample_img.shape[1]}")
print(f"    - Patch Window Size      : {tile_size}x{tile_size}")
print(f"    - Extracted Patches      : {len(tiles)} uniform sub-tiles")

# 4. Save Patch Grid Visualization
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    if i < len(tiles):
        ax.imshow(tiles[i])
        ax.set_title(f"Patch {i+1} (Pos: y={positions[i][0]}, x={positions[i][1]})", fontsize=10, fontweight="bold")
    ax.axis("off")

plt.suptitle("Module 2: Satellite Image Tiling & Patch Extraction Pipeline", fontsize=12, fontweight="bold")
plt.tight_layout()
out_fig = "module2_dataset_patches.png"
plt.savefig(out_fig, dpi=150)
plt.close()

print(f"\n[SUCCESS] Dataset preparation artifact saved to: '{out_fig}'")
print("=" * 65)

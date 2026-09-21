import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=" * 65)
print("  MODULE 1: INTRODUCTION TO SATELLITE IMAGE ANALYSIS")
print("=" * 65)

# 1. Define Paths to Sample Satellite Images
split_dir = "data/dataset_split/test"
before_dir = os.path.join(split_dir, "before")
after_dir = os.path.join(split_dir, "after")
mask_dir = os.path.join(split_dir, "masks")

if not os.path.exists(before_dir) or len(os.listdir(before_dir)) == 0:
    print("Error: Test dataset directory not found!")
    exit(1)

sample_file = sorted(os.listdir(before_dir))[0]
b_path = os.path.join(before_dir, sample_file)
a_path = os.path.join(after_dir, sample_file)
m_path = os.path.join(mask_dir, sample_file)

# 2. Explore Satellite Imagery & Characteristics
img_t1 = cv2.cvtColor(cv2.imread(b_path), cv2.COLOR_BGR2RGB)
img_t2 = cv2.cvtColor(cv2.imread(a_path), cv2.COLOR_BGR2RGB)
mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)

print(f"\n[1] Satellite Image Exploration:")
print(f"    - Sample Filename        : {sample_file}")
print(f"    - Spatial Dimensions     : Height = {img_t1.shape[0]}px, Width = {img_t1.shape[1]}px")
print(f"    - Spectral Channels      : {img_t1.shape[2]} (Red, Green, Blue)")
print(f"    - Data Type & Range      : {img_t1.dtype}, [{img_t1.min()}, {img_t1.max()}]")
print(f"    - Temporal Aspect        : Bi-temporal Pair (Date T1 vs. Date T2)")

# 3. Multi-temporal Pixel Statistics
t1_mean, t1_std = img_t1.mean(axis=(0, 1)), img_t1.std(axis=(0, 1))
t2_mean, t2_std = img_t2.mean(axis=(0, 1)), img_t2.std(axis=(0, 1))

print(f"\n[2] Spectral Channel Statistics (RGB):")
print(f"    - Date T1 Mean per channel: R={t1_mean[0]:.1f}, G={t1_mean[1]:.1f}, B={t1_mean[2]:.1f}")
print(f"    - Date T2 Mean per channel: R={t2_mean[0]:.1f}, G={t2_mean[1]:.1f}, B={t2_mean[2]:.1f}")

# 4. Manual Visual Change Analysis & Overlay
diff_vis = cv2.absdiff(img_t1, img_t2)
overlay = img_t2.copy()
overlay[mask > 127] = [255, 0, 0] # Highlight change in red

change_pixels = (mask > 127).sum()
total_pixels = mask.shape[0] * mask.shape[1]
change_pct = (change_pixels / total_pixels) * 100.0

print(f"\n[3] Ground Truth Change Assessment:")
print(f"    - Changed Pixels Area    : {change_pixels:,} px ({change_pct:.2f}% of tile area)")
print(f"    - Types of Target Changes: Urban expansion, construction, road development, vegetation loss")

# 5. Save Visualization Figure
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
axes[0].imshow(img_t1)
axes[0].set_title("1. Before (Date T1)", fontsize=11, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(img_t2)
axes[1].set_title("2. After (Date T2)", fontsize=11, fontweight="bold")
axes[1].axis("off")

axes[2].imshow(diff_vis)
axes[2].set_title("3. Multi-Temporal Diff (RGB)", fontsize=11, fontweight="bold")
axes[2].axis("off")

axes[3].imshow(overlay)
axes[3].set_title(f"4. Change Mask Overlay ({change_pct:.1f}%)", fontsize=11, fontweight="bold")
axes[3].axis("off")

plt.suptitle("Module 1: Satellite Image Analysis & Multi-Temporal Change Exploration", fontsize=13, fontweight="bold")
plt.tight_layout()
out_fig = "module1_image_analysis.png"
plt.savefig(out_fig, dpi=150)
plt.close()

print(f"\n[SUCCESS] Visual analysis artifact saved to: '{out_fig}'")
print("=" * 65)

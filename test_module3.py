import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing import match_histograms, apply_clahe, denoise_bilateral, normalize_minmax

print("=" * 65)
print("  MODULE 3: SATELLITE IMAGE PREPROCESSING & NORMALIZATION")
print("=" * 65)

# 1. Load Sample Bi-Temporal Images
b_dir = "data/dataset_split/train/before"
a_dir = "data/dataset_split/train/after"
files = sorted(os.listdir(b_dir))
sample_f = files[0]

img1 = cv2.cvtColor(cv2.imread(os.path.join(b_dir, sample_f)), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread(os.path.join(a_dir, sample_f)), cv2.COLOR_BGR2RGB)

print(f"\n[1] Input Imagery:")
print(f"    - Sample Filename        : {sample_f}")
print(f"    - Original Resolution    : {img1.shape[0]}x{img1.shape[1]}")

# 2. Histogram Matching for Atmospheric / Illumination Correction
img2_matched = match_histograms(img2, img1)

# 3. Bilateral Filtering for Edge-Preserving Denoising
img1_denoised = denoise_bilateral(img1, d=9, sigma_color=50, sigma_space=50)
img2_denoised = denoise_bilateral(img2_matched, d=9, sigma_color=50, sigma_space=50)

# 4. CLAHE Contrast Enhancement
img1_clahe = apply_clahe(img1_denoised, clip_limit=2.0)
img2_clahe = apply_clahe(img2_denoised, clip_limit=2.0)

# 5. Normalization
img1_norm = normalize_minmax(img1_clahe)
img2_norm = normalize_minmax(img2_clahe)

# 6. Preprocessed Difference Analysis
raw_diff = cv2.absdiff(img1, img2)
prep_diff = cv2.absdiff(img1_clahe, img2_clahe)

print(f"\n[2] Preprocessing Transformations Applied:")
print(f"    - Histogram Matching     : Equalized illumination across Date T1 and T2")
print(f"    - Bilateral Denoising    : Suppressed sensor grain while retaining boundary sharpness")
print(f"    - CLAHE Contrast Tuning  : Enhanced local terrain features")
print(f"    - Min-Max Normalization  : Float32 scaled to [{img1_norm.min():.2f}, {img1_norm.max():.2f}]")
print(f"    - Noise Diff Suppression : Raw Diff Mean={raw_diff.mean():.2f} -> Prep Diff Mean={prep_diff.mean():.2f}")

# 7. Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].imshow(img1)
axes[0, 0].set_title("1. Date T1 (Raw)", fontweight="bold")
axes[0, 0].axis("off")

axes[0, 1].imshow(img2)
axes[0, 1].set_title("2. Date T2 (Raw)", fontweight="bold")
axes[0, 1].axis("off")

axes[0, 2].imshow(raw_diff)
axes[0, 2].set_title("3. Raw Difference Map", fontweight="bold")
axes[0, 2].axis("off")

axes[1, 0].imshow(img1_clahe)
axes[1, 0].set_title("4. Date T1 (Denoised + CLAHE)", fontweight="bold")
axes[1, 0].axis("off")

axes[1, 1].imshow(img2_clahe)
axes[1, 1].set_title("5. Date T2 (HistMatched + CLAHE)", fontweight="bold")
axes[1, 1].axis("off")

axes[1, 2].imshow(prep_diff)
axes[1, 2].set_title("6. Preprocessed Difference Map", fontweight="bold")
axes[1, 2].axis("off")

plt.suptitle("Module 3: Satellite Image Preprocessing & Radiometric Calibration", fontsize=13, fontweight="bold")
plt.tight_layout()
out_fig = "module3_preprocessing_preview.png"
plt.savefig(out_fig, dpi=150)
plt.close()

print(f"\n[SUCCESS] Preprocessing artifact saved to: '{out_fig}'")
print("=" * 65)

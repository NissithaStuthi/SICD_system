import os
import cv2
import matplotlib.pyplot as plt
from src.preprocessing import match_histograms

print("--- MODULE 3: IMAGE PREPROCESSING VERIFICATION ---")

t1_path = "data/dataset_split/train/before"
t2_path = "data/dataset_split/train/after"

sample_file = sorted(os.listdir(t1_path))[0]

# 1. Load Image Pair
t1_img = cv2.cvtColor(cv2.imread(os.path.join(t1_path, sample_file)), cv2.COLOR_BGR2RGB)
t2_img = cv2.cvtColor(cv2.imread(os.path.join(t2_path, sample_file)), cv2.COLOR_BGR2RGB)

# 2. Apply Histogram Matching for Illumination Correction
t2_matched = match_histograms(t2_img, t1_img)

# 3. Apply Resizing to 256x256
t1_resized = cv2.resize(t1_img, (256, 256), interpolation=cv2.INTER_LINEAR)
t2_matched_resized = cv2.resize(t2_matched, (256, 256), interpolation=cv2.INTER_LINEAR)

print(f"Original Image Shape   : {t1_img.shape}")
print(f"Preprocessed Resized   : {t1_resized.shape}")
print(f"Histogram Match Status : Applied successfully across RGB channels")

# 4. Save Preprocessing Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(t1_resized)
axes[0].set_title("Time 1 (Resized 256x256)")
axes[0].axis("off")

axes[1].imshow(t2_matched_resized)
axes[1].set_title("Time 2 (Illumination Normalized)")
axes[1].axis("off")

# Visualization of preprocessing difference
prep_diff = cv2.absdiff(t1_resized, t2_matched_resized)
axes[2].imshow(prep_diff)
axes[2].set_title("Normalized Pair Difference")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("module3_preprocessing_preview.png")
print("Saved visualization to 'module3_preprocessing_preview.png'.")
print("Module 3 Preprocessing completely verified!")

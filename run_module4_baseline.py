import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.baseline import compute_cva, apply_threshold_and_morphology

print("==============================================================")
print("     MODULE 4: TRADITIONAL CHANGE DETECTION EVALUATION        ")
print("==============================================================")

test_before_dir = "data/dataset_split/test/before"
test_after_dir = "data/dataset_split/test/after"
test_mask_dir = "data/dataset_split/test/masks"

files = sorted(os.listdir(test_before_dir))
print(f"Evaluating traditional pipeline on {len(files)} test image pairs...\n")

total_iou, total_f1, total_precision, total_recall = 0.0, 0.0, 0.0, 0.0

for f in files:
    img1 = cv2.cvtColor(cv2.imread(os.path.join(test_before_dir, f)), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(os.path.join(test_after_dir, f)), cv2.COLOR_BGR2RGB)
    gt_mask = cv2.imread(os.path.join(test_mask_dir, f), cv2.IMREAD_GRAYSCALE)
    _, gt_bin = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)

    # Practical Pipeline Flow
    diff_map = compute_cva(img1, img2)
    pred_mask = apply_threshold_and_morphology(diff_map) // 255

    # Metrics computation
    tp = np.logical_and(pred_mask == 1, gt_bin == 1).sum()
    fp = np.logical_and(pred_mask == 1, gt_bin == 0).sum()
    fn = np.logical_and(pred_mask == 0, gt_bin == 1).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)

    total_precision += precision
    total_recall += recall
    total_f1 += f1
    total_iou += iou

n = len(files)
print(f"--- TRADITIONAL BASELINE RESULTS ({n} Test Samples) ---")
print(f"   [x] Mean Precision : {total_precision / n:.4f}")
print(f"   [x] Mean Recall    : {total_recall / n:.4f}")
print(f"   [x] Mean F1-Score  : {total_f1 / n:.4f}")
print(f"   [x] Mean IoU Score : {total_iou / n:.4f}")
print("==============================================================")

# Save visualization of the Practical Pipeline
sample_f = files[0]
s_img1 = cv2.cvtColor(cv2.imread(os.path.join(test_before_dir, sample_f)), cv2.COLOR_BGR2RGB)
s_img2 = cv2.cvtColor(cv2.imread(os.path.join(test_after_dir, sample_f)), cv2.COLOR_BGR2RGB)
s_gt = cv2.imread(os.path.join(test_mask_dir, sample_f), cv2.IMREAD_GRAYSCALE)
s_diff = compute_cva(s_img1, s_img2)
s_pred = apply_threshold_and_morphology(s_diff)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(s_img1)
axes[0].set_title("1. Before Image")
axes[0].axis("off")

axes[1].imshow(s_img2)
axes[1].set_title("2. After Image")
axes[1].axis("off")

axes[2].imshow(s_diff, cmap="gray")
axes[2].set_title("3. Pixel Difference (CVA)")
axes[2].axis("off")

axes[3].imshow(s_pred, cmap="gray")
axes[3].set_title("4. Threshold + Morph Mask")
axes[3].axis("off")

axes[4].imshow(s_gt, cmap="gray")
axes[4].set_title("5. Ground Truth Mask")
axes[4].axis("off")

plt.tight_layout()
plt.savefig("module4_traditional_baseline_result.png")
print("\nSaved pipeline visualization to 'module4_traditional_baseline_result.png'")

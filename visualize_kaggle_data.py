import os
import cv2
import matplotlib.pyplot as plt

# Paths to your organized Kaggle files
t1_dir = "data/raw/time1"
t2_dir = "data/raw/time2"
mask_dir = "data/raw/masks"

# Get the first image filename
sample_file = sorted(os.listdir(t1_dir))[0]

# Read images
t1_img = cv2.cvtColor(cv2.imread(os.path.join(t1_dir, sample_file)), cv2.COLOR_BGR2RGB)
t2_img = cv2.cvtColor(cv2.imread(os.path.join(t2_dir, sample_file)), cv2.COLOR_BGR2RGB)
mask_img = cv2.imread(os.path.join(mask_dir, sample_file), cv2.IMREAD_GRAYSCALE)

# Plot actual Kaggle satellite pairs
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(t1_img)
axes[0].set_title(f"Kaggle Date 1: {sample_file}")
axes[0].axis("off")

axes[1].imshow(t2_img)
axes[1].set_title(f"Kaggle Date 2: {sample_file}")
axes[1].axis("off")

axes[2].imshow(mask_img, cmap="gray")
axes[2].set_title("Ground Truth Change Mask")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("real_kaggle_sample.png")
print("Saved real Kaggle image visualization to 'real_kaggle_sample.png'!")

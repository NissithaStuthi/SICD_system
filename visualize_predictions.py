import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.model import SiameseUNet
from src.baseline import compute_cva, apply_threshold_and_morphology

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_before_dir = "data/dataset_split/test/before"
test_after_dir = "data/dataset_split/test/after"
test_mask_dir = "data/dataset_split/test/masks"

files = sorted(os.listdir(test_before_dir))
sample_f = files[0]

# Load Images
img1 = cv2.cvtColor(cv2.imread(os.path.join(test_before_dir, sample_f)), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread(os.path.join(test_after_dir, sample_f)), cv2.COLOR_BGR2RGB)
gt = cv2.imread(os.path.join(test_mask_dir, sample_f), cv2.IMREAD_GRAYSCALE)

# 1. Traditional Baseline Mask
diff = compute_cva(img1, img2)
trad_mask = apply_threshold_and_morphology(diff)

# 2. Deep Learning Siamese U-Net Mask
model = SiameseUNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("best_siamese_model.pth", map_location=device))
model.eval()

img1_res = cv2.resize(img1, (256, 256))
img2_res = cv2.resize(img2, (256, 256))

t1 = torch.tensor(img1_res, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
t2 = torch.tensor(img2_res, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

with torch.no_grad():
    out = model(t1.to(device), t2.to(device))
    pred = (torch.sigmoid(out) > 0.5).float().squeeze().cpu().numpy()

# Plot Comparison
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(img1_res)
axes[0].set_title("Before Image")
axes[0].axis("off")

axes[1].imshow(img2_res)
axes[1].set_title("After Image")
axes[1].axis("off")

axes[2].imshow(cv2.resize(trad_mask, (256, 256)), cmap="gray")
axes[2].set_title("Traditional Mask (Mod 4)")
axes[2].axis("off")

axes[3].imshow(pred, cmap="gray")
axes[3].set_title("Siamese U-Net (Mod 5)")
axes[3].axis("off")

axes[4].imshow(cv2.resize(gt, (256, 256)), cmap="gray")
axes[4].set_title("Ground Truth")
axes[4].axis("off")

plt.tight_layout()
plt.savefig("module5_model_comparison_result.png")
print("Saved comparison figure to 'module5_model_comparison_result.png'")

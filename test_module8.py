import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from src.dataset import BiTemporalChangeDataset
from src.model import EarlyFusionUNet, SiameseCNN, SiameseUNetBasic, AttentionSiameseUNet
from src.metrics import calculate_metrics

print("=" * 78)
print("  MODULE 8: ADVANCED CHANGE DETECTION ARCHITECTURES & BENCHMARK")
print("=" * 78)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware Compute Device: {device}")

# 1. Dataset Setup
test_ds = BiTemporalChangeDataset("data/dataset_split/test", img_size=(256, 256))
test_subset = Subset(test_ds, range(min(30, len(test_ds))))
test_loader = DataLoader(test_subset, batch_size=6, shuffle=False)

# 2. Instantiate all 4 Architectural Models
models_dict = {
    "1. Early-Fusion U-Net": EarlyFusionUNet(in_channels=6, out_channels=1).to(device),
    "2. Siamese CNN": SiameseCNN(in_channels=3, out_channels=1).to(device),
    "3. Siamese U-Net": SiameseUNetBasic(in_channels=3, out_channels=1).to(device),
    "4. Attention Siamese U-Net": AttentionSiameseUNet(in_channels=3, out_channels=1).to(device)
}

# If existing trained checkpoint is present, load it into Attention Siamese U-Net
if os.path.exists("best_siamese_model.pth"):
    try:
        models_dict["4. Attention Siamese U-Net"].load_state_dict(
            torch.load("best_siamese_model.pth", map_location=device)
        )
        print("[*] Loaded pre-trained weights for Attention Siamese U-Net checkpoint.")
    except Exception as e:
        print(f"[*] Note on weight load: {e}")

# 3. Model Benchmark Evaluation
results = []
sample_predictions = {}
sample_b, sample_a, sample_gt = None, None, None

print(f"\n[1] Benchmarking all 4 Architectures across {len(test_subset)} Test Pairs...\n")

for model_name, model in models_dict.items():
    model.eval()
    total_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        for b_img, a_img, masks in test_loader:
            b_img, a_img, masks = b_img.to(device), a_img.to(device), masks.to(device)
            
            if "Early-Fusion" in model_name:
                outputs = model(b_img, a_img)
            else:
                outputs = model(b_img, a_img)
                
            m = calculate_metrics(outputs, masks)
            for k in total_metrics:
                total_metrics[k] += m[k]

            if sample_b is None:
                sample_b = b_img[0].cpu().permute(1, 2, 0).numpy()
                sample_a = a_img[0].cpu().permute(1, 2, 0).numpy()
                sample_gt = masks[0, 0].cpu().numpy()

            if model_name not in sample_predictions:
                probs = torch.sigmoid(outputs[0, 0]).cpu().numpy()
                sample_predictions[model_name] = (probs > 0.5).astype(np.float32)

    n_batches = len(test_loader)
    for k in total_metrics:
        total_metrics[k] /= n_batches

    results.append({
        "Model": model_name,
        "Params": f"{total_params:,}",
        "Precision": total_metrics["precision"],
        "Recall": total_metrics["recall"],
        "F1": total_metrics["f1"],
        "IoU": total_metrics["iou"]
    })

# 4. Print Comparison Table
print("-" * 78)
print(f"{'Model Architecture':<28} | {'Params':<12} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'IoU':<8}")
print("-" * 78)
for r in results:
    print(f"{r['Model']:<28} | {r['Params']:<12} | {r['Precision']:<9.4f} | {r['Recall']:<8.4f} | {r['F1']:<8.4f} | {r['IoU']:<8.4f}")
print("-" * 78)

# 5. Visual Comparison Figure
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].imshow(sample_b)
axes[0, 0].set_title("1. Date T1 (Before)", fontweight="bold")
axes[0, 0].axis("off")

axes[0, 1].imshow(sample_a)
axes[0, 1].set_title("2. Date T2 (After)", fontweight="bold")
axes[0, 1].axis("off")

axes[0, 2].imshow(sample_gt, cmap="gray")
axes[0, 2].set_title("3. Ground Truth Mask", fontweight="bold")
axes[0, 2].axis("off")

model_keys = list(models_dict.keys())
axes[1, 0].imshow(sample_predictions[model_keys[0]], cmap="gray")
axes[1, 0].set_title(f"4. {model_keys[0]}\n(IoU: {results[0]['IoU']:.3f})", fontweight="bold")
axes[1, 0].axis("off")

axes[1, 1].imshow(sample_predictions[model_keys[1]], cmap="gray")
axes[1, 1].set_title(f"5. {model_keys[1]}\n(IoU: {results[1]['IoU']:.3f})", fontweight="bold")
axes[1, 1].axis("off")

axes[1, 2].imshow(sample_predictions[model_keys[3]], cmap="gray")
axes[1, 2].set_title(f"6. {model_keys[3]}\n(IoU: {results[3]['IoU']:.3f})", fontweight="bold", color="green")
axes[1, 2].axis("off")

plt.suptitle("Module 8: Advanced Architecture Benchmark (Attention Siamese U-Net)", fontsize=14, fontweight="bold")
plt.tight_layout()
out_fig = "module8_architecture_comparison.png"
plt.savefig(out_fig, dpi=150)
plt.close()

print(f"\n[SUCCESS] Architecture comparison artifact saved to: '{out_fig}'")
print("=" * 78)

"""
Module 9 - Advanced Deep Learning: CNN + Transformer Hybrid Change Detection
=============================================================================
Model: HybridTransformerSiameseUNet
  * Shared-weight CNN encoder  (dual-branch, 4 scales)
  * TransformerBottleneck      (global self-attention context)
  * CBAM ChannelSpatialAttention on skip connections
  * U-Net Decoder              (precise spatial reconstruction)

Output artifact: module9_transformer_result.png
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from src.dataset import BiTemporalChangeDataset
from src.model import HybridTransformerSiameseUNet, _AttentionSiameseUNet
from src.metrics import BCEDiceLoss, calculate_iou, calculate_metrics

# ==============================================================================
# Banner
# ==============================================================================
print("=" * 78)
print("  MODULE 9: ADVANCED DEEP LEARNING -- CNN + TRANSFORMER HYBRID")
print("  Model : HybridTransformerSiameseUNet")
print("  Innovations: TransformerBottleneck + CBAM Channel/Spatial Attention")
print("=" * 78)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware Compute Device : {device}")

# ==============================================================================
# 1. Dataset Setup
# ==============================================================================
TRAIN_DIR = "data/dataset_split/train"
TEST_DIR  = "data/dataset_split/test"
IMG_SIZE  = (256, 256)

train_ds = BiTemporalChangeDataset(TRAIN_DIR, img_size=IMG_SIZE)
test_ds  = BiTemporalChangeDataset(TEST_DIR,  img_size=IMG_SIZE)

train_subset = Subset(train_ds, range(min(24, len(train_ds))))
test_subset  = Subset(test_ds,  range(min(30, len(test_ds))))

train_loader = DataLoader(train_subset, batch_size=4, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_subset,  batch_size=4, shuffle=False, num_workers=0)

print(f"Train samples : {len(train_subset)}")
print(f"Test  samples : {len(test_subset)}")

# ==============================================================================
# 2. Build HybridTransformerSiameseUNet (Module 9)
# ==============================================================================
model = HybridTransformerSiameseUNet(in_channels=3, out_channels=1).to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n[1] HybridTransformerSiameseUNet Architecture Summary:")
print(f"    +- Shared CNN Encoder      : 4-stage DoubleConv + MaxPool")
print(f"    +- Temporal Fusion         : Concatenate T1 & T2 features at each scale")
print(f"    +- CNN Bottleneck          : DoubleConv(1024 -> 512)")
print(f"    +- TransformerBottleneck   : embed_dim=256, num_heads=4, 2 encoder layers")
print(f"    +- CBAM Skip Attention     : Channel + Spatial attention at 3 scales")
print(f"    +- U-Net Decoder           : 3-stage up-sampling with attended skip conns")
print(f"    +- Output Head             : Conv1x1 -> binary change logit map")
print(f"    Total Parameters           : {total_params:,}")
print(f"    Trainable Parameters       : {trainable_params:,}")

# ==============================================================================
# 3. Training Loop (2 epochs -- demonstration)
# ==============================================================================
criterion = BCEDiceLoss(bce_weight=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-6)

NUM_EPOCHS = 2
print(f"\n[2] Training HybridTransformerSiameseUNet ({NUM_EPOCHS} epochs) ...")
print(f"    Optimizer : AdamW  (lr=1e-4, weight_decay=1e-4)")
print(f"    Scheduler : CosineAnnealingLR")
print(f"    Loss      : BCE + Dice hybrid")
print()

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss, epoch_iou = 0.0, 0.0

    for before, after, masks in train_loader:
        before = before.to(device)
        after  = after.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        logits = model(before, after)
        loss   = criterion(logits, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_iou  += calculate_iou(logits.detach(), masks)

    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    avg_iou  = epoch_iou  / len(train_loader)
    lr_now   = optimizer.param_groups[0]["lr"]
    print(f"    Epoch [{epoch}/{NUM_EPOCHS}] | Loss: {avg_loss:.4f} | "
          f"IoU: {avg_iou:.4f} | LR: {lr_now:.2e}")

# ==============================================================================
# 4. Evaluation -- Module 9 vs Module 8
# ==============================================================================
print("\n[3] Evaluating Models on Test Set ...")

# Module 8 baseline
m8_model = _AttentionSiameseUNet(in_channels=3, out_channels=1).to(device)
if os.path.exists("best_siamese_model.pth"):
    try:
        m8_model.load_state_dict(
            torch.load("best_siamese_model.pth", map_location=device)
        )
        print("    [*] Loaded pre-trained weights -> Module 8 (AttentionSiameseUNet)")
    except Exception as e:
        print(f"    [*] Module 8 weight load (random init used): {e}")

eval_models = {
    "Module 8 - Attention Siamese U-Net":          m8_model,
    "Module 9 - HybridTransformer Siamese U-Net":  model,
}

results       = []
sample_vis    = {}
sample_before = None
sample_after  = None
sample_gt     = None

for model_name, eval_model in eval_models.items():
    eval_model.eval()
    total_m  = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    n_params = sum(p.numel() for p in eval_model.parameters() if p.requires_grad)

    with torch.no_grad():
        for b_img, a_img, masks in test_loader:
            b_img = b_img.to(device)
            a_img = a_img.to(device)
            masks = masks.to(device)

            logits = eval_model(b_img, a_img)
            m = calculate_metrics(logits, masks)
            for k in total_m:
                total_m[k] += m[k]

            if sample_before is None:
                sample_before = b_img[0].cpu().permute(1, 2, 0).numpy()
                sample_after  = a_img[0].cpu().permute(1, 2, 0).numpy()
                sample_gt     = masks[0, 0].cpu().numpy()

            if model_name not in sample_vis:
                probs = torch.sigmoid(logits[0, 0]).cpu().numpy()
                sample_vis[model_name] = (probs > 0.5).astype(np.float32)

    n_batches = len(test_loader)
    for k in total_m:
        total_m[k] /= n_batches

    results.append({
        "Model":     model_name,
        "Params":    f"{n_params:,}",
        "Precision": total_m["precision"],
        "Recall":    total_m["recall"],
        "F1":        total_m["f1"],
        "IoU":       total_m["iou"],
    })

# ==============================================================================
# 5. Print Comparison Table
# ==============================================================================
print("\n" + "=" * 78)
print(f"  {'Architecture':<44} {'Params':<14} {'F1':<8} {'IoU':<8}")
print("=" * 78)
for r in results:
    marker = " << NEW" if "HybridTransformer" in r["Model"] else ""
    print(f"  {r['Model']:<44} {r['Params']:<14} {r['F1']:<8.4f} {r['IoU']:<8.4f}{marker}")
print("-" * 78)

m9 = results[-1]
print(f"\n  Detailed Metrics - Module 9 HybridTransformerSiameseUNet:")
print(f"    Precision : {m9['Precision']:.4f}")
print(f"    Recall    : {m9['Recall']:.4f}")
print(f"    F1-Score  : {m9['F1']:.4f}")
print(f"    IoU       : {m9['IoU']:.4f}")

# ==============================================================================
# 6. Feature Map Extraction for Visualisation
# ==============================================================================
print("\n[4] Extracting internal feature maps for visualisation ...")

model.eval()
with torch.no_grad():
    t1_t = torch.tensor(sample_before).permute(2, 0, 1).unsqueeze(0).to(device)
    t2_t = torch.tensor(sample_after ).permute(2, 0, 1).unsqueeze(0).to(device)

    # CNN multi-scale features
    t1_x1, t1_x2, t1_x3, t1_x4 = model.forward_single(t1_t)
    t2_x1, t2_x2, t2_x3, t2_x4 = model.forward_single(t2_t)

    # Feature difference at bottleneck scale (E4)
    diff_e4 = torch.abs(t1_x4 - t2_x4).mean(dim=1).squeeze().cpu().numpy()

    # CBAM spatial attention map at scale E3
    f3 = torch.cat([t1_x3, t2_x3], dim=1)
    avg_s = torch.mean(f3, dim=1, keepdim=True)
    max_s, _ = torch.max(f3, dim=1, keepdim=True)
    cbam_input = torch.cat([avg_s, max_s], dim=1)
    spatial_att = model.att3.sigmoid_spatial(
        model.att3.conv_spatial(cbam_input)
    ).squeeze().cpu().numpy()

print(f"    CNN bottleneck diff map shape  : {diff_e4.shape}")
print(f"    CBAM spatial attention shape   : {spatial_att.shape}")

# ==============================================================================
# 7. 6-Panel Visualisation Figure
# ==============================================================================
print("\n[5] Generating visualisation figure ...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor("#0d1117")

PANEL_STYLE = dict(fontsize=11, fontweight="bold", color="white", pad=8)

def style_ax(ax, title):
    ax.set_title(title, **PANEL_STYLE)
    ax.axis("off")
    ax.set_facecolor("#161b22")

# Panel 1: T1 Before
axes[0, 0].imshow(np.clip(sample_before, 0, 1))
style_ax(axes[0, 0], "[1] T1 - Before Image")

# Panel 2: T2 After
axes[0, 1].imshow(np.clip(sample_after, 0, 1))
style_ax(axes[0, 1], "[2] T2 - After Image")

# Panel 3: CNN Bottleneck Diff Feature (E4)
diff_norm = (diff_e4 - diff_e4.min()) / (diff_e4.max() - diff_e4.min() + 1e-8)
im3 = axes[0, 2].imshow(diff_norm, cmap="inferno", interpolation="bilinear")
style_ax(axes[0, 2], "[3] CNN Bottleneck Diff Feature (E4)")
fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

# Panel 4: CBAM Spatial Attention Heatmap
att_norm = (spatial_att - spatial_att.min()) / (spatial_att.max() - spatial_att.min() + 1e-8)
im4 = axes[1, 0].imshow(att_norm, cmap="plasma", interpolation="bilinear")
style_ax(axes[1, 0], "[4] CBAM Spatial Attention Map (E3 skip)")
fig.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

# Panel 5: Module 9 Prediction
m9_key  = "Module 9 - HybridTransformer Siamese U-Net"
pred_m9 = sample_vis.get(m9_key, np.zeros_like(sample_gt))
axes[1, 1].imshow(pred_m9, cmap="gray", vmin=0, vmax=1)
style_ax(axes[1, 1],
         f"[5] Module 9 Prediction\n(F1: {m9['F1']:.3f}  IoU: {m9['IoU']:.3f})")

# Panel 6: Ground Truth
axes[1, 2].imshow(sample_gt, cmap="gray", vmin=0, vmax=1)
style_ax(axes[1, 2], "[6] Ground Truth Change Mask")

fig.suptitle(
    "Module 9 - CNN + Transformer Hybrid Change Detection\n"
    "Dual CNN Encoder | TransformerBottleneck | CBAM Channel+Spatial Attention | U-Net Decoder",
    fontsize=13, fontweight="bold", color="white", y=0.98
)
plt.tight_layout(rect=[0, 0, 1, 0.96])

OUT_FIG = "module9_transformer_result.png"
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"    Saved -> '{OUT_FIG}'")

# ==============================================================================
# 8. Architecture Innovation Summary
# ==============================================================================
print("\n" + "=" * 78)
print("  MODULE 9 INNOVATIONS vs MODULE 8 (Attention Siamese U-Net)")
print("=" * 78)
print()
print(f"  {'Component':<28} {'Module 8':<25} {'Module 9'}")
print("  " + "-" * 74)
print(f"  {'Bottleneck':<28} {'DoubleConv':<25} DoubleConv + TransformerBottleneck")
print(f"  {'Global Context':<28} {'None':<25} Self-Attention (MHSA, 4 heads)")
print(f"  {'Skip Attention':<28} {'Spatial AttentionGate':<25} CBAM (Channel + Spatial)")
print(f"  {'Skip Features':<28} {'Temporal concat':<25} CBAM-attended temporal concat")
print(f"  {'Training':<28} {'Adam':<25} AdamW + CosineAnnealingLR")
print("  " + "-" * 74)
print()
print(f"[SUCCESS] Module 9 complete. Artifact saved to: '{OUT_FIG}'")
print("=" * 78)

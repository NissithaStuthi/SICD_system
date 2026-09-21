"""
Module 10 - Model Training: Complete Training Pipeline (Quick Test)
==================================================================

Tests all Module 10 topics in quick mode:
- Training/validation pipeline
- Batch size
- Learning rate
- Optimizers (Adam/AdamW)
- Learning-rate scheduling
- Early stopping
- Checkpointing
- GPU training
- Hyperparameter tuning
- Loss functions (BCE, Dice, Focal, BCE+Dice, etc.)
- Train multiple architectures
- Track training/validation loss
- Save the best model
- Analyze overfitting
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
from src.model import (
    HybridTransformerSiameseUNet,
    EarlyFusionUNet,
    SiameseCNN,
    SiameseUNetBasic,
    AttentionSiameseUNet,
)
from src.metrics import (
    BCEDiceLoss,
    BinaryCrossEntropyLoss,
    DiceLoss,
    FocalLoss,
    BCEFocalLoss,
    DiceFocalLoss,
    get_loss_function,
    calculate_metrics,
)
from src.training import (
    TrainingConfig,
    Trainer,
    get_optimizer,
    get_scheduler,
    create_data_loaders,
    compare_optimizers,
    run_hp_search,
    HPConfig,
    generate_hp_grid,
    sample_hp_random,
    get_gpu_info,
    estimate_batch_size,
    analyze_overfitting,
    train_multiple_architectures,
    plot_training_history,
    compare_loss_functions,
)

print("=" * 78)
print("  MODULE 10: MODEL TRAINING - Complete Pipeline (Quick Test)")
print("=" * 78)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {device}")

# Synthetic Dataset - smaller for speed
num_samples = 32
img_size = (64, 64)  # Smaller images for speed
C = 3
torch.manual_seed(42)
np.random.seed(42)
before_imgs = torch.randn(num_samples, C, *img_size)
after_imgs = torch.randn(num_samples, C, *img_size)
masks = (torch.rand(num_samples, 1, *img_size) > 0.5).float()
full_dataset = TensorDataset(before_imgs, after_imgs, masks)
train_ds = Subset(full_dataset, range(20))
val_ds = Subset(full_dataset, range(20, 30))

# ==============================================================================
# 1. Loss Functions
# ==============================================================================
print("\n[1] Loss Function Factory:")
for loss_name in ["bce", "dice", "focal", "bce_dice", "bce_focal", "dice_focal"]:
    loss_fn = get_loss_function(loss_name)
    print(f"  {loss_name:<12} -> {loss_fn.__class__.__name__}")

print("\n  Forward passes:")
test_logits = torch.randn(4, 1, 16, 16)
test_targets = torch.randint(0, 2, (4, 1, 16, 16)).float()
for loss_name in ["bce", "dice", "focal", "bce_dice"]:
    loss_fn = get_loss_function(loss_name)
    print(f"  {loss_name:<12} -> {loss_fn(test_logits, test_targets).item():.4f}")

# ==============================================================================
# 2. Batch Size
# ==============================================================================
print("\n[2] Batch Size Effects:")
for bs in [4, 8, 16]:
    tr, vl = create_data_loaders(train_ds, val_ds, batch_size=bs, num_workers=0)
    print(f"  BS {bs:2d} -> Train: {len(tr)}, Val: {len(vl)}")

# ==============================================================================
# 3. Optimizers
# ==============================================================================
print("\n[3] Optimizer Factory:")
model = HybridTransformerSiameseUNet(in_channels=3, out_channels=1).to(device)
for opt_name in ["adam", "adamw", "sgd", "rmsprop"]:
    config = TrainingConfig(batch_size=8, learning_rate=1e-4, optimizer_name=opt_name, num_epochs=1, device=device)
    print(f"  {opt_name.upper():<8} -> {get_optimizer(model, config).__class__.__name__}")

# ==============================================================================
# 4. LR Schedulers
# ==============================================================================
print("\n[4] LR Scheduler Factory:")
for sched_name in ["cosine", "step", "exponential", "reduce_on_plateau", "onecycle", "cosine_warm_restarts", "polynomial", "constant"]:
    config = TrainingConfig(batch_size=8, learning_rate=1e-4, optimizer_name="adamw", scheduler_name=sched_name, warmup_epochs=1, num_epochs=5, device=device)
    opt = get_optimizer(model, config)
    sched = get_scheduler(opt, config, config.num_epochs)
    print(f"  {sched_name:<25} -> {sched.__class__.__name__}")

# Warmup + Cosine
print("\n  Warmup+Cosine LR (5 epochs):")
config = TrainingConfig(batch_size=8, learning_rate=1e-4, optimizer_name="adamw", scheduler_name="cosine", warmup_epochs=2, num_epochs=5, device=device)
opt = get_optimizer(model, config)
sched = get_scheduler(opt, config, config.num_epochs)
lrs = [opt.param_groups[0]['lr']]
for _ in range(4):
    sched.step()
    lrs.append(opt.param_groups[0]['lr'])
print(f"    {[f'{lr:.2e}' for lr in lrs]}")

# ==============================================================================
# 5. Full Training with Scheduling + Early Stopping (2 epochs max)
# ==============================================================================
print("\n[5] Full Training Pipeline (2 epochs max):")
config = TrainingConfig(
    batch_size=8, learning_rate=1e-4, weight_decay=1e-4, optimizer_name="adamw", num_epochs=2,
    device=device, mixed_precision=False, gradient_clip_norm=1.0,
    checkpoint_dir="ckpt_full", log_interval=2,
    scheduler_name="cosine", scheduler_params={"eta_min": 1e-6}, warmup_epochs=1,
    early_stopping_patience=1, early_stopping_monitor="val_loss",
    save_top_k=2, save_last=True,
)
tr_loader, vl_loader = create_data_loaders(train_ds, val_ds, batch_size=8, num_workers=0)
trainer = Trainer(HybridTransformerSiameseUNet(in_channels=3, out_channels=1).to(device), tr_loader, vl_loader, BCEDiceLoss(), config, calculate_metrics)
history = trainer.fit()

print(f"  Epochs: {len(history['train_loss'])}")
print(f"  Train Loss: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
print(f"  Val Loss:   {history['val_loss'][0]:.4f} -> {history['val_loss'][-1]:.4f}")

# ==============================================================================
# 6. Overfitting Analysis
# ==============================================================================
print("\n[6] Overfitting Analysis:")
oa = analyze_overfitting(history)
print(f"  Status: {oa['status']} - {oa['message']}")
print(f"  Gap: {oa['avg_loss_gap']:.4f}")

# Synthetic overfitting test
fake_hist = {
    "train_loss": [0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08],
    "val_loss": [0.85, 0.7, 0.55, 0.45, 0.42, 0.45, 0.5, 0.58],
    "train_metrics": [{"iou": 0.3}, {"iou": 0.45}, {"iou": 0.55}, {"iou": 0.65}, {"iou": 0.72}, {"iou": 0.78}, {"iou": 0.82}, {"iou": 0.85}],
    "val_metrics": [{"iou": 0.25}, {"iou": 0.38}, {"iou": 0.48}, {"iou": 0.55}, {"iou": 0.55}, {"iou": 0.52}, {"iou": 0.48}, {"iou": 0.42}],
}
fa = analyze_overfitting(fake_hist)
print(f"  Synthetic test: {fa['status']} (gap={fa['avg_loss_gap']:.4f})")

# ==============================================================================
# 7. Checkpoints
# ==============================================================================
print("\n[7] Checkpoints:")
if os.path.exists("ckpt_full"):
    for f in sorted(os.listdir("ckpt_full")):
        print(f"  {f}")

# ==============================================================================
# 8. Resume Training
# ==============================================================================
print("\n[8] Resume Training:")
last_path = os.path.join("ckpt_full", "last_model.pth")
if os.path.exists(last_path):
    resume_model = HybridTransformerSiameseUNet(in_channels=3, out_channels=1).to(device)
    rt = Trainer(resume_model, tr_loader, vl_loader, BCEDiceLoss(), 
                 TrainingConfig(batch_size=8, learning_rate=1e-4, optimizer_name="adamw", num_epochs=3, device=device, checkpoint_dir="ckpt_full"), calculate_metrics)
    rt.load_checkpoint(last_path)
    print(f"  Resumed from epoch {rt.current_epoch}")
    rt.fit()

# ==============================================================================
# 9. Multi-Architecture Training (2 epochs each)
# ==============================================================================
print("\n[9] Multi-Architecture Training:")
model_configs = [
    {"name": "EarlyFusionUNet", "model_fn": EarlyFusionUNet, "model_kwargs": {"in_channels": 6, "out_channels": 1}},
    {"name": "SiameseCNN", "model_fn": SiameseCNN, "model_kwargs": {"in_channels": 3, "out_channels": 1}},
    {"name": "HybridTransformerSiameseUNet", "model_fn": HybridTransformerSiameseUNet, "model_kwargs": {"in_channels": 3, "out_channels": 1}},
]
mc = TrainingConfig(batch_size=4, learning_rate=1e-4, optimizer_name="adamw", num_epochs=2, device=device, checkpoint_dir="ckpt_multi", scheduler_name="cosine", warmup_epochs=0)
hp_tr, hp_vl = create_data_loaders(Subset(full_dataset, range(12)), Subset(full_dataset, range(12, 18)), batch_size=4, num_workers=0)
arch_results = train_multiple_architectures(model_configs, hp_tr, hp_vl, BCEDiceLoss(), mc, calculate_metrics, device)

# ==============================================================================
# 10. Plot History
# ==============================================================================
print("\n[10] Plot History:")
plot_training_history(history, save_path="training_hist_module10.png")
print("  Saved training_hist_module10.png")

# ==============================================================================
# 11. Loss Function Comparison (2 epochs)
# ==============================================================================
print("\n[11] Loss Function Comparison:")
loss_results = compare_loss_functions(
    lambda: HybridTransformerSiameseUNet(in_channels=3, out_channels=1),
    hp_tr, hp_vl, ["bce", "dice", "bce_dice"], lr=1e-4, epochs=2, device=device, metrics_fn=calculate_metrics
)

# ==============================================================================
# 12. GPU Info & Batch Size Estimation
# ==============================================================================
print("\n[12] GPU Info & Batch Size:")
gpu = get_gpu_info()
print(f"  CUDA: {gpu['cuda_available']}")
if device.type == "cuda":
    est = estimate_batch_size(HybridTransformerSiameseUNet(in_channels=3, out_channels=1), (3, 64, 64), "cuda")
    print(f"  Est. max BS: {est}")

# ==============================================================================
# 13. Hyperparameter Search
# ==============================================================================
print("\n[13] Hyperparameter Search:")
space = {"learning_rate": [1e-3, 1e-4], "batch_size": [4], "optimizer_name": ["adamw"], "weight_decay": [1e-4], "scheduler_name": ["cosine"]}
grid = generate_hp_grid(space)[:2]
hp_results = run_hp_search(
    lambda: HybridTransformerSiameseUNet(in_channels=3, out_channels=1),
    hp_tr, hp_vl, BCEDiceLoss(), grid, device, calculate_metrics, early_stopping_patience=1
)

print("\n" + "=" * 78)
print("  MODULE 10 QUICK TEST COMPLETE - ALL TOPICS WORKING")
print("=" * 78)
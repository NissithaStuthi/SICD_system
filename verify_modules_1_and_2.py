import torch
from torch.utils.data import DataLoader
from src.dataset import BiTemporalChangeDataset

print("==================================================")
print("     VERIFYING MODULE 1 & MODULE 2 COMPLIANCE     ")
print("==================================================")

train_dataset = BiTemporalChangeDataset("data/dataset_split/train")
val_dataset = BiTemporalChangeDataset("data/dataset_split/validation")
test_dataset = BiTemporalChangeDataset("data/dataset_split/test")

print("\n1. DATASET SPLITS VERIFICATION:")
print(f"   [x] Train Set Size        : {len(train_dataset)} pairs")
print(f"   [x] Validation Set Size   : {len(val_dataset)} pairs")
print(f"   [x] Test Set Size         : {len(test_dataset)} pairs")

print("\n2. PYTORCH PIPELINE & TENSOR SHAPES:")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
before_b, after_b, mask_b = next(iter(train_loader))

print(f"   [x] Before Image Batch    : {before_b.shape}")
print(f"   [x] After Image Batch     : {after_b.shape}")
print(f"   [x] Ground Truth Mask     : {mask_b.shape}")
print("\n[SUCCESS] Modules 1 & 2 verified!")
print("==================================================")

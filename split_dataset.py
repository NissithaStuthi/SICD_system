import os
import shutil
import random

# Source directories
t1_src = "data/raw/time1"
t2_src = "data/raw/time2"
mask_src = "data/raw/masks"

# Base target path matching curriculum layout
base_dir = "data/dataset_split"
splits = ["train", "validation", "test"]

for split in splits:
    for folder in ["before", "after", "masks"]:
        os.makedirs(os.path.join(base_dir, split, folder), exist_ok=True)

filenames = sorted(os.listdir(t1_src))
random.seed(42)
random.shuffle(filenames)

total = len(filenames)
train_end = int(total * 0.70)
val_end = train_end + int(total * 0.15)

train_files = filenames[:train_end]
val_files = filenames[train_end:val_end]
test_files = filenames[val_end:]

def copy_split(files, split_name):
    for f in files:
        shutil.copy(os.path.join(t1_src, f), os.path.join(base_dir, split_name, "before", f))
        shutil.copy(os.path.join(t2_src, f), os.path.join(base_dir, split_name, "after", f))
        shutil.copy(os.path.join(mask_src, f), os.path.join(base_dir, split_name, "masks", f))

copy_split(train_files, "train")
copy_split(val_files, "validation")
copy_split(test_files, "test")

print(f"Dataset split complete!")
print(f"Train samples      : {len(train_files)} pairs")
print(f"Validation samples : {len(val_files)} pairs")
print(f"Test samples       : {len(test_files)} pairs")

import os
import shutil

source_dir = "data/raw"
t1_target = os.path.abspath("data/raw/time1")
t2_target = os.path.abspath("data/raw/time2")
mask_target = os.path.abspath("data/raw/masks")

# Create directories if they do not exist
for path in [t1_target, t2_target, mask_target]:
    os.makedirs(path, exist_ok=True)

print("Mapping Kaggle dataset folders into project architecture...")

copied_count = {"t1": 0, "t2": 0, "mask": 0}

for root, dirs, files in os.walk(source_dir):
    abs_root = os.path.abspath(root)
    # Skip target directories during search to avoid recursive copies
    if abs_root in [t1_target, t2_target, mask_target]:
        continue

    for file in files:
        if file.endswith(('.png', '.jpg', '.tif', '.jpeg')):
            src_path = os.path.join(root, file)
            normalized_path = src_path.replace('\\', '/')

            # Identify LEVIR-CD folder markers (A = Time 1, B = Time 2, label/mask = Ground Truth)
            if '/A/' in normalized_path or '/train/A' in normalized_path:
                dst_path = os.path.join(t1_target, file)
                if os.path.abspath(src_path) != os.path.abspath(dst_path):
                    shutil.copy(src_path, dst_path)
                    copied_count["t1"] += 1
            elif '/B/' in normalized_path or '/train/B' in normalized_path:
                dst_path = os.path.join(t2_target, file)
                if os.path.abspath(src_path) != os.path.abspath(dst_path):
                    shutil.copy(src_path, dst_path)
                    copied_count["t2"] += 1
            elif '/label/' in normalized_path or '/label' in normalized_path or '/masks/' in normalized_path:
                dst_path = os.path.join(mask_target, file)
                if os.path.abspath(src_path) != os.path.abspath(dst_path):
                    shutil.copy(src_path, dst_path)
                    copied_count["mask"] += 1

print(f"Data mapping complete!")
print(f"Time 1 images mapped: {copied_count['t1']}")
print(f"Time 2 images mapped: {copied_count['t2']}")
print(f"Mask images mapped  : {copied_count['mask']}")

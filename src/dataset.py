import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class BiTemporalChangeDataset(Dataset):
    """
    Module 2 / 6 / 7 / 8 Dataset:
    Loads bi-temporal satellite image pairs (Before & After) and binary change masks.
    """
    def __init__(self, split_dir, img_size=(256, 256), transform=None):
        self.before_dir = os.path.join(split_dir, "before")
        self.after_dir = os.path.join(split_dir, "after")
        self.mask_dir = os.path.join(split_dir, "masks")
        self.img_size = img_size
        self.transform = transform
        
        if os.path.exists(self.before_dir):
            self.filenames = sorted([f for f in os.listdir(self.before_dir) if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        else:
            self.filenames = []

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        b_path = os.path.join(self.before_dir, filename)
        a_path = os.path.join(self.after_dir, filename)
        m_path = os.path.join(self.mask_dir, filename)

        before_img = cv2.cvtColor(cv2.imread(b_path), cv2.COLOR_BGR2RGB)
        after_img = cv2.cvtColor(cv2.imread(a_path), cv2.COLOR_BGR2RGB)
        
        mask_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            mask_img = np.zeros((before_img.shape[0], before_img.shape[1]), dtype=np.uint8)
        else:
            _, mask_img = cv2.threshold(mask_img, 127, 1, cv2.THRESH_BINARY)

        # Resize to target resolution if specified
        if self.img_size:
            before_img = cv2.resize(before_img, self.img_size, interpolation=cv2.INTER_LINEAR)
            after_img = cv2.resize(after_img, self.img_size, interpolation=cv2.INTER_LINEAR)
            mask_img = cv2.resize(mask_img, self.img_size, interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=before_img, image0=after_img, mask=mask_img)
            before_img, after_img, mask_img = augmented['image'], augmented['image0'], augmented['mask']

        before_tensor = torch.tensor(before_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        after_tensor = torch.tensor(after_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask_tensor = torch.tensor(mask_img, dtype=torch.float32).unsqueeze(0)
            
        return before_tensor, after_tensor, mask_tensor


class BiTemporalPatchClassificationDataset(Dataset):
    """
    Module 5 Dataset:
    Extracts image patches and assigns binary labels:
    - 1: Changed (if change mask within patch > threshold, e.g., > 5% changed pixels)
    - 0: Unchanged
    """
    def __init__(self, split_dir, patch_size=128, min_change_ratio=0.05, max_samples=200):
        self.patch_size = patch_size
        self.min_change_ratio = min_change_ratio
        
        b_dir = os.path.join(split_dir, "before")
        a_dir = os.path.join(split_dir, "after")
        m_dir = os.path.join(split_dir, "masks")
        
        files = sorted([f for f in os.listdir(b_dir) if not f.startswith('.')]) if os.path.exists(b_dir) else []
        self.patches = []

        for f in files[:max_samples]:
            b_img = cv2.cvtColor(cv2.imread(os.path.join(b_dir, f)), cv2.COLOR_BGR2RGB)
            a_img = cv2.cvtColor(cv2.imread(os.path.join(a_dir, f)), cv2.COLOR_BGR2RGB)
            m_img = cv2.imread(os.path.join(m_dir, f), cv2.IMREAD_GRAYSCALE)
            _, m_bin = cv2.threshold(m_img, 127, 1, cv2.THRESH_BINARY)

            h, w = b_img.shape[:2]
            # Extract patches
            for y in range(0, h - patch_size + 1, patch_size):
                for x in range(0, w - patch_size + 1, patch_size):
                    b_p = b_img[y:y+patch_size, x:x+patch_size]
                    a_p = a_img[y:y+patch_size, x:x+patch_size]
                    m_p = m_bin[y:y+patch_size, x:x+patch_size]

                    change_ratio = m_p.sum() / (patch_size * patch_size)
                    label = 1.0 if change_ratio >= min_change_ratio else 0.0

                    # 6-channel stacked patch
                    pair_stack = np.concatenate([b_p, a_p], axis=-1)
                    self.patches.append((pair_stack, label))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        pair_patch, label = self.patches[idx]
        patch_tensor = torch.tensor(pair_patch, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return patch_tensor, label_tensor

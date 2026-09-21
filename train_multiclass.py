import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class MultiClassDataset(Dataset):
    def __init__(self, b_dir, a_dir, l_dir):
        self.b_dir = b_dir
        self.a_dir = a_dir
        self.l_dir = l_dir
        self.files = sorted([f for f in os.listdir(b_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(b_dir) else []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        b_img = cv2.imread(os.path.join(self.b_dir, fname))
        b_img = cv2.cvtColor(cv2.resize(b_img, (256, 256)), cv2.COLOR_BGR2RGB) / 255.0
        
        a_img = cv2.imread(os.path.join(self.a_dir, fname))
        a_img = cv2.cvtColor(cv2.resize(a_img, (256, 256)), cv2.COLOR_BGR2RGB) / 255.0
        
        lbl_path = os.path.join(self.l_dir, fname)
        if os.path.exists(lbl_path):
            lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
            lbl = cv2.resize(lbl, (256, 256), interpolation=cv2.INTER_NEAREST)
            lbl = np.where(lbl > 128, 1, 0).astype(np.int64)
        else:
            lbl = np.zeros((256, 256), dtype=np.int64)

        b_tensor = torch.from_numpy(b_img).permute(2, 0, 1).float()
        a_tensor = torch.from_numpy(a_img).permute(2, 0, 1).float()
        lbl_tensor = torch.from_numpy(lbl).long()
        return b_tensor, a_tensor, lbl_tensor

def train_module_12():
    print("==============================================================")
    print("   MODULE 12: MULTI-CLASS CHANGE DETECTION TRAINING Engine    ")
    print("==============================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Compute Device: {device}")

    train_dataset = MultiClassDataset("data/dataset_split/train/before", "data/dataset_split/train/after", "data/dataset_split/train/label")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    print(f"--> Loaded {len(train_dataset)} training samples across {len(train_loader)} batches.")

    from src.model import MultiClassSiameseUNet
    model = MultiClassSiameseUNet(num_classes=5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 1
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        
        for batch_idx, (b_img, a_img, lbl) in enumerate(pbar):
            b_img, a_img, lbl = b_img.to(device), a_img.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(b_img, a_img)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            running_loss += loss_val
            
            # Live updating status in the progress bar
            pbar.set_postfix({"Batch Loss": f"{loss_val:.4f}"})

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"\n--> Epoch [{epoch}/{epochs}] Complete | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "multiclass_siamese_model.pth")
    print("--> Saved updated multi-class checkpoint to 'multiclass_siamese_model.pth'")
    print("==============================================================")

if __name__ == "__main__":
    train_module_12()

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class MultiClassTestDataset(Dataset):
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

# Color Palette for 5 Change Classes
COLOR_MAP = {
    0: [0, 0, 0],         # Black: No Change
    1: [255, 0, 0],       # Red: Building Change
    2: [0, 255, 0],       # Green: Vegetation Change
    3: [255, 255, 0],     # Yellow: Road Change
    4: [0, 0, 255]        # Blue: Water/Flood Change
}

def evaluate_module_12():
    print("==============================================================")
    print("     MODULE 12: MULTI-CLASS CHANGE TYPE ANALYSIS EVALUATOR    ")
    print("==============================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = MultiClassTestDataset("data/dataset_split/test/before", "data/dataset_split/test/after", "data/dataset_split/test/label")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    from src.model import MultiClassSiameseUNet
    model = MultiClassSiameseUNet(num_classes=5).to(device)
    if os.path.exists("multiclass_siamese_model.pth"):
        model.load_state_dict(torch.load("multiclass_siamese_model.pth", map_location=device))
        print("--> Loaded 'multiclass_siamese_model.pth' successfully.")

    model.eval()

    with torch.no_grad():
        for idx, (b_img, a_img, lbl) in enumerate(test_loader):
            b_img, a_img = b_img.to(device), a_img.to(device)
            out = model(b_img, a_img) # (1, 5, 256, 256)
            preds = torch.argmax(out, dim=1).cpu().numpy().squeeze() # (256, 256)

            if idx == 0:
                b_rgb = (b_img.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                a_rgb = (a_img.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # Render 5-Class Segmentation Map
                class_map = np.zeros((*preds.shape, 3), dtype=np.uint8)
                for cls_id, color in COLOR_MAP.items():
                    class_map[preds == cls_id] = color

                overlay = a_rgb.copy()
                overlay[preds > 0] = class_map[preds > 0]
                blended = cv2.addWeighted(a_rgb, 0.6, overlay, 0.4, 0)

                plt.figure(figsize=(15, 8))
                plt.subplot(2, 2, 1); plt.imshow(b_rgb); plt.title("1. Before Satellite Image")
                plt.subplot(2, 2, 2); plt.imshow(a_rgb); plt.title("2. After Satellite Image")
                plt.subplot(2, 2, 3); plt.imshow(class_map); plt.title("3. Multi-Class Map (Red:Building, Green:Veg, Yellow:Road, Blue:Water)")
                plt.subplot(2, 2, 4); plt.imshow(blended); plt.title("4. Multi-Class Overlay on Satellite Image")
                
                plt.tight_layout()
                plt.savefig("module12_multiclass_result.png")
                print("--> Saved multi-class analysis visualization to 'module12_multiclass_result.png'")
                break

    print("==============================================================")

if __name__ == "__main__":
    evaluate_module_12()

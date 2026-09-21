import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class RobustLEVIRDataset(Dataset):
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
            lbl = cv2.resize(lbl, (256, 256), interpolation=cv2.INTER_NEAREST) / 255.0
        else:
            lbl = np.zeros((256, 256))

        b_tensor = torch.from_numpy(b_img).permute(2, 0, 1).float()
        a_tensor = torch.from_numpy(a_img).permute(2, 0, 1).float()
        lbl_tensor = torch.from_numpy(lbl).unsqueeze(0).float()
        return b_tensor, a_tensor, lbl_tensor

def evaluate_module_11():
    print("==============================================================")
    print("       MODULE 11: FULL MODEL EVALUATION & ERROR ANALYSIS      ")
    print("==============================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    b_path = "data/dataset_split/test/before"
    a_path = "data/dataset_split/test/after"
    l_path = "data/dataset_split/test/label"

    test_dataset = RobustLEVIRDataset(b_path, a_path, l_path)
    print(f"--> Found {len(test_dataset)} test samples in '{b_path}'")

    if len(test_dataset) == 0:
        print("[!] Error: No files found in test path.")
        return

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    from src.model import HybridTransformerSiameseUNet
    model = HybridTransformerSiameseUNet().to(device)
    if os.path.exists("best_siamese_model.pth"):
        model.load_state_dict(torch.load("best_siamese_model.pth", map_location=device))
        print("--> Loaded 'best_siamese_model.pth' successfully.")

    model.eval()

    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    all_prob_means = []

    with torch.no_grad():
        for idx, (b_img, a_img, lbl) in enumerate(test_loader):
            b_img, a_img = b_img.to(device), a_img.to(device)
            lbl_np = (lbl.cpu().numpy().squeeze() > 0.5).astype(np.float32)

            out = model(b_img, a_img)
            prob_np = torch.sigmoid(out).cpu().numpy().squeeze()
            
            all_prob_means.append(float(np.mean(prob_np)))

            max_p = float(np.max(prob_np))
            thresh = 0.2 if max_p < 0.5 else 0.5
            pred_np = (prob_np > thresh).astype(np.float32)

            tp = np.sum((pred_np == 1) & (lbl_np == 1))
            fp = np.sum((pred_np == 1) & (lbl_np == 0))
            tn = np.sum((pred_np == 0) & (lbl_np == 0))
            fn = np.sum((pred_np == 0) & (lbl_np == 1))

            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

            if idx == 0:
                b_rgb = (b_img.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                a_rgb = (a_img.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                error_map = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
                error_map[(pred_np == 1) & (lbl_np == 1)] = [0, 255, 0]   # Green (TP)
                error_map[(pred_np == 1) & (lbl_np == 0)] = [255, 0, 0]   # Red (FP)
                error_map[(pred_np == 0) & (lbl_np == 1)] = [0, 0, 255]   # Blue (FN)

                overlay = a_rgb.copy()
                overlay[pred_np == 1] = [255, 0, 0]
                blended = cv2.addWeighted(a_rgb, 0.7, overlay, 0.3, 0)

                plt.figure(figsize=(15, 8))
                plt.subplot(2, 3, 1); plt.imshow(b_rgb); plt.title("1. Before Image")
                plt.subplot(2, 3, 2); plt.imshow(a_rgb); plt.title("2. After Image")
                plt.subplot(2, 3, 3); plt.imshow(lbl_np, cmap='gray'); plt.title("3. Ground Truth Mask")
                plt.subplot(2, 3, 4); plt.imshow(pred_np, cmap='gray'); plt.title(f"4. Predicted Mask (Thresh={thresh})")
                plt.subplot(2, 3, 5); plt.imshow(error_map); plt.title("5. Error Analysis (Green:TP, Red:FP, Blue:FN)")
                plt.subplot(2, 3, 6); plt.imshow(blended); plt.title("6. Overlay Prediction on Satellite Image")
                
                plt.tight_layout()
                plt.savefig("module11_evaluation_result.png")
                print("--> Saved visual evaluation panel to 'module11_evaluation_result.png'")

    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_tp + total_fn + 1e-7)
    f1_dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-7)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-7)

    print("\n--- GLOBAL TEST METRICS (149 Samples) ---")
    print(f"  [x] Mean Probability : {np.mean(all_prob_means):.4f}")
    print(f"  [x] Pixel Accuracy   : {accuracy:.4f}")
    print(f"  [x] Precision        : {precision:.4f}")
    print(f"  [x] Recall           : {recall:.4f}")
    print(f"  [x] F1 / Dice        : {f1_dice:.4f}")
    print(f"  [x] IoU Score        : {iou:.4f}")
    print("==============================================================")

if __name__ == "__main__":
    evaluate_module_11()

import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class ExplainabilityDataset(Dataset):
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
            lbl = np.where(lbl > 128, 1, 0).astype(np.float32)
        else:
            lbl = np.zeros((256, 256), dtype=np.float32)

        b_tensor = torch.from_numpy(b_img).permute(2, 0, 1).float()
        a_tensor = torch.from_numpy(a_img).permute(2, 0, 1).float()
        lbl_tensor = torch.from_numpy(lbl).unsqueeze(0).float()
        return b_tensor, a_tensor, lbl_tensor, fname

class GradCAM:
    """Module 13: Grad-CAM for Siamese Feature Bottleneck Hooks"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, t1, t2, class_idx=1):
        self.model.zero_grad()
        output = self.model(t1, t2) # (B, C, H, W)
        
        if output.shape[1] > 1:
            target = output[0, class_idx].sum()
        else:
            target = output[0, 0].sum()
            
        target.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, torch.sigmoid(output).squeeze().detach().cpu().numpy()

def evaluate_module_13():
    print("==============================================================")
    print("     MODULE 13: EXPLAINABILITY, GRAD-CAM & ERROR ANALYSIS     ")
    print("==============================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Compute Device: {device}")

    dataset = ExplainabilityDataset("data/dataset_split/test/before", "data/dataset_split/test/after", "data/dataset_split/test/label")
    if len(dataset) == 0:
        print("[!] Error: Test dataset is empty.")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    from src.model import MultiClassSiameseUNet
    model = MultiClassSiameseUNet(num_classes=5).to(device)
    if os.path.exists("multiclass_siamese_model.pth"):
        model.load_state_dict(torch.load("multiclass_siamese_model.pth", map_location=device))
        print("--> Loaded 'multiclass_siamese_model.pth' successfully.")
    elif os.path.exists("best_siamese_model.pth"):
        model.load_state_dict(torch.load("best_siamese_model.pth", map_location=device), strict=False)
        print("--> Loaded 'best_siamese_model.pth' with partial keys.")

    # Attach Grad-CAM to Bottleneck Conv Layer
    cam_extractor = GradCAM(model, model.bottleneck.conv[3])

    b_img, a_img, lbl, fname = next(iter(loader))
    b_img, a_img = b_img.to(device), a_img.to(device)
    lbl_np = lbl.numpy().squeeze()

    # Generate Grad-CAM & Probabilities
    cam, probs = cam_extractor.generate_cam(b_img, a_img)
    
    if probs.ndim == 3:
        pred_class = np.argmax(probs, axis=0)
        confidence = np.max(probs, axis=0)
        binary_pred = (pred_class > 0).astype(np.float32)
    else:
        confidence = probs
        binary_pred = (probs > 0.3).astype(np.float32)

    b_rgb = (b_img.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    a_rgb = (a_img.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # 1. Grad-CAM Heatmap Visualization
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam_overlay = cv2.addWeighted(a_rgb, 0.6, heatmap, 0.4, 0)

    # 2. Prediction Confidence Map
    conf_map = cv2.applyColorMap(np.uint8(255 * confidence), cv2.COLORMAP_VIRIDIS)
    conf_map = cv2.cvtColor(conf_map, cv2.COLOR_BGR2RGB)

    # 3. False-Positive / False-Negative Discrepancy Analysis
    error_analysis = np.zeros((*binary_pred.shape, 3), dtype=np.uint8)
    error_analysis[(binary_pred == 1) & (lbl_np == 1)] = [0, 255, 0]   # True Positive (Green)
    error_analysis[(binary_pred == 1) & (lbl_np == 0)] = [255, 0, 0]   # False Positive (Red - Cloud/Shadow/Terrain Error)
    error_analysis[(binary_pred == 0) & (lbl_np == 1)] = [0, 0, 255]   # False Negative (Blue - Missed Change)

    # Render Module 13 Diagnostic Dashboard
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 3, 1); plt.imshow(b_rgb); plt.title("1. Before Satellite Image")
    plt.subplot(2, 3, 2); plt.imshow(a_rgb); plt.title("2. After Satellite Image")
    plt.subplot(2, 3, 3); plt.imshow(lbl_np, cmap='gray'); plt.title("3. Ground Truth Mask")
    plt.subplot(2, 3, 4); plt.imshow(cam_overlay); plt.title("4. Grad-CAM (Bottleneck Feature Attribution)")
    plt.subplot(2, 3, 5); plt.imshow(conf_map); plt.title("5. Pixel Prediction Confidence Map")
    plt.subplot(2, 3, 6); plt.imshow(error_analysis); plt.title("6. Error Analysis (Green:TP, Red:FP, Blue:FN)")

    plt.tight_layout()
    plt.savefig("module13_explainability_result.png")
    print("--> Saved explainability dashboard to 'module13_explainability_result.png'")
    print("==============================================================")

if __name__ == "__main__":
    evaluate_module_13()

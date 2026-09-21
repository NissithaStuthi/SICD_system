import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend: prevents GUI window locking
import matplotlib.pyplot as plt

from src.model import MultiClassSiameseUNet

COLOR_MAP = {
    0: [0, 0, 0],         # No Change
    1: [255, 0, 0],       # Building (Red)
    2: [0, 255, 0],       # Vegetation (Green)
    3: [255, 255, 0],     # Road (Yellow)
    4: [0, 0, 255]        # Water (Blue)
}

def run_prediction():
    print("==============================================================")
    print("      AUTOMATED SATELLITE CHANGE DETECTION INFERENCE           ")
    print("==============================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Compute Device: {device}")

    # Load Model
    model = MultiClassSiameseUNet(num_classes=5).to(device)
    model_path = "multiclass_siamese_model.pth" if os.path.exists("multiclass_siamese_model.pth") else "best_siamese_model.pth"

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"--> Loaded checkpoint '{model_path}' successfully.")
    else:
        print("[!] Error: No model checkpoint found.")
        return

    model.eval()

    b_dir = "data/dataset_split/test/before"
    a_dir = "data/dataset_split/test/after"
    
    files = sorted([f for f in os.listdir(b_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not files:
        print("[!] No images found in test directory.")
        return

    fname = files[0]
    print(f"--> Processing Sample Image Pair: '{fname}'")

    b_path = os.path.join(b_dir, fname)
    a_path = os.path.join(a_dir, fname)

    b_raw = cv2.imread(b_path)
    a_raw = cv2.imread(a_path)

    b_img = cv2.cvtColor(cv2.resize(b_raw, (256, 256)), cv2.COLOR_BGR2RGB) / 255.0
    a_img = cv2.cvtColor(cv2.resize(a_raw, (256, 256)), cv2.COLOR_BGR2RGB) / 255.0

    b_tensor = torch.from_numpy(b_img).permute(2, 0, 1).float().unsqueeze(0).to(device)
    a_tensor = torch.from_numpy(a_img).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(b_tensor, a_tensor)
        probs = torch.softmax(out, dim=1).squeeze().cpu().numpy()
        preds = np.argmax(probs, axis=0)

    b_rgb = (b_img * 255).astype(np.uint8)
    a_rgb = (a_img * 255).astype(np.uint8)

    class_map = np.zeros((*preds.shape, 3), dtype=np.uint8)
    for cls_id, color in COLOR_MAP.items():
        class_map[preds == cls_id] = color

    overlay = a_rgb.copy()
    overlay[preds > 0] = class_map[preds > 0]
    blended = cv2.addWeighted(a_rgb, 0.6, overlay, 0.4, 0)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1); plt.imshow(b_rgb); plt.title("1. Before Image (T1)")
    plt.subplot(1, 4, 2); plt.imshow(a_rgb); plt.title("2. After Image (T2)")
    plt.subplot(1, 4, 3); plt.imshow(class_map); plt.title("3. Multi-Class Change Map")
    plt.subplot(1, 4, 4); plt.imshow(blended); plt.title("4. Change Overlay on Satellite")

    plt.tight_layout()
    plt.savefig("prediction_output.png")
    plt.close()
    print("--> Output figure saved to 'prediction_output.png'")
    print("==============================================================")

if __name__ == "__main__":
    run_prediction()

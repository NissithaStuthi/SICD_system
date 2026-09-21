import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import MultiClassSiameseUNet

COLOR_MAP = {
    0: [0, 0, 0],         # No Change
    1: [255, 0, 0],       # Building (Red)
    2: [0, 255, 0],       # Vegetation (Green)
    3: [255, 255, 0],     # Road (Yellow)
    4: [0, 0, 255]        # Water (Blue)
}

# 1. Automatic Image Registration (ORB + Homography)
def register_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return img2

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        return img2

    height, width, _ = img1.shape
    registered_img2 = cv2.warpPerspective(img2, H, (width, height))
    return registered_img2

# 2. Tiled Batch Inference & Reconstruction
def run_tiled_inference(model, img1, img2, tile_size=256, device="cpu"):
    h, w, c = img1.shape
    # Pad image to multiples of tile_size
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size

    img1_padded = np.pad(img1, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    img2_padded = np.pad(img2, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    full_h, full_w, _ = img1_padded.shape
    prediction_map = np.zeros((full_h, full_w), dtype=np.int64)

    for y in range(0, full_h, tile_size):
        for x in range(0, full_w, tile_size):
            t1 = img1_padded[y:y+tile_size, x:x+tile_size] / 255.0
            t2 = img2_padded[y:y+tile_size, x:x+tile_size] / 255.0

            t1_tensor = torch.from_numpy(t1).permute(2, 0, 1).float().unsqueeze(0).to(device)
            t2_tensor = torch.from_numpy(t2).permute(2, 0, 1).float().unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(t1_tensor, t2_tensor)
                probs = torch.softmax(out, dim=1).squeeze().cpu().numpy()
                tile_pred = np.argmax(probs, axis=0)

            prediction_map[y:y+tile_size, x:x+tile_size] = tile_pred

    return prediction_map[:h, :w]

# 3. Morphological Post-Processing
def post_process(pred_map):
    binary_mask = (pred_map > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Remove isolated salt-and-pepper noise
    cleaned_binary = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # Fill small holes inside detected change regions
    cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, kernel)
    return pred_map * cleaned_binary

def run_module_14():
    print("==============================================================")
    print("     MODULE 14: REAL-WORLD END-TO-END INFERENCE PIPELINE       ")
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
        print("[!] Error: No checkpoint found.")
        return

    model.eval()

    b_dir = "data/dataset_split/test/before"
    a_dir = "data/dataset_split/test/after"
    files = sorted([f for f in os.listdir(b_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if not files:
        print("[!] Error: No test images found.")
        return

    fname = files[0]
    print(f"--> Stage 1: Loading Raw Satellite Images ('{fname}')")
    img1_raw = cv2.cvtColor(cv2.imread(os.path.join(b_dir, fname)), cv2.COLOR_BGR2RGB)
    img2_raw = cv2.cvtColor(cv2.imread(os.path.join(a_dir, fname)), cv2.COLOR_BGR2RGB)

    print("--> Stage 2: Performing Automatic Image Registration (ORB Feature Matching)")
    img2_registered = register_images(img1_raw, img2_raw)

    print("--> Stage 3: Executing Tiled Deep Learning Inference")
    raw_pred = run_tiled_inference(model, img1_raw, img2_registered, tile_size=256, device=device)

    print("--> Stage 4: Morphological Post-Processing & Surface Area Analytics")
    clean_pred = post_process(raw_pred)

    total_pixels = clean_pred.size
    changed_pixels = np.sum(clean_pred > 0)
    changed_percentage = (changed_pixels / total_pixels) * 100

    print("\n--- SURFACE AREA ANALYTICS METRICS ---")
    print(f"  [x] Total Image Pixels  : {total_pixels:,} px")
    print(f"  [x] Changed Pixels      : {changed_pixels:,} px")
    print(f"  [x] Total Area Change   : {changed_percentage:.2f}%")

    # Build Visualization Dashboard
    class_map = np.zeros((*clean_pred.shape, 3), dtype=np.uint8)
    for cls_id, color in COLOR_MAP.items():
        class_map[clean_pred == cls_id] = color

    overlay = img2_registered.copy()
    overlay[clean_pred > 0] = class_map[clean_pred > 0]
    blended = cv2.addWeighted(img2_registered, 0.6, overlay, 0.4, 0)

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 2, 1); plt.imshow(img1_raw); plt.title("1. Preprocessed Date 1 Satellite Image")
    plt.subplot(2, 2, 2); plt.imshow(img2_registered); plt.title("2. Registered Date 2 Satellite Image")
    plt.subplot(2, 2, 3); plt.imshow(class_map); plt.title(f"3. Change Map (Change: {changed_percentage:.2f}%)")
    plt.subplot(2, 2, 4); plt.imshow(blended); plt.title("4. Post-Processed Overlay")

    plt.tight_layout()
    plt.savefig("module14_pipeline_result.png")
    plt.close()
    print("\n--> Pipeline complete! Output saved to 'module14_pipeline_result.png'")
    print("==============================================================")

if __name__ == "__main__":
    run_module_14()

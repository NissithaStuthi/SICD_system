import cv2
import numpy as np

def compute_absolute_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Computes single-channel grayscale absolute difference between two images.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
    return cv2.absdiff(gray1, gray2)

def compute_cva(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Computes multi-spectral / RGB Change Vector Analysis (CVA) magnitude.
    ||I_t2 - I_t1|| = sqrt(sum((I_t2_c - I_t1_c)^2))
    """
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    magnitude = np.sqrt(np.sum(diff ** 2, axis=-1))
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Alias for backwards compatibility
change_vector_analysis = compute_cva

def apply_threshold_and_morphology(diff_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Applies Otsu automated thresholding followed by Morphological Opening & Closing for noise removal.
    """
    _, binary_mask = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    return clean_mask

# Alias for backwards compatibility
generate_baseline_mask = apply_threshold_and_morphology

def evaluate_traditional_pair(img1: np.ndarray, img2: np.ndarray, gt_mask: np.ndarray):
    """
    Runs the full traditional change detection pipeline on an image pair and evaluates against ground truth.
    """
    diff_map = compute_cva(img1, img2)
    pred_mask = apply_threshold_and_morphology(diff_map) // 255
    gt_bin = (gt_mask > 0).astype(np.uint8) if gt_mask.max() > 1 else gt_mask.astype(np.uint8)

    tp = np.logical_and(pred_mask == 1, gt_bin == 1).sum()
    fp = np.logical_and(pred_mask == 1, gt_bin == 0).sum()
    fn = np.logical_and(pred_mask == 0, gt_bin == 1).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)

    return {
        "diff_map": diff_map,
        "pred_mask": pred_mask,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou
    }

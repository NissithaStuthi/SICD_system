import cv2
import numpy as np

def match_histograms(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Matches color histogram of source image to reference image to normalize lighting and atmospheric conditions.
    """
    matched = np.zeros_like(source)
    for i in range(3):
        s_hist, _ = np.histogram(source[:, :, i], 256, [0, 256])
        r_hist, _ = np.histogram(reference[:, :, i], 256, [0, 256])
        
        s_cdf = np.cumsum(s_hist) / (s_hist.sum() + 1e-6)
        r_cdf = np.cumsum(r_hist) / (r_hist.sum() + 1e-6)
        
        lookup = np.interp(s_cdf, r_cdf, np.arange(256))
        matched[:, :, i] = cv2.LUT(source[:, :, i], lookup.astype(np.uint8))
    return matched

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) for local contrast enhancement.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 2 or image.shape[2] == 1:
        return clahe.apply(image)
    
    # Apply on LAB Luminance channel for RGB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def denoise_bilateral(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Bilateral filter for edge-preserving noise reduction.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def normalize_minmax(image: np.ndarray) -> np.ndarray:
    """
    Normalizes image pixel values into [0.0, 1.0] range.
    """
    return (image.astype(np.float32) - image.min()) / (image.max() - image.min() + 1e-6)

def extract_tiles(image: np.ndarray, tile_size: int = 256, stride: int = 256):
    """
    Splits high-resolution satellite rasters into uniform patch tiles.
    """
    h, w = image.shape[:2]
    tiles = []
    positions = []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
            positions.append((y, x))
    return tiles, positions

def align_images_ecc(source: np.ndarray, reference: np.ndarray, max_iters: int = 50) -> np.ndarray:
    """
    Aligns source image to reference image using Enhanced Correlation Coefficient (ECC) maximization.
    """
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY) if len(reference.shape) == 3 else reference
    src_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY) if len(source.shape) == 3 else source
    
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters, 1e-4)
    try:
        _, warp_matrix = cv2.findTransformECC(ref_gray, src_gray, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        aligned = cv2.warpAffine(source, warp_matrix, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned
    except Exception:
        # If ECC doesn't converge, return original
        return source

"""
01_preprocess_and_slice.py
==========================
Step 1 in Sonar Debris Detection Pipeline:
- Reads raw Side-Scan Sonar (SSS) data (.xtf, .tif, .png, etc.) from 'raw_data/'
- Applies acoustic corrections: Percentile Normalization, Despeckle Filtering, and CLAHE
- Slices large waterfall images into overlapping patches (e.g., 640x640)
- Saves slices into 'data/images/train' and 'data/images/val'
"""

import os
import glob
import random
import cv2
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw_data"
IMG_TRAIN_DIR = BASE_DIR / "data" / "images" / "train"
IMG_VAL_DIR = BASE_DIR / "data" / "images" / "val"
LBL_TRAIN_DIR = BASE_DIR / "data" / "labels" / "train"
LBL_VAL_DIR = BASE_DIR / "data" / "labels" / "val"

# Parameters
TILE_SIZE = 640       # YOLO input size (640x640)
OVERLAP_RATIO = 0.20  # 20% overlap between adjacent patches
VAL_SPLIT_RATIO = 0.2 # 20% validation split


def load_raw_sonar(file_path: Path) -> np.ndarray:
    """Loads sonar image from raster or .xtf format."""
    ext = file_path.suffix.lower()

    if ext == ".xtf":
        try:
            import pyxtf
            (header, packets) = pyxtf.xtf_read(str(file_path))
            # Extract port and starboard channels if available
            port = pyxtf.concatenate_channel(packets, file_header=header, channel=0)
            stbd = pyxtf.concatenate_channel(packets, file_header=header, channel=1)
            # Combine port and starboard horizontally
            combined = np.hstack([np.fliplr(port), stbd])
            return combined.astype(np.float32)
        except ImportError:
            raise ImportError("pyxtf is required to read .xtf files. Run: pip install pyxtf")
    else:
        # Standard raster (TIF, PNG, JPG)
        # Load unchanged (handles 8-bit or 16-bit grayscale)
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {file_path}")
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32)


def enhance_sonar_patch(sonar_array: np.ndarray) -> np.ndarray:
    """
    Applies acoustic normalization:
    1. Percentile contrast stretching (eliminates acoustic sensor spikes)
    2. Bilateral filter (despeckling while preserving debris edges)
    3. CLAHE (corrects slant-range acoustic attenuation)
    """
    p_low, p_high = np.percentile(sonar_array, (1, 99))
    if p_high == p_low:
        clipped = np.zeros_like(sonar_array, dtype=np.uint8)
    else:
        clipped = np.clip(sonar_array, p_low, p_high)
        clipped = ((clipped - p_low) / (p_high - p_low + 1e-6) * 255.0).astype(np.uint8)

    # Speckle reduction
    denoised = cv2.bilateralFilter(clipped, d=5, sigmaColor=50, sigmaSpace=50)

    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Convert to 3-channel for standard YOLOv8 models
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def slice_sonar_waterfall(sonar_img: np.ndarray, tile_size=640, overlap=0.2):
    """Slices waterfall image using an overlapping sliding window."""
    h, w = sonar_img.shape[:2]
    step = int(tile_size * (1 - overlap))
    tiles = []

    # If image is smaller than tile_size, pad it
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h > 0 or pad_w > 0:
        sonar_img = cv2.copyMakeBorder(sonar_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        h, w = sonar_img.shape[:2]

    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            patch = sonar_img[y:y + tile_size, x:x + tile_size]
            tiles.append((patch, x, y))

    return tiles


def main():
    for d in [IMG_TRAIN_DIR, IMG_VAL_DIR, LBL_TRAIN_DIR, LBL_VAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    supported_extensions = ["*.xtf", "*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]
    raw_files = []
    for ext in supported_extensions:
        raw_files.extend(RAW_DIR.glob(ext))

    if not raw_files:
        print(f"[!] No raw files found in: {RAW_DIR}")
        print(f"    Please place your SSS files (.xtf, .tif, .png, etc.) inside: {RAW_DIR}")
        return

    print(f"[*] Found {len(raw_files)} raw sonar survey files. Processing...")
    total_saved = 0

    for file_path in raw_files:
        print(f" -> Loading & preprocessing: {file_path.name}")
        try:
            raw_sonar = load_raw_sonar(file_path)
        except Exception as e:
            print(f"[!] Error loading {file_path.name}: {e}")
            continue

        enhanced = enhance_sonar_patch(raw_sonar)
        tiles = slice_sonar_waterfall(enhanced, tile_size=TILE_SIZE, overlap=OVERLAP_RATIO)
        print(f"    Extracted {len(tiles)} slices ({TILE_SIZE}x{TILE_SIZE})")

        stem = file_path.stem
        for patch, x, y in tiles:
            # Deterministic/random split
            is_val = (random.random() < VAL_SPLIT_RATIO)
            target_dir = IMG_VAL_DIR if is_val else IMG_TRAIN_DIR

            out_filename = f"{stem}_x{x}_y{y}.png"
            cv2.imwrite(str(target_dir / out_filename), patch)
            total_saved += 1

    print(f"\n[+] Slicing complete! Total patches saved: {total_saved}")
    print(f"    Train images: {len(list(IMG_TRAIN_DIR.glob('*.png')))} in {IMG_TRAIN_DIR}")
    print(f"    Val images:   {len(list(IMG_VAL_DIR.glob('*.png')))} in {IMG_VAL_DIR}")
    print("\n[NEXT STEP]:")
    print("1. Open an annotation tool (e.g., AnyLabeling, CVAT, or Label Studio).")
    print("2. Annotate debris items (highlight + acoustic shadow).")
    print(f"3. Export YOLO format .txt labels into:")
    print(f"     Train labels -> {LBL_TRAIN_DIR}")
    print(f"     Val labels   -> {LBL_VAL_DIR}")
    print("4. Then run: python 02_train.py")


if __name__ == "__main__":
    main()

"""
train_fls.py
============
YOLOv8 Training specifically tailored for the Marine Debris FLS Dataset
(Forward-Looking Sonar / Acoustic Camera imagery)

- Uses YOLOv8s pretrained weights
- Automatically uses NVIDIA GPU when CUDA is available
- Uses batch size 2 for RTX 3050 6GB stability
- Uses workers=0 to avoid Windows multiprocessing/paging-file issues
- Saves the best model as models/best_fls.pt
"""

import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_YAML = BASE_DIR / "data" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

MODEL_VARIANT = "yolov8s.pt"

IMAGE_SIZE = 640
EPOCHS = 100

# RTX 3050 6GB:
# Batch 8 caused CUDA OOM.
# Batch 2 is safer.
BATCH_SIZE = 2

# Automatically use NVIDIA GPU if available
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Windows multiprocessing can cause paging-file issues.
WORKERS = 0


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("Marine Debris FLS YOLOv8 Training")
    print("=" * 70)

    print(f"[*] Python/PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"[*] Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[*] CUDA runtime: {torch.version.cuda}")

    print(f"[*] Model: {MODEL_VARIANT}")
    print(f"[*] Image size: {IMAGE_SIZE}")
    print(f"[*] Batch size: {BATCH_SIZE}")
    print(f"[*] Epochs: {EPOCHS}")
    print(f"[*] Workers: {WORKERS}")
    print(f"[*] Dataset config: {DATA_YAML}")

    # --------------------------------------------------------
    # Check dataset configuration
    # --------------------------------------------------------

    if not DATA_YAML.exists():
        print(f"[!] Dataset configuration not found:")
        print(f"    {DATA_YAML}")
        return

    # --------------------------------------------------------
    # Load pretrained YOLOv8 model
    # --------------------------------------------------------

    print("\n[*] Initializing YOLOv8 model...")

    model = YOLO(MODEL_VARIANT)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n[*] Starting FLS training...")
    print("=" * 70)

    results = model.train(

        # Dataset
        data=str(DATA_YAML),

        # Training duration
        epochs=EPOCHS,

        # Image resolution
        imgsz=IMAGE_SIZE,

        # GPU memory-safe batch size
        batch=BATCH_SIZE,

        # NVIDIA GPU / CPU
        device=DEVICE,

        # Windows-safe dataloader
        workers=WORKERS,

        # Output directory
        project="runs/detect",
        name="fls_debris_run",
        exist_ok=True,

        # ----------------------------------------------------
        # SONAR-SPECIFIC AUGMENTATION
        # ----------------------------------------------------

        # Sonar imagery is monochrome
        hsv_h=0.0,
        hsv_s=0.0,

        # Simulate acoustic intensity/gain variation
        hsv_v=0.3,

        # Horizontal symmetry is reasonable for FLS
        fliplr=0.5,

        # Vertical flipping is disabled because
        # acoustic shadows have directional meaning
        flipud=0.0,

        # Small orientation variation
        degrees=10.0,

        # Object scale variation
        scale=0.2,

        # Context mixing
        mosaic=1.0,

        # Disable mosaic near the end for cleaner localization
        close_mosaic=10,

        # ----------------------------------------------------
        # TRAINING CONTROL
        # ----------------------------------------------------

        # Stop if validation performance stops improving
        patience=20,

        # Save checkpoints
        save=True,
    )

    # ========================================================
    # FIND BEST WEIGHTS
    # ========================================================

    best_weights = Path(results.save_dir) / "weights" / "best.pt"

    print("\n" + "=" * 70)
    print("Training finished.")
    print("=" * 70)

    print(f"[*] Training output:")
    print(f"    {results.save_dir}")

    if best_weights.exists():

        target_path = MODELS_DIR / "best_fls.pt"

        shutil.copy2(best_weights, target_path)

        print("\n[+] BEST MODEL SAVED")
        print(f"    {target_path}")

        print("\n[NEXT STEP]")
        print("Run FLS inference using:")
        print()
        print("    python predict_fls.py --input path_to_fls_image.png")

    else:

        print("\n[!] WARNING")
        print("Could not find best.pt.")
        print(f"Expected location:")
        print(f"    {best_weights}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
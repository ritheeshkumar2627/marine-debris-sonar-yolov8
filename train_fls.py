"""
train_fls.py
============
YOLOv8 Training specifically tailored for the Marine Debris FLS Dataset
(Forward-Looking Sonar / Acoustic Camera imagery)
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "data" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

import torch

# Training Hyperparameters
# yolov8s or yolov8m works best for acoustic features (highlight + shadow)
MODEL_VARIANT = "yolov8s.pt"
IMAGE_SIZE = 640   # 640 or 960 (ARIS 3000 frames are ~960x512)
EPOCHS = 100
BATCH_SIZE = 16
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():
    print(f"[*] Initializing YOLOv8 model ({MODEL_VARIANT})...")
    model = YOLO(MODEL_VARIANT)

    print(f"[*] Training on Marine Debris FLS dataset with config: {DATA_YAML}")
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project="runs/detect",
        name="fls_debris_run",
        exist_ok=True,

        # Acoustic Camera (FLS) Specific Augmentation:
        hsv_h=0.0,        # Sonar is monochrome; disable hue jitter
        hsv_s=0.0,        # Disable saturation jitter
        hsv_v=0.3,        # Brightness variation (acoustic gain shifts)
        fliplr=0.5,       # Left/Right horizontal flip (swath symmetry)
        flipud=0.0,       # Disable vertical flip (acoustic shadows always point away from sensor)
        degrees=10.0,     # Small angle rotation
        scale=0.2,        # Scale variation
        mosaic=1.0,       # Context mixing
        close_mosaic=10,  # Turn off mosaic in final 10 epochs for crisp bounding boxes

        patience=20,
        save=True
    )

    # Automatically save the best weights to models/best_fls.pt
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        target_path = MODELS_DIR / "best_fls.pt"
        shutil.copy(best_weights, target_path)
        print(f"\n[+] FLS Training complete!")
        print(f"    Best weights saved to: {target_path}")
        print("\n[NEXT STEP]: Test inference using:")
        print("    python predict_fls.py --input path_to_fls_image.png")
    else:
        print(f"[!] Warning: Could not find best.pt in {results.save_dir}")


if __name__ == "__main__":
    main()

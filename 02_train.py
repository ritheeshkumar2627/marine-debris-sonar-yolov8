"""
02_train.py
===========
Step 2 in Sonar Debris Detection Pipeline:
- Configures and trains YOLOv8 on the sliced sonar dataset
- Applies acoustic-tailored data augmentations (monochrome-safe)
- Saves best weights to 'models/best.pt'
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "data" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters
MODEL_VARIANT = "yolov8s.pt"  # Small variant: good balance for small acoustic targets
IMAGE_SIZE = 640
EPOCHS = 100
BATCH_SIZE = 16
DEVICE = 0  # GPU 0, or set to 'cpu' if no CUDA GPU available


def main():
    print(f"[*] Initializing YOLOv8 model ({MODEL_VARIANT})...")
    model = YOLO(MODEL_VARIANT)

    print(f"[*] Starting fine-tuning using config: {DATA_YAML}")
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project="runs/detect",
        name="sonar_debris_run",
        exist_ok=True,
        
        # Sonar-Specific Augmentation Strategy:
        # Acoustic backscatter has NO hue or saturation.
        hsv_h=0.0,       # Turn off hue jitter
        hsv_s=0.0,       # Turn off saturation jitter
        hsv_v=0.3,       # Moderate brightness / intensity gain variation
        
        # Spatial Augmentation:
        fliplr=0.5,      # Horizontal flip (port <-> starboard symmetry)
        flipud=0.5,      # Vertical flip (along-track survey direction)
        degrees=10.0,    # Slight rotation
        scale=0.2,       # Scale jitter (objects at varying acoustic altitudes)
        mosaic=1.0,      # Mix context patches
        close_mosaic=10, # Disable mosaic during final 10 epochs for stable convergence
        
        # Early Stopping
        patience=20,
        save=True
    )

    # Copy the best weights to models/best.pt for simple inference reference
    best_weights_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights_path.exists():
        target_path = MODELS_DIR / "best.pt"
        shutil.copy(best_weights_path, target_path)
        print(f"\n[+] Training complete!")
        print(f"    Best model weights saved to: {target_path}")
        print("\n[NEXT STEP]:")
        print("Run inference on full-scale raw sonar waterfall surveys using:")
        print("    python 03_predict_sahi.py --input path_to_test_sonar.png")
    else:
        print(f"[!] Warning: Could not locate best.pt in {results.save_dir}")


if __name__ == "__main__":
    main()

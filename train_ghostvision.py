"""
train_ghostvision.py
====================
YOLOv8 Training specifically tailored for the GhostVision Side-Scan Sonar (SSS) dataset
(Derelict Crab Pot & Marine Debris Detection, PING Ecosystem)
"""

import shutil
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
GV_DATA_DIR = BASE_DIR / "data_ghostvision"
DATA_YAML = GV_DATA_DIR / "data.yaml"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
MODEL_VARIANT = "yolov8s.pt"  # Pretrained YOLOv8 small
IMAGE_SIZE = 640
EPOCHS = 100
BATCH_SIZE = 16
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 70)
    print("Training YOLOv8 on GhostVision Side-Scan Sonar (SSS) Dataset")
    print("=" * 70)

    if not DATA_YAML.exists():
        print(f"[!] Config not found at: {DATA_YAML}")
        print("    Please run 'python download_and_prep_ghostvision.py' first!")
        return

    # Dynamically ensure data.yaml has current runtime's absolute path (works on Windows, Linux & Colab)
    with open(DATA_YAML, "r") as f:
        config = yaml.safe_load(f)
    config["path"] = str(GV_DATA_DIR).replace("\\", "/")
    with open(DATA_YAML, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"[*] Initializing model ({MODEL_VARIANT})...")
    model = YOLO(MODEL_VARIANT)

    print(f"[*] Training on {DEVICE}...")
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project="runs/detect",
        name="ghostvision_sss_run",
        exist_ok=True,

        # Side-Scan Sonar Acoustic Augmentations:
        hsv_h=0.0,        # Sonar is monochrome; no hue variation
        hsv_s=0.0,        # No saturation variation
        hsv_v=0.3,        # Brightness / gain variation
        fliplr=0.5,       # Horizontal flip (port <-> starboard range symmetry)
        flipud=0.5,       # Vertical flip (along-track tow direction)
        degrees=10.0,     # Small angle rotation
        scale=0.2,        # Scale variation
        mosaic=1.0,       # Context mixing
        close_mosaic=10,  # Turn off mosaic in last 10 epochs

        patience=20,
        save=True
    )

    # Save best checkpoint to models/best_ghostvision.pt
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        target_path = MODELS_DIR / "best_ghostvision.pt"
        shutil.copy(best_weights, target_path)
        print("\n" + "=" * 70)
        print(f"[+] GhostVision SSS Training Complete!")
        print(f"    Best weights saved to: {target_path}")
        print("=" * 70)
        print("\n[NEXT STEP]: Run sliced inference on full-scale raw SSS surveys with:")
        print(f"    python 03_predict_sahi.py --input path_to_raw_sss.png --model {target_path}")
    else:
        print(f"[!] Warning: Could not locate best.pt in {results.save_dir}")


if __name__ == "__main__":
    main()

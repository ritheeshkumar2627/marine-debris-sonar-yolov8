"""
predict_fls.py
==============
Inference script for Marine Debris detection on Forward-Looking Sonar (FLS) frames or video.
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = BASE_DIR / "models" / "best_fls.pt"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


import torch
DEFAULT_DEVICE = "0" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 Debris Detection on FLS frames")
    parser.add_argument("--input", type=str, required=False, help="Path to FLS image, folder, or video")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained weights")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="CUDA device index or 'cpu'")
    return parser.parse_args()



def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        # Fallback to models/best.pt if best_fls.pt doesn't exist
        alt_model = BASE_DIR / "models" / "best.pt"
        if alt_model.exists():
            model_path = alt_model
        else:
            print(f"[!] Model not found at: {model_path}")
            print("    Please run 'python train_fls.py' first.")
            sys.exit(1)

    if not args.input:
        val_samples = list((BASE_DIR / "data" / "images" / "val").glob("*.png")) + \
                      list((BASE_DIR / "data" / "images" / "val").glob("*.jpg"))
        if not val_samples:
            print("[!] No --input provided and no validation sample found.")
            print("    Usage: python predict_fls.py --input path_to_fls_image.png")
            sys.exit(1)
        input_source = str(val_samples[0])
        print(f"[*] No input specified. Running on validation sample: {input_source}")
    else:
        input_source = args.input

    print(f"[*] Loading model {model_path}...")
    model = YOLO(str(model_path))

    print(f"[*] Running inference on {input_source} (conf={args.conf})...")
    results = model.predict(
        source=input_source,
        conf=args.conf,
        device=args.device,
        save=True,
        project=str(OUTPUT_DIR),
        name="fls_predictions",
        exist_ok=True
    )

    print(f"\n[+] Detections completed! Saved annotated images to:")
    print(f"    {OUTPUT_DIR / 'fls_predictions'}")


if __name__ == "__main__":
    main()

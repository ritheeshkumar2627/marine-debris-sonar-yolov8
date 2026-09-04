"""
03_predict_sahi.py
==================
Step 3 in Sonar Debris Detection Pipeline:
- Loads trained YOLOv8 model from 'models/best.pt'
- Uses SAHI (Slicing Aided Hyper Inference) to inspect massive waterfall images
  without downsampling small acoustic debris targets
- Stitches sliced detections and exports annotated results to 'outputs/'
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sliced Inference on Raw/Large Sonar Images")
    parser.add_argument("--input", type=str, required=False, help="Path to large sonar waterfall image (.tif, .png, etc.)")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to trained weights (default: models/best.pt)")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument("--tile-size", type=int, default=640, help="Slice size for sliding window (default: 640)")
    parser.add_argument("--overlap", type=float, default=0.20, help="Overlap ratio between slices (default: 0.20)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device: cuda:0 or cpu")
    return parser.parse_args()


def run_sahi_inference(image_path: Path, model_path: Path, conf: float, tile_size: int, overlap: float, device: str):
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    print(f"[*] Loading model from {model_path} on {device}...")
    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=str(model_path),
            confidence_threshold=conf,
            device=device
        )
    except Exception as e:
        print(f"[!] Warning CUDA failed or unavailable ({e}). Falling back to CPU.")
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=str(model_path),
            confidence_threshold=conf,
            device="cpu"
        )

    print(f"[*] Running Sliced Inference on: {image_path.name}")
    print(f"    Slice dimensions: {tile_size}x{tile_size} | Overlap: {overlap * 100:.0f}%")

    result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        perform_standard_pred=False
    )

    num_detections = len(result.object_prediction_list)
    print(f"[+] Total debris targets detected: {num_detections}")

    # Print summary of detections
    for idx, pred in enumerate(result.object_prediction_list):
        bbox = pred.bbox.to_xywh()
        print(f"  Target #{idx+1}: {pred.category.name} (Conf: {pred.score.value:.2f}) at [x={int(bbox[0])}, y={int(bbox[1])}, w={int(bbox[2])}, h={int(bbox[3])}]")

    # Export annotated image
    export_folder = OUTPUT_DIR / f"run_{image_path.stem}"
    export_folder.mkdir(parents=True, exist_ok=True)
    result.export_visuals(export_dir=str(export_folder), file_name="detected_mosaic")
    print(f"[+] Annotated image exported to: {export_folder / 'detected_mosaic.png'}")


def main():
    args = parse_args()
    model_file = Path(args.model)

    if not model_file.exists():
        print(f"[!] Model weights not found at: {model_file}")
        print("    Please run 'python 02_train.py' first, or specify an existing model with --model.")
        sys.exit(1)

    if not args.input:
        # Check raw_data or data/images/val for a sample image
        sample_candidates = list((BASE_DIR / "raw_data").glob("*.png")) + \
                            list((BASE_DIR / "data" / "images" / "val").glob("*.png"))
        if not sample_candidates:
            print("[!] No test image provided and no candidate images found.")
            print("    Usage: python 03_predict_sahi.py --input path_to_image.png")
            sys.exit(1)
        image_path = sample_candidates[0]
        print(f"[*] No --input specified. Using candidate sample: {image_path}")
    else:
        image_path = Path(args.input)
        if not image_path.exists():
            print(f"[!] Input image does not exist: {image_path}")
            sys.exit(1)

    run_sahi_inference(
        image_path=image_path,
        model_path=model_file,
        conf=args.conf,
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=args.device
    )


if __name__ == "__main__":
    main()

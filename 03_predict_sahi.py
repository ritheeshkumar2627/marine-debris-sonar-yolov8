"""
03_predict_sahi.py
==================
Full Pipeline for Inference on Raw Side-Scan Sonar (SSS) Surveys:
1. Ingests raw SSS files (.xtf, 16-bit GeoTIFF, PNG, JPG)
2. Normalizes dynamic range (1%-99% percentile stretch, bilateral despeckling, CLAHE)
3. Sliced Inference via SAHI (sliding window across massive waterfall strips)
4. Exports:
   - Annotated full-resolution mosaic (detected_mosaic.png)
   - Structured detections table (detections.csv)
"""

import csv
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "best_fls.pt"
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = BASE_DIR / "models" / "best.pt"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DEVICE = "0" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sliced Inference on Raw SSS Waterfall Images")
    parser.add_argument("--input", type=str, required=True, help="Path to raw SSS image (.xtf, .tif, .png, etc.)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained weights")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold (default: 0.30)")
    parser.add_argument("--tile-size", type=int, default=640, help="Sliding window tile size (default: 640)")
    parser.add_argument("--overlap", type=float, default=0.20, help="Overlap ratio between tiles (default: 0.20)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="CUDA device (e.g., '0') or 'cpu'")
    return parser.parse_args()


def load_and_preprocess_raw_sss(input_path: Path) -> Path:
    """Loads and enhances raw SSS image or .xtf to standard 8-bit image."""
    ext = input_path.suffix.lower()

    if ext == ".xtf":
        try:
            import pyxtf
            print(f"[*] Reading .xtf sonar packets from {input_path.name}...")
            (header, packets) = pyxtf.xtf_read(str(input_path))
            port = pyxtf.concatenate_channel(packets, file_header=header, channel=0)
            stbd = pyxtf.concatenate_channel(packets, file_header=header, channel=1)
            raw_sonar = np.hstack([np.fliplr(port), stbd]).astype(np.float32)
        except ImportError:
            raise ImportError("pyxtf is required for .xtf files. Run: pip install pyxtf")
    else:
        raw_sonar = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if raw_sonar is None:
            raise ValueError(f"Could not read image: {input_path}")
        if len(raw_sonar.shape) == 3:
            raw_sonar = cv2.cvtColor(raw_sonar, cv2.COLOR_BGR2GRAY)
        raw_sonar = raw_sonar.astype(np.float32)

    # 1. Percentile contrast stretching (eliminates acoustic sensor spikes)
    p_low, p_high = np.percentile(raw_sonar, (1, 99))
    if p_high == p_low:
        clipped = np.zeros_like(raw_sonar, dtype=np.uint8)
    else:
        clipped = np.clip(raw_sonar, p_low, p_high)
        clipped = ((clipped - p_low) / (p_high - p_low + 1e-6) * 255.0).astype(np.uint8)

    # 2. Despeckle filtering
    denoised = cv2.bilateralFilter(clipped, d=5, sigmaColor=50, sigmaSpace=50)

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Save cached preprocessed image for SAHI
    cache_dir = OUTPUT_DIR / "preprocessed_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{input_path.stem}_enhanced.png"
    cv2.imwrite(str(out_path), enhanced_bgr)
    return out_path


def run_sahi_inference(image_path: Path, model_path: Path, conf: float, tile_size: int, overlap: float, device: str):
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    # Preprocess raw SSS
    print(f"[*] Step 1: Ingesting & normalizing raw sonar image: {image_path.name}")
    processed_image_path = load_and_preprocess_raw_sss(image_path)

    # Load model
    print(f"[*] Step 2: Loading YOLOv8 model from {model_path} on {device}...")
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

    # Run Sliced Hyper Inference
    print(f"[*] Step 3: Running Sliced Hyper Inference (SAHI)...")
    print(f"    Tile size: {tile_size}x{tile_size} | Overlap: {overlap * 100:.0f}%")
    result = get_sliced_prediction(
        str(processed_image_path),
        detection_model,
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        perform_standard_pred=False
    )

    num_detections = len(result.object_prediction_list)
    print(f"\n" + "=" * 70)
    print(f"[+] Detection Complete! Found {num_detections} debris targets.")
    print("=" * 70)

    # Export folder
    export_folder = OUTPUT_DIR / f"run_{image_path.stem}"
    export_folder.mkdir(parents=True, exist_ok=True)

    # Write detections to CSV
    csv_path = export_folder / "detections.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Target_ID", "Class_Name", "Confidence", "X_pixel", "Y_pixel", "Width_px", "Height_px"])
        for idx, pred in enumerate(result.object_prediction_list):
            bbox = pred.bbox.to_xywh()
            writer.writerow([
                idx + 1,
                pred.category.name,
                f"{pred.score.value:.3f}",
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3])
            ])
            print(f"  Target #{idx+1:02d}: {pred.category.name:<16} (Conf: {pred.score.value:.2f}) at [x={int(bbox[0])}, y={int(bbox[1])}, w={int(bbox[2])}, h={int(bbox[3])}]")

    # Export annotated visual mosaic
    result.export_visuals(export_dir=str(export_folder), file_name="detected_mosaic")
    print(f"\n[+] Output Results:")
    print(f"    1. Annotated Visual: {export_folder / 'detected_mosaic.png'}")
    print(f"    2. Coordinates Table: {csv_path}")


def main():
    args = parse_args()
    input_path = Path(args.input)
    model_path = Path(args.model)

    if not input_path.exists():
        print(f"[!] Input file does not exist: {input_path}")
        sys.exit(1)

    if not model_path.exists():
        print(f"[!] Model weights not found at: {model_path}")
        print("    Please train the model first with 'python train_fls.py', or supply --model path.")
        sys.exit(1)

    run_sahi_inference(
        image_path=input_path,
        model_path=model_path,
        conf=args.conf,
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=args.device
    )


if __name__ == "__main__":
    main()

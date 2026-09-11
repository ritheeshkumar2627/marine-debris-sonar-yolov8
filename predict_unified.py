"""
predict_unified.py
==================
Smart Unified Predictor for Underwater Sonar:
Automatically analyzes the input image and dispatches to the correct model & inference pipeline:

1. If Input is Forward-Looking Sonar (FLS) frame (e.g. 605x500 standard aspect ratio):
   -> Automatically selects 'models/best_fls.pt' (10 Debris Classes: bottles, tires, cans, etc.)
   -> Runs fast direct frame prediction.

2. If Input is Side-Scan Sonar (SSS) survey (e.g. waterfall strip, high aspect ratio, or .xtf):
   -> Automatically selects 'models/best_ghostvision.pt' (Ghost Crab Pots & SSS Debris)
   -> Runs Sliced Hyper-Inference (SAHI) with 640x640 sliding window & coordinate stitching.
   -> Computes GPS Latitude & Longitude for each detected target.
   -> Exports detections.csv and interactive detections.geojson for GIS/mapping.
"""

import csv
import argparse
import sys
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

from georeference import (
    calculate_target_gps,
    extract_xtf_ping_info,
    read_geotiff_coords,
    pixel_to_gps,
    export_geojson
)

BASE_DIR = Path(__file__).resolve().parent
FLS_MODEL_PATH = BASE_DIR / "models" / "best_fls.pt"
SSS_MODEL_PATH = BASE_DIR / "models" / "best_ghostvision.pt"
OUTPUT_DIR = BASE_DIR / "outputs" / "unified_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DEVICE = "0" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Sonar Debris & Ghost Gear Predictor with GPS Georeferencing")
    parser.add_argument("--input", type=str, required=True, help="Path to sonar image (.png, .jpg, .tif, .xtf)")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "fls", "sss", "both"],
                        help="Sonar mode: 'auto' (detects geometry automatically), 'fls', 'sss', or 'both'")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold (default: 0.30)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="CUDA device or 'cpu'")
    
    # Georeferencing options
    parser.add_argument("--vessel-lat", type=float, default=None, help="Vessel Latitude in decimal degrees (e.g. 38.7512)")
    parser.add_argument("--vessel-lon", type=float, default=None, help="Vessel Longitude in decimal degrees (e.g. -75.0824)")
    parser.add_argument("--vessel-heading", type=float, default=0.0, help="Vessel course heading in degrees (0=North, 90=East)")
    parser.add_argument("--max-range", type=float, default=50.0, help="Maximum sonar slant range per channel in meters (default: 50m)")
    return parser.parse_args()


def inspect_sonar_type(file_path: Path) -> str:
    """
    Analyzes file extension, aspect ratio, and pixel dimensions
    to determine whether it is Side-Scan Sonar (SSS) or Forward-Looking Sonar (FLS).
    """
    ext = file_path.suffix.lower()
    if ext == ".xtf":
        return "sss"

    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return "fls"

    h, w = img.shape[:2]
    aspect_ratio = max(h, w) / max(min(h, w), 1)

    # Side-Scan Sonar waterfall surveys have high aspect ratios (> 1.8) or large dimensions (> 1500px)
    if aspect_ratio >= 1.8 or max(h, w) >= 1500:
        return "sss"
    return "fls"


def run_fls_prediction(input_path: Path, conf: float, device: str):
    if not FLS_MODEL_PATH.exists():
        print(f"[!] FLS model weights not found at: {FLS_MODEL_PATH}")
        print("    Please train FLS model first ('python train_fls.py') or place weights in models/.")
        return

    print(f"[*] Dispatching to: FLS Marine Debris Model ({FLS_MODEL_PATH.name})")
    model = YOLO(str(FLS_MODEL_PATH))
    results = model.predict(
        source=str(input_path),
        conf=conf,
        device=device,
        save=True,
        project=str(OUTPUT_DIR),
        name=f"fls_{input_path.stem}",
        exist_ok=True
    )

    out_folder = OUTPUT_DIR / f"fls_{input_path.stem}"
    csv_path = out_folder / "detections.csv"

    # Save structured CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Target_ID", "Class_Name", "Confidence", "X_pixel", "Y_pixel", "Width_px", "Height_px"])
        target_id = 1
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, str(cls_id))
                score = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                x = int(xyxy[0])
                y = int(xyxy[1])
                w = int(xyxy[2] - xyxy[0])
                h = int(xyxy[3] - xyxy[1])
                writer.writerow([target_id, cls_name, f"{score:.3f}", x, y, w, h])
                print(f"  Target #{target_id:02d}: {cls_name:<16} (Conf: {score:.2f}) at [x={x}, y={y}, w={w}, h={h}]")
                target_id += 1

    print(f"[+] FLS Prediction Complete! Results saved to: {out_folder}")
    print(f"    1. Annotated Visual: {out_folder / input_path.name}")
    print(f"    2. Coordinates Table: {csv_path}")


def run_sss_prediction(
    input_path: Path,
    conf: float,
    device: str,
    vessel_lat: float = None,
    vessel_lon: float = None,
    vessel_heading: float = 0.0,
    max_range: float = 50.0
):
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    # Use GhostVision model if available, fallback to general best.pt
    model_path = SSS_MODEL_PATH if SSS_MODEL_PATH.exists() else (BASE_DIR / "models" / "best.pt")
    if not model_path.exists():
        model_path = FLS_MODEL_PATH
    if not model_path.exists():
        print(f"[!] SSS model weights not found at: {SSS_MODEL_PATH}")
        print("    Please train GhostVision model ('python train_ghostvision.py') or place weights in models/.")
        return

    print(f"[*] Dispatching to: SSS GhostVision Model ({model_path.name}) with SAHI Slicing")

    # Import acoustic preprocessor from 03_predict_sahi
    try:
        from importlib.machinery import SourceFileLoader
        sahi_module = SourceFileLoader("predict_sahi", str(BASE_DIR / "03_predict_sahi.py")).load_module()
        processed_path, img_w, img_h, nav_pings = sahi_module.load_and_preprocess_raw_sss(input_path)
    except Exception as e:
        print(f"[!] Falling back to direct image: {e}")
        processed_path = input_path
        img = cv2.imread(str(input_path))
        img_h, img_w = img.shape[:2] if img is not None else (1000, 1000)
        nav_pings = []

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(model_path),
        confidence_threshold=conf,
        device=device
    )

    result = get_sliced_prediction(
        str(processed_path),
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.20,
        overlap_width_ratio=0.20,
        perform_standard_pred=False
    )

    export_dir = OUTPUT_DIR / f"sss_{input_path.stem}"
    export_dir.mkdir(parents=True, exist_ok=True)

    csv_path = export_dir / "detections.csv"
    geojson_path = export_dir / "detections.geojson"
    geo_detections = []

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Target_ID", "Class_Name", "Confidence",
            "X_pixel", "Y_pixel", "Width_px", "Height_px",
            "Latitude", "Longitude"
        ])

        for idx, pred in enumerate(result.object_prediction_list):
            bbox = pred.bbox.to_xywh()
            x_center = bbox[0] + bbox[2] / 2.0
            y_center = bbox[1] + bbox[3] / 2.0

            # Calculate GPS
            lat, lon = None, None
            if input_path.suffix.lower() in [".tif", ".tiff"]:
                lat, lon = read_geotiff_coords(input_path, x_center, y_center)

            if lat is None or lon is None:
                lat, lon = pixel_to_gps(
                    x_px=x_center,
                    y_px=y_center,
                    image_width=img_w,
                    image_height=img_h,
                    vessel_lat=vessel_lat,
                    vessel_lon=vessel_lon,
                    vessel_heading=vessel_heading,
                    max_range_m=max_range,
                    nav_pings=nav_pings
                )

            lat_str = f"{lat:.7f}" if lat is not None else "N/A"
            lon_str = f"{lon:.7f}" if lon is not None else "N/A"

            writer.writerow([
                idx + 1,
                pred.category.name,
                f"{pred.score.value:.3f}",
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
                lat_str,
                lon_str
            ])

            geo_detections.append({
                "target_id": idx + 1,
                "class_name": pred.category.name,
                "confidence": float(pred.score.value),
                "x_pixel": int(bbox[0]),
                "y_pixel": int(bbox[1]),
                "width_px": int(bbox[2]),
                "height_px": int(bbox[3]),
                "latitude": lat,
                "longitude": lon
            })

            gps_display = f"GPS: ({lat_str}, {lon_str})" if lat is not None else "GPS: N/A"
            print(f"  Target #{idx+1:02d}: {pred.category.name:<16} (Conf: {pred.score.value:.2f}) at [x={int(bbox[0])}, y={int(bbox[1])}] | {gps_display}")

    has_gps = any(d["latitude"] is not None for d in geo_detections)
    if has_gps:
        export_geojson(geo_detections, geojson_path)

    result.export_visuals(export_dir=str(export_dir), file_name="detected_mosaic")
    print(f"\n[+] SSS Sliced Prediction Complete! Results saved to: {export_dir}")
    print(f"    1. Annotated Visual:    {export_dir / 'detected_mosaic.png'}")
    print(f"    2. Coordinates Table:   {csv_path}")
    if has_gps:
        print(f"    3. Interactive GeoJSON: {geojson_path}")


def main():
    args = parse_args()
    input_file = Path(args.input)

    if not input_file.exists():
        print(f"[!] Input file not found: {input_file}")
        sys.exit(1)

    print("=" * 70)
    print(f"Unified Sonar Predictor | Inspecting: {input_file.name}")
    print("=" * 70)

    # Determine mode
    if args.mode == "auto":
        detected_type = inspect_sonar_type(input_file)
        print(f"[*] Auto-detected Sonar Geometry: {detected_type.upper()}")
        mode = detected_type
    else:
        mode = args.mode

    if mode == "fls":
        run_fls_prediction(input_file, conf=args.conf, device=args.device)
    elif mode == "sss":
        run_sss_prediction(
            input_file,
            conf=args.conf,
            device=args.device,
            vessel_lat=args.vessel_lat,
            vessel_lon=args.vessel_lon,
            vessel_heading=args.vessel_heading,
            max_range=args.max_range
        )
    elif mode == "both":
        print("[*] Running Dual-Model Ensemble (FLS + SSS GhostVision)...")
        run_fls_prediction(input_file, conf=args.conf, device=args.device)
        run_sss_prediction(
            input_file,
            conf=args.conf,
            device=args.device,
            vessel_lat=args.vessel_lat,
            vessel_lon=args.vessel_lon,
            vessel_heading=args.vessel_heading,
            max_range=args.max_range
        )


if __name__ == "__main__":
    main()

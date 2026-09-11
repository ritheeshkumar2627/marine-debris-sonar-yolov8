"""
03_predict_sahi.py
==================
Full Pipeline for Inference on Raw Side-Scan Sonar (SSS) Surveys:
1. Ingests raw SSS files (.xtf, 16-bit GeoTIFF, PNG, JPG)
2. Normalizes dynamic range (1%-99% percentile stretch, bilateral despeckling, CLAHE)
3. Sliced Inference via SAHI (sliding window across massive waterfall strips)
4. Georeferencing: Calculates real-world Latitude & Longitude from:
   - Embedded .XTF acoustic navigation headers (pyxtf)
   - GeoTIFF geospatial transforms (rasterio)
   - User-supplied GPS vessel position, heading, and slant range
5. Exports:
   - Annotated full-resolution mosaic (detected_mosaic.png)
   - Structured detections table with GPS (detections.csv)
   - Interactive GIS map for QGIS / Google Earth (detections.geojson)
"""

import csv
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

from georeference import (
    calculate_target_gps,
    extract_xtf_ping_info,
    read_geotiff_coords,
    pixel_to_gps,
    export_geojson
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "best_ghostvision.pt"
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = BASE_DIR / "models" / "best_fls.pt"
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = BASE_DIR / "models" / "best.pt"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DEVICE = "0" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sliced Inference with GPS Georeferencing on SSS Sonar")
    parser.add_argument("--input", type=str, required=True, help="Path to raw SSS image (.xtf, .tif, .png, etc.)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained weights")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold (default: 0.30)")
    parser.add_argument("--tile-size", type=int, default=640, help="Sliding window tile size (default: 640)")
    parser.add_argument("--overlap", type=float, default=0.20, help="Overlap ratio between tiles (default: 0.20)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="CUDA device (e.g., '0') or 'cpu'")
    
    # Georeferencing arguments
    parser.add_argument("--vessel-lat", type=float, default=None, help="Vessel Latitude in decimal degrees (e.g. 38.7512)")
    parser.add_argument("--vessel-lon", type=float, default=None, help="Vessel Longitude in decimal degrees (e.g. -75.0824)")
    parser.add_argument("--vessel-heading", type=float, default=0.0, help="Vessel course heading in degrees (0=North, 90=East)")
    parser.add_argument("--max-range", type=float, default=50.0, help="Maximum sonar slant range per channel in meters (default: 50m)")
    return parser.parse_args()


def load_and_preprocess_raw_sss(input_path: Path):
    """Loads and enhances raw SSS image or .xtf to standard 8-bit image."""
    ext = input_path.suffix.lower()
    nav_pings = []

    if ext == ".xtf":
        try:
            import pyxtf
            print(f"[*] Reading .xtf sonar packets from {input_path.name}...")
            (header, packets) = pyxtf.xtf_read(str(input_path))
            port = pyxtf.concatenate_channel(packets, file_header=header, channel=0)
            stbd = pyxtf.concatenate_channel(packets, file_header=header, channel=1)
            raw_sonar = np.hstack([np.fliplr(port), stbd]).astype(np.float32)
            nav_pings = extract_xtf_ping_info(input_path)
            print(f"[*] Extracted navigation metadata for {len(nav_pings)} acoustic pings from .xtf.")
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
    return out_path, enhanced_bgr.shape[1], enhanced_bgr.shape[0], nav_pings


def run_sahi_inference(
    image_path: Path,
    model_path: Path,
    conf: float,
    tile_size: int,
    overlap: float,
    device: str,
    vessel_lat: float = None,
    vessel_lon: float = None,
    vessel_heading: float = 0.0,
    max_range: float = 50.0
):
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    # Preprocess raw SSS
    print(f"[*] Step 1: Ingesting & normalizing raw sonar image: {image_path.name}")
    processed_image_path, img_w, img_h, nav_pings = load_and_preprocess_raw_sss(image_path)

    # Load model
    print(f"[*] Step 2: Loading YOLO model from {model_path} on {device}...")
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

    # Write detections to CSV and collect GeoJSON features
    csv_path = export_folder / "detections.csv"
    geojson_path = export_folder / "detections.geojson"
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
            # Method A: GeoTIFF transform
            if image_path.suffix.lower() in [".tif", ".tiff"]:
                lat, lon = read_geotiff_coords(image_path, x_center, y_center)

            # Method B: XTF nav pings or user-supplied vessel coordinates
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

            gps_display = f"GPS: ({lat_str}, {lon_str})" if lat is not None else "GPS: N/A (supply --vessel-lat/lon or .xtf)"
            print(f"  Target #{idx+1:02d}: {pred.category.name:<16} (Conf: {pred.score.value:.2f}) at [x={int(bbox[0])}, y={int(bbox[1])}] | {gps_display}")

    # Export GeoJSON if any targets have coordinates
    has_gps = any(d["latitude"] is not None for d in geo_detections)
    if has_gps:
        export_geojson(geo_detections, geojson_path)

    # Export annotated visual mosaic
    result.export_visuals(export_dir=str(export_folder), file_name="detected_mosaic")
    print(f"\n[+] Output Results:")
    print(f"    1. Annotated Visual:   {export_folder / 'detected_mosaic.png'}")
    print(f"    2. Coordinates Table:  {csv_path}")
    if has_gps:
        print(f"    3. Interactive GeoJSON: {geojson_path}")


def main():
    args = parse_args()
    input_path = Path(args.input)
    model_path = Path(args.model)

    if not input_path.exists():
        print(f"[!] Input file does not exist: {input_path}")
        sys.exit(1)

    if not model_path.exists():
        print(f"[!] Model weights not found at: {model_path}")
        print("    Please train the model first or supply --model path.")
        sys.exit(1)

    run_sahi_inference(
        image_path=input_path,
        model_path=model_path,
        conf=args.conf,
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=args.device,
        vessel_lat=args.vessel_lat,
        vessel_lon=args.vessel_lon,
        vessel_heading=args.vessel_heading,
        max_range=args.max_range
    )


if __name__ == "__main__":
    main()

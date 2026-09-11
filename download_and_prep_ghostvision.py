"""
download_and_prep_ghostvision.py
================================
Downloads and prepares the official GhostVision Side-Scan Sonar (SSS) dataset:
6,674 Side-Scan Sonar images of derelict crab pots & marine debris.

Sources:
1. Zenodo Archive (Public direct download, no login/token required):
   https://zenodo.org/records/20056679
2. Hugging Face Hub (gated, requires HF account & login):
   PINGEcosystem/sss-crab-pot-detection-ds

Classes:
    0: Crab-Pot
    1: Maybe-Crab-Pot

Automatically converts annotations to standard normalized YOLO format (.txt)
and organizes data into:
    data_ghostvision/
      images/train/, images/val/
      labels/train/, labels/val/
      data.yaml
"""

import os
import sys
import json
import shutil
import random
import zipfile
import urllib.request
from pathlib import Path
import yaml
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
GV_DATA_DIR = BASE_DIR / "data_ghostvision"
MODELS_DIR = BASE_DIR / "models"

IMG_TRAIN_DIR = GV_DATA_DIR / "images" / "train"
IMG_VAL_DIR = GV_DATA_DIR / "images" / "val"
LBL_TRAIN_DIR = GV_DATA_DIR / "labels" / "train"
LBL_VAL_DIR = GV_DATA_DIR / "labels" / "val"

ZENODO_URL = "https://zenodo.org/api/records/20056679/files/GhostVision_DatasetAndModels.zip/content"
ZIP_PATH = GV_DATA_DIR / "GhostVision_DatasetAndModels.zip"

VAL_SPLIT = 0.20

CLASS_MAP = {
    "crab-pot": 0,
    "maybe-crab-pot": 1
}


def download_with_progress(url: str, output_path: Path):
    """Downloads a file with clean progress reporting."""
    print(f"[*] Downloading dataset from: {url}")
    print(f"[*] Destination: {output_path}")

    class ProgressHook:
        def __init__(self):
            self.last_percent = -1

        def __call__(self, block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = int(downloaded * 100 / total_size)
                if percent % 10 == 0 and percent != self.last_percent:
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    print(f"    [{percent:3d}%] {mb_down:.1f} MB / {mb_tot:.1f} MB downloaded...")
                    self.last_percent = percent

    urllib.request.urlretrieve(url, str(output_path), reporthook=ProgressHook())
    print("[+] Download complete!")


def convert_metadata_jsonl_to_yolo(split_folder: Path, target_split: str, box_counter: list) -> int:
    """Parses a metadata.jsonl file and writes YOLO label .txt files and copies images."""
    meta_path = split_folder / "metadata.jsonl"
    if not meta_path.exists():
        return 0

    target_img_dir = IMG_VAL_DIR if target_split == "val" else IMG_TRAIN_DIR
    target_lbl_dir = LBL_VAL_DIR if target_split == "val" else LBL_TRAIN_DIR

    processed = 0
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            file_name = item.get("file_name")
            if not file_name:
                continue

            src_img = split_folder / file_name
            if not src_img.exists():
                continue

            try:
                with Image.open(src_img) as im:
                    img_w, img_h = im.size
            except Exception:
                continue

            stem = src_img.stem
            objects = item.get("objects", {})
            bboxes = objects.get("bbox", [])
            categories = objects.get("category", [])

            yolo_lines = []
            for bbox, cat in zip(bboxes, categories):
                cat_name = str(cat).strip().lower()
                cls_id = CLASS_MAP.get(cat_name, 0)

                x, y, w, h = bbox
                xc = (x + w / 2.0) / float(img_w)
                yc = (y + h / 2.0) / float(img_h)
                nw = w / float(img_w)
                nh = h / float(img_h)

                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                box_counter[0] += 1

            # Destination files
            dst_img = target_img_dir / src_img.name
            dst_lbl = target_lbl_dir / f"{stem}.txt"

            shutil.copy2(src_img, dst_img)
            with open(dst_lbl, "w") as lf:
                if yolo_lines:
                    lf.write("\n".join(yolo_lines) + "\n")

            processed += 1

    return processed


def prep_via_zenodo():
    """Fallback method downloading official Zenodo archive with zero authentication required."""
    print("\n[*] Initializing direct Zenodo download pipeline...")
    GV_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        download_with_progress(ZENODO_URL, ZIP_PATH)
    else:
        print(f"[*] Found existing archive at: {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024*1024):.1f} MB)")

    print("[*] Extracting GhostVision dataset and models...")
    extract_dir = GV_DATA_DIR / "zenodo_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(extract_dir)

    print("[+] Archive extracted successfully. Converting annotations to YOLO format...")

    for d in [IMG_TRAIN_DIR, IMG_VAL_DIR, LBL_TRAIN_DIR, LBL_VAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    box_counter = [0]
    total_imgs = 0

    ds_root = extract_dir / "sss-crab-pot-detection-ds"
    if not ds_root.exists():
        # Search recursively
        matches = list(extract_dir.rglob("metadata.jsonl"))
        folders_to_process = [m.parent for m in matches]
    else:
        folders_to_process = []
        for split in ["train", "valid", "val", "test"]:
            p = ds_root / split
            if p.exists():
                folders_to_process.append(p)

    for folder in folders_to_process:
        split_name = folder.name.lower()
        target_split = "val" if split_name in ["valid", "val", "test"] else "train"
        count = convert_metadata_jsonl_to_yolo(folder, target_split, box_counter)
        print(f"    Processed {count} images from '{folder.name}' -> mapped to {target_split}")
        total_imgs += count

    # Also copy pre-trained YOLO weights from Zenodo if available
    pretrain_matches = list(extract_dir.rglob("*.safetensors")) + list(extract_dir.rglob("*.onnx"))
    for model_file in pretrain_matches:
        dst = MODELS_DIR / model_file.name
        shutil.copy2(model_file, dst)
        print(f"[+] Found pre-trained GhostVision model: {model_file.name} -> copied to models/")

    return total_imgs, box_counter[0]


def prep_via_huggingface():
    """Downloads from Hugging Face Hub (works if user has HF login / token)."""
    from datasets import load_dataset

    for d in [IMG_TRAIN_DIR, IMG_VAL_DIR, LBL_TRAIN_DIR, LBL_VAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("[*] Attempting to load from Hugging Face Hub...")
    ds = load_dataset("PINGEcosystem/sss-crab-pot-detection-ds")

    split_name = "train" if "train" in ds else list(ds.keys())[0]
    data_split = ds[split_name]
    total_records = len(data_split)
    print(f"[*] Loaded {total_records} records from Hugging Face '{split_name}'.")

    random.seed(42)
    box_count = 0
    processed_count = 0

    for idx, item in enumerate(data_split):
        img_obj = item.get("image")
        file_name = item.get("file_name", f"sss_ghostvision_{idx:05d}.jpg")
        stem = Path(file_name).stem

        if img_obj is None:
            continue

        img_w, img_h = img_obj.size
        objects = item.get("objects", {})
        bboxes = objects.get("bbox", [])
        categories = objects.get("category", [])

        yolo_lines = []
        for bbox, cat in zip(bboxes, categories):
            cat_name = str(cat).strip().lower()
            cls_id = CLASS_MAP.get(cat_name, 0)

            x, y, w, h = bbox
            xc = (x + w / 2.0) / float(img_w)
            yc = (y + h / 2.0) / float(img_h)
            nw = w / float(img_w)
            nh = h / float(img_h)

            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            box_count += 1

        is_val = (random.random() < VAL_SPLIT)
        target_img_dir = IMG_VAL_DIR if is_val else IMG_TRAIN_DIR
        target_lbl_dir = LBL_VAL_DIR if is_val else LBL_TRAIN_DIR

        out_img_path = target_img_dir / f"{stem}.jpg"
        img_obj.save(out_img_path)

        out_lbl_path = target_lbl_dir / f"{stem}.txt"
        with open(out_lbl_path, "w") as lf:
            if yolo_lines:
                lf.write("\n".join(yolo_lines) + "\n")

        processed_count += 1

    return processed_count, box_count


def main():
    print("=" * 70)
    print("GhostVision Side-Scan Sonar (SSS) Dataset Preparation")
    print("=" * 70)

    total_images = 0
    total_boxes = 0

    # Try Hugging Face first; if gated/authentication fails, fallback to Zenodo
    try:
        total_images, total_boxes = prep_via_huggingface()
    except Exception as e:
        print(f"\n[!] Hugging Face download note: {e}")
        print("[*] Gated repository detected. Switching automatically to public Zenodo archive...")
        total_images, total_boxes = prep_via_zenodo()

    # Write data.yaml
    yaml_config = {
        "path": str(GV_DATA_DIR).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "Crab-Pot",
            1: "Maybe-Crab-Pot"
        }
    }
    yaml_path = GV_DATA_DIR / "data.yaml"
    with open(yaml_path, "w") as yf:
        yaml.dump(yaml_config, yf, sort_keys=False)

    print("\n" + "=" * 70)
    print("[+] GhostVision SSS dataset successfully converted to YOLO format!")
    print(f"    Total images:         {total_images}")
    print(f"    Total bounding boxes: {total_boxes}")
    print(f"    YAML configuration:   {yaml_path}")
    print(f"    Train images:         {len(list(IMG_TRAIN_DIR.glob('*')))} in {IMG_TRAIN_DIR}")
    print(f"    Val images:           {len(list(IMG_VAL_DIR.glob('*')))} in {IMG_VAL_DIR}")
    print("=" * 70)
    print("\n[NEXT STEP]: Train YOLOv8 on GhostVision SSS using:")
    print("    python train_ghostvision.py")


if __name__ == "__main__":
    main()

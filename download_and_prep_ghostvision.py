"""
download_and_prep_ghostvision.py
================================
Downloads and prepares the official GhostVision Side-Scan Sonar (SSS) dataset:
"PINGEcosystem/sss-crab-pot-detection-ds" from Hugging Face.

- 6,674 Side-Scan Sonar images of derelict crab pots & marine debris
- Classes:
    0: Crab-Pot
    1: Maybe-Crab-Pot
- Automatically converts annotations to normalized YOLO format (.txt)
- Organizes into data_ghostvision/images/ and data_ghostvision/labels/
"""

import os
import random
from pathlib import Path
import yaml
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
GV_DATA_DIR = BASE_DIR / "data_ghostvision"

IMG_TRAIN_DIR = GV_DATA_DIR / "images" / "train"
IMG_VAL_DIR = GV_DATA_DIR / "images" / "val"
LBL_TRAIN_DIR = GV_DATA_DIR / "labels" / "train"
LBL_VAL_DIR = GV_DATA_DIR / "labels" / "val"

VAL_SPLIT = 0.20

CLASS_MAP = {
    "crab-pot": 0,
    "maybe-crab-pot": 1
}


def main():
    print("=" * 70)
    print("GhostVision Side-Scan Sonar (SSS) Dataset Preparation")
    print("Dataset: PINGEcosystem/sss-crab-pot-detection-ds (Hugging Face)")
    print("=" * 70)

    try:
        from datasets import load_dataset
    except ImportError:
        print("[!] The 'datasets' library is required to download from Hugging Face.")
        print("    Please install it: pip install datasets huggingface_hub")
        return

    for d in [IMG_TRAIN_DIR, IMG_VAL_DIR, LBL_TRAIN_DIR, LBL_VAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("[*] Fetching dataset from Hugging Face...")
    ds = load_dataset("PINGEcosystem/sss-crab-pot-detection-ds")

    # Combine splits or use default train split
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    data_split = ds[split_name]
    total_records = len(data_split)
    print(f"[*] Loaded {total_records} records from split: '{split_name}'. Processing...")

    random.seed(42)
    processed_count = 0
    box_count = 0

    for idx, item in enumerate(data_split):
        # Image can be PIL Image object or file_name
        img_obj = item.get("image")
        file_name = item.get("file_name", f"sss_ghostvision_{idx:05d}.jpg")
        stem = Path(file_name).stem

        if img_obj is None:
            continue

        img_w, img_h = img_obj.size

        # Parse bounding boxes
        objects = item.get("objects", {})
        bboxes = objects.get("bbox", [])
        categories = objects.get("category", [])

        yolo_lines = []
        for bbox, cat in zip(bboxes, categories):
            cat_name = str(cat).strip().lower()
            cls_id = CLASS_MAP.get(cat_name, 0)

            # bbox format in HuggingFace SSS dataset is [x, y, w, h] in pixels
            x, y, w, h = bbox
            xc = (x + w / 2.0) / float(img_w)
            yc = (y + h / 2.0) / float(img_h)
            nw = w / float(img_w)
            nh = h / float(img_h)

            # Clamp
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            box_count += 1

        # Train / Val Split
        is_val = (random.random() < VAL_SPLIT)
        target_img_dir = IMG_VAL_DIR if is_val else IMG_TRAIN_DIR
        target_lbl_dir = LBL_VAL_DIR if is_val else LBL_TRAIN_DIR

        # Save Image
        out_img_path = target_img_dir / f"{stem}.jpg"
        img_obj.save(out_img_path)

        # Save YOLO Label
        out_lbl_path = target_lbl_dir / f"{stem}.txt"
        with open(out_lbl_path, "w") as lf:
            if yolo_lines:
                lf.write("\n".join(yolo_lines) + "\n")

        processed_count += 1
        if (idx + 1) % 500 == 0 or (idx + 1) == total_records:
            print(f"[*] Processed {idx + 1}/{total_records} sonar images...")

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
    print(f"    Total images:         {processed_count}")
    print(f"    Total bounding boxes: {box_count}")
    print(f"    YAML configuration:   {yaml_path}")
    print(f"    Train images:         {len(list(IMG_TRAIN_DIR.glob('*')))} in {IMG_TRAIN_DIR}")
    print(f"    Val images:           {len(list(IMG_VAL_DIR.glob('*')))} in {IMG_VAL_DIR}")
    print("=" * 70)
    print("\n[NEXT STEP]: Train YOLOv8 on GhostVision SSS using:")
    print("    python train_ghostvision.py")


if __name__ == "__main__":
    main()

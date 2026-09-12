"""
fls_prepare_dataset.py
======================
Dataset Preparation for the Marine Debris Forward-Looking Sonar (FLS) Dataset
(Watertank Release 1.0, ARIS Explorer 3000, Heriot-Watt University)

1. Reads annotations.json and FLS images from raw_data/
2. Converts top-left bounding boxes to YOLO normalized format (class_id xc yc w h)
3. Performs an 80/20 train/val split
4. Populates data/images/train, data/images/val, data/labels/train, data/labels/val
"""

import json
import random
import shutil
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw_data"
JSON_PATH = RAW_DIR / "marine-debris-watertank-release" / "fls-images" / "annotations.json"
FLS_IMG_DIR = RAW_DIR / "marine-debris-watertank-release" / "fls-images"

DATA_DIR = BASE_DIR / "data"
IMG_TRAIN_DIR = DATA_DIR / "images" / "train"
IMG_VAL_DIR = DATA_DIR / "images" / "val"
LBL_TRAIN_DIR = DATA_DIR / "labels" / "train"
LBL_VAL_DIR = DATA_DIR / "labels" / "val"

VAL_RATIO = 0.20


def main():
    print("=" * 70)
    print("Preparing Marine Debris FLS Dataset for YOLOv8")
    print("=" * 70)

    for d in [IMG_TRAIN_DIR, IMG_VAL_DIR, LBL_TRAIN_DIR, LBL_VAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if not JSON_PATH.exists():
        print(f"[!] Annotations file not found at: {JSON_PATH}")
        return

    print(f"[*] Reading annotations from: {JSON_PATH.name} ...")
    with open(JSON_PATH, "r") as f:
        annotations = json.load(f)

    image_filenames = list(annotations.keys())
    print(f"[*] Found annotations for {len(image_filenames)} images.")

    random.seed(42)
    random.shuffle(image_filenames)

    train_count = 0
    val_count = 0
    total_boxes = 0

    for idx, fname in enumerate(image_filenames):
        img_file = FLS_IMG_DIR / fname
        if not img_file.exists():
            continue

        item = annotations[fname]
        boxes_info = item.get("bounding-boxes", [])

        # We know ARIS 3000 images in this dataset are 500x605, but let's read dynamically or set
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        yolo_lines = []
        for b in boxes_info:
            cls_id = b["class-id"]
            tl_x = b["top-left-x"]
            tl_y = b["top-left-y"]
            bw = b["width"]
            bh = b["height"]

            # Calculate normalized YOLO coordinates
            xc = (tl_x + bw / 2.0) / float(img_w)
            yc = (tl_y + bh / 2.0) / float(img_h)
            nw = bw / float(img_w)
            nh = bh / float(img_h)

            # Clamp between 0 and 1
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            total_boxes += 1

        is_val = (random.random() < VAL_RATIO)
        target_img_dir = IMG_VAL_DIR if is_val else IMG_TRAIN_DIR
        target_lbl_dir = LBL_VAL_DIR if is_val else LBL_TRAIN_DIR

        # Copy image
        shutil.copy2(img_file, target_img_dir / fname)

        # Save label text
        stem = Path(fname).stem
        label_path = target_lbl_dir / f"{stem}.txt"
        with open(label_path, "w") as lf:
            if yolo_lines:
                lf.write("\n".join(yolo_lines) + "\n")

        if is_val:
            val_count += 1
        else:
            train_count += 1

        if (idx + 1) % 200 == 0 or (idx + 1) == len(image_filenames):
            print(f"[*] Processed {idx + 1}/{len(image_filenames)} images...")

    print("\n" + "=" * 70)
    print(f"[+] Dataset successfully converted and populated!")
    print(f"    Total images:      {train_count + val_count}")
    print(f"    Total bounding boxes: {total_boxes}")
    print(f"    Train images:      {train_count} in {IMG_TRAIN_DIR}")
    print(f"    Train labels:      {len(list(LBL_TRAIN_DIR.glob('*.txt')))} in {LBL_TRAIN_DIR}")
    print(f"    Val images:        {val_count} in {IMG_VAL_DIR}")
    print(f"    Val labels:        {len(list(LBL_VAL_DIR.glob('*.txt')))} in {LBL_VAL_DIR}")
    print("=" * 70)
    print("\n[READY TO TRAIN]: Run:")
    print("    python train_fls.py")


if __name__ == "__main__":
    main()

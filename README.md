# Marine Debris Sonar Detection with YOLOv8
**Marine Debris Forward-Looking Sonar (FLS) Dataset (ARIS Explorer 3000)**

This repository provides an automated, end-to-end pipeline for training and running **YOLOv8** on acoustic camera sonar imagery to detect 10 classes of underwater debris.

---

## 1. Quickstart for Partners & Collaborators

When partners clone this repository, they can reproduce everything in **3 simple commands**:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and prepare the dataset (auto-populates data/ in YOLO format)
python download_fls_dataset.py
python fls_prepare_dataset.py

# 3. Train the model
python train_fls.py
```

---

## 2. Repository Structure

```text
sonar_debris_yolov8/
│
├── data/
│   └── data.yaml                   # YOLO configuration & class names
│
├── download_fls_dataset.py         # Step 1: Downloads the official Watertank dataset (345MB)
├── fls_prepare_dataset.py          # Step 2: Parses annotations.json & creates train/val splits
├── train_fls.py                    # Step 3: Trains YOLOv8 with acoustic-safe augmentations
├── predict_fls.py                  # Step 4: Runs detection on images, folders, or video
│
├── .gitignore                      # Excludes large raw data (>100MB) from git history
├── requirements.txt                # Python package dependencies
└── README.md
```

---

## 3. Detected Classes (10 Categories)

The model detects the following debris classes:
- `0: can`
- `1: bottle`
- `2: drink-carton`
- `3: chain`
- `4: propeller`
- `5: tire`
- `6: hook`
- `7: valve`
- `8: shampoo-bottle`
- `9: standing-bottle`

---

## 4. Scripts Overview

- **`download_fls_dataset.py`**: Fetches the official Marine Debris FLS Dataset (Watertank Release 1.0) and extracts it to `raw_data/`.
- **`fls_prepare_dataset.py`**: Parses bounding boxes from `annotations.json` and writes normalized YOLO labels into `data/labels/train` and `data/labels/val`.
- **`train_fls.py`**: Trains YOLOv8 on the dataset. Automatically selects GPU (CUDA) if available, otherwise runs on CPU.
- **`predict_fls.py`**: Runs inference and exports detection results to `outputs/fls_predictions/`.

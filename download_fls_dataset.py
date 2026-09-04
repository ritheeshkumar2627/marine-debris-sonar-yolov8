"""
download_fls_dataset.py
=======================
Downloads the official Marine Debris Forward-Looking Sonar (FLS) Dataset
(Watertank Release 1.0, ARIS Explorer 3000, Heriot-Watt University / Matias Valdenegro-Toro)
Directly into the raw_data/ directory and extracts it.
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET_URL = "https://github.com/mvaldenegro/marine-debris-fls-datasets/releases/download/watertank-v1.0/marine-debris-watertank-release-1.0.tar.bz2"
ARCHIVE_PATH = RAW_DATA_DIR / "marine-debris-watertank-release-1.0.tar.bz2"


def download_progress_hook(count, block_size, total_size):
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        mb_downloaded = count * block_size / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r[*] Downloading: {percent}% [{mb_downloaded:.1f} MB / {mb_total:.1f} MB]")
        sys.stdout.flush()


def main():
    print("=" * 70)
    print("Downloading Marine Debris FLS Dataset (Watertank Release 1.0)")
    print(f"Target: {RAW_DATA_DIR}")
    print("=" * 70)

    # 1. Download
    if not ARCHIVE_PATH.exists():
        print(f"[*] Fetching archive from:\n    {DATASET_URL}")
        try:
            urllib.request.urlretrieve(DATASET_URL, ARCHIVE_PATH, reporthook=download_progress_hook)
            print("\n[+] Download completed successfully!")
        except Exception as e:
            print(f"\n[!] Failed to download dataset: {e}")
            if ARCHIVE_PATH.exists():
                ARCHIVE_PATH.unlink()
            return
    else:
        print(f"[*] Archive already exists at: {ARCHIVE_PATH}")

    # 2. Extract
    print(f"[*] Extracting archive into: {RAW_DATA_DIR} ...")
    try:
        with tarfile.open(ARCHIVE_PATH, "r:bz2") as tar:
            tar.extractall(path=RAW_DATA_DIR)
        print("[+] Extraction finished successfully!")
    except Exception as e:
        print(f"[!] Error during extraction: {e}")
        return

    print("\n[+] Dataset is downloaded and ready in 'raw_data/'!")


if __name__ == "__main__":
    main()

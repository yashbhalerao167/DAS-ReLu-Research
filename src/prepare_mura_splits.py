# src/prepare_mura_splits.py

import os
import random
from pathlib import Path

# =========================
# CONFIG (LOCKED)
# =========================
PROJECT_ROOT = Path(r"D:/YASH/CODES/PROJECTS/Research Project")
MURA_ROOT = PROJECT_ROOT / "MURA-v1.1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_RATIO = 0.9
VAL_RATIO = 0.1
RANDOM_SEED = 42

# =========================
# UTILITIES
# =========================
def get_label(study_folder_name):
    if "positive" in study_folder_name:
        return 1
    elif "negative" in study_folder_name:
        return 0
    else:
        raise ValueError(f"Unknown label in folder name: {study_folder_name}")

def collect_images(split_name):
    split_dir = MURA_ROOT / split_name
    samples = []

    for body_part in split_dir.iterdir():
        if not body_part.is_dir():
            continue

        for patient in body_part.iterdir():
            for study in patient.iterdir():
                label = get_label(study.name)
                for img in study.glob("*.png"):
                     if img.name.startswith("._"):
                        continue

                rel_path = img.relative_to(MURA_ROOT)
                samples.append((rel_path.as_posix(), label))

    return samples

# =========================
# MAIN
# =========================
def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting train images...")
    train_samples = collect_images("train")
    random.shuffle(train_samples)

    split_idx = int(len(train_samples) * TRAIN_RATIO)
    train_set = train_samples[:split_idx]
    val_set = train_samples[split_idx:]

    print("Collecting test images...")
    test_set = collect_images("valid")

    def write_split(filename, data):
        with open(OUTPUT_DIR / filename, "w") as f:
            for path, label in data:
                f.write(f"{path} {label}\n")

    write_split("train.txt", train_set)
    write_split("val.txt", val_set)
    write_split("test.txt", test_set)

    print("Done.")
    print(f"Train: {len(train_set)}")
    print(f"Val:   {len(val_set)}")
    print(f"Test:  {len(test_set)}")

if __name__ == "__main__":
    main()

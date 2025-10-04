import os
import shutil
import random

def split_and_copy_files(source_dir, file_suffix, train_dir, unseen_dir, split_ratio=0.8):
    """Splits and copies CSV files by suffix into train and unseen sets."""
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(unseen_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(source_dir) if f.endswith(".csv") and file_suffix in f]
    random.seed(42)
    random.shuffle(csv_files)

    split_index = int(len(csv_files) * split_ratio)
    train_files = csv_files[:split_index]
    unseen_files = csv_files[split_index:]

    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))
    for f in unseen_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(unseen_dir, f))

    print(f"✅ {file_suffix}: {len(train_files)} train | {len(unseen_files)} unseen")
    print(f"📂 Train → {train_dir}")
    print(f"📂 Unseen → {unseen_dir}")


if __name__ == "__main__":
    BASE_SOURCE = "data/all_keypoints_csv"

    split_and_copy_files(
        source_dir=BASE_SOURCE,
        file_suffix="_mask_s",
        train_dir="data/train_keypoints_csv",
        unseen_dir="data/unseen_keypoints_csv"
    )

    split_and_copy_files(
        source_dir=BASE_SOURCE,
        file_suffix="_mask_c",
        train_dir="data/train_keypoints_csv_chair",
        unseen_dir="data/unseen_keypoints_csv_chair"
    )

    split_and_copy_files(
        source_dir=BASE_SOURCE,
        file_suffix="_mask_b",
        train_dir="data/train_keypoints_csv_bed",
        unseen_dir="data/unseen_keypoints_csv_bed"
    )

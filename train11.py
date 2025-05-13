import torch
import yaml
from ultralytics import YOLO
import os
import shutil
import random
from sklearn.model_selection import KFold
from pathlib import Path


def kfold_split(images_dir, labels_dir, num_folds=5, seed=42):
    # Get all image files
    image_files = sorted(list(images_dir.glob("*.png")))

    # Create KFold instance
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        print(f"\n🟡 Starting Fold {fold + 1}/{num_folds}...")

        fold_dir = images_dir.parent / f"fold{fold}"
        (fold_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (fold_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

        # Copy files to the respective fold train and val directories
        for idx in train_idx:
            img_path = image_files[idx]
            label_path = labels_dir / f"{img_path.stem}.txt"
            shutil.copy(img_path, fold_dir / "images" / "train" / img_path.name)
            shutil.copy(label_path, fold_dir / "labels" / "train" / label_path.name)

        for idx in val_idx:
            img_path = image_files[idx]
            label_path = labels_dir / f"{img_path.stem}.txt"
            shutil.copy(img_path, fold_dir / "images" / "val" / img_path.name)
            shutil.copy(label_path, fold_dir / "labels" / "val" / label_path.name)

        # Update data.yaml for the current fold
        data_yaml = {
            'train': str(fold_dir / "images" / "train"),
            'val': str(fold_dir / "images" / "val"),
            'nc': 28,
            'names': ["court", "building_52", "building_5153", "building_54", "building_55", "building_56",
                      "building_58", "building_60", "building_1", "building_2", "building_3", "building_4",
                      "building_5", "building_6", "building_7", "building_8", "building_9", "building_10",
                      "building_11", "building_3a", "building_ac", "building_21", "building_20", "soccer_field",
                      "building_police", "gate_uppercampus", "building_29", "building_30"]
        }

        with open(fold_dir / 'data.yaml', 'w') as yaml_file:
            yaml.dump(data_yaml, yaml_file)

        # Train the model for the current fold
        model = YOLO('yolo11n-seg.pt')  # Replace with your YOLOv11 model file
        results = model.train(data=str(fold_dir / 'data.yaml'), epochs=100, imgsz=640, device='cuda')

def main():
    # Check PyTorch version and CUDA availability
    print(torch.__version__)  # Check PyTorch version
    print(torch.cuda.is_available())  # Should return True if CUDA is available

    # Check if CUDA is available and set the device to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Print the device being used
    print(f"Using device: {device}")

    # Define directories for images and labels
    base_path = Path("C:/Users/danie/PycharmProjects/mymodel2/dataset_split")

    images_dir = base_path / "images" / "all"
    labels_dir = base_path / "labels" / "all"

    print(images_dir)
    print(labels_dir)

    # Start K-Fold cross-validation
    kfold_split(images_dir, labels_dir, num_folds=5)


if __name__ == '__main__':
    main()

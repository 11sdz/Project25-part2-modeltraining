import os
import shutil
import random

# Paths
dataset_path = "dataset2"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

# Output paths
output_path = "dataset_split"
train_images = os.path.join(output_path, "images/train")
val_images = os.path.join(output_path, "images/val")
train_labels = os.path.join(output_path, "labels/train")
val_labels = os.path.join(output_path, "labels/val")

# Create directories
for folder in [train_images, val_images, train_labels, val_labels]:
    os.makedirs(folder, exist_ok=True)

# List all images
all_images = [f for f in os.listdir(images_path) if f.endswith(".png")]

# Shuffle and split (80% train, 20% val)
random.shuffle(all_images)
split_index = int(len(all_images) * 0.8)
train_files = all_images[:split_index]
val_files = all_images[split_index:]

# Move images and corresponding labels
for file_list, img_dest, lbl_dest in [(train_files, train_images, train_labels), (val_files, val_images, val_labels)]:
    for img_file in file_list:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"

        # Move images
        shutil.move(os.path.join(images_path, img_file), os.path.join(img_dest, img_file))

        # Move labels (if exists)
        if os.path.exists(os.path.join(labels_path, label_file)):
            shutil.move(os.path.join(labels_path, label_file), os.path.join(lbl_dest, label_file))

print("✅ Dataset successfully split into train and val folders!")

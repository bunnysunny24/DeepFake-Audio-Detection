import os
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # <-- Import tqdm

# Initialize face detector
mtcnn = MTCNN(image_size=224, margin=20, post_process=True)

# Augmentation transformations
transform = A.Compose([
    A.GaussianBlur(p=0.2),
    A.GaussNoise(p=0.2),
    A.ImageCompression(quality=90, p=0.5),  # <-- Fixed ImageCompression issue
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# Extract face from an image
def extract_face(image_path):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)
    return face.permute(1, 2, 0).numpy() if face is not None else None

# Process image and save
def process_image(image_path, save_dir):
    face = extract_face(image_path)
    if face is not None:
        augmented = transform(image=face.astype(np.uint8))['image']
        filename = os.path.basename(image_path)
        output_img = (augmented.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, filename), output_img)

# Process dataset (Only `raw` images)
def process_dataset(input_folder, output_folder, split_ratio=0.2):
    if not os.path.exists(input_folder):
        print(f"❌ Error: Folder {input_folder} does not exist!")
        return

    image_files = [os.path.join(input_folder, img) for img in os.listdir(input_folder) if img.endswith(('.jpg', '.png'))]

    if not image_files:
        print(f"❌ Error: No images found in {input_folder}")
        return

    print(f"✅ Found {len(image_files)} images in {input_folder}")

    # Split into train and val sets
    train_files, val_files = train_test_split(image_files, test_size=split_ratio, random_state=42)

    for dataset_type, files in zip(["train", "val"], [train_files, val_files]):
        save_path = os.path.join(output_folder, dataset_type)
        os.makedirs(save_path, exist_ok=True)

        print(f"⚡ Processing {dataset_type} images ({len(files)} files)...")
        for image_file in tqdm(files, desc=f"Processing {dataset_type} images"):
            process_image(image_file, save_path)

# Paths (Only processing `raw`)
frames_raw = "../image_data/frames/raw"
output_raw = "../image_data/image_dataset-3/raw"

# Process only the raw dataset
print("🚀 Starting image processing...")
process_dataset(frames_raw, output_raw)
print("✅ Processing complete!")

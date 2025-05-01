import os
import cv2
import torch
import gc
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Progress bar

# Initialize face detector
mtcnn = MTCNN(image_size=224, margin=20, post_process=True)

# Augmentations
transform = A.Compose([
    A.GaussianBlur(p=0.2),
    A.GaussNoise(p=0.2),
    A.ImageCompression(quality=90, p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# Batch Size (Reduce if memory issues occur)
BATCH_SIZE = 50  # Reduced for memory optimization

# Extract face from image
def extract_face(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        face = mtcnn(img)
        if face is not None:
            return face.permute(1, 2, 0).numpy().astype(np.float32)  # Use float32 to reduce memory usage
        return None
    except Exception as e:
        print(f"❌ Error extracting face from {image_path}: {e}")
        return None

# Process batch of images
def process_batch(image_paths, save_dir):
    for image_path in image_paths:
        try:
            face = extract_face(image_path)
            if face is not None:
                augmented = transform(image=face)['image']
                filename = os.path.basename(image_path)
                output_img = (augmented.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # Ensure directory exists before saving
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, filename), output_img)
        except Exception as e:
            print(f"⚠️ Error processing {image_path}: {e}")

    # Free memory after processing each batch
    gc.collect()

# Resume processing dataset with batching
def continue_processing(category_folder):
    frame_dir = os.path.join(category_folder, "frames")
    train_dir = os.path.join(category_folder, "train")
    val_dir = os.path.join(category_folder, "val")

    # Ensure train and val directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    image_files = [os.path.join(frame_dir, img) for img in os.listdir(frame_dir) if img.endswith(".jpg")]

    # Skip already processed images
    processed_train = set(os.listdir(train_dir))
    processed_val = set(os.listdir(val_dir))
    remaining_files = [img for img in image_files if os.path.basename(img) not in processed_train and os.path.basename(img) not in processed_val]

    if not remaining_files:
        print("✅ All frames are already processed!")
        return

    # Split remaining files into train and validation sets
    train_files, val_files = train_test_split(remaining_files, test_size=0.2, random_state=42)

    # Process files in batches
    for dataset_type, files, save_path in [("train", train_files, train_dir), ("val", val_files, val_dir)]:
        print(f"⚡ Processing remaining {dataset_type} images ({len(files)} files)...")
        for i in tqdm(range(0, len(files), BATCH_SIZE), desc=f"Processing {dataset_type} images in batches"):
            batch = files[i:i + BATCH_SIZE]
            process_batch(batch, save_path)

# Paths
output_root = "../image_data/image_dataset-4"
category = "Celeb-real"  # You can also process "Celeb-synthesis" later

# Resume processing
continue_processing(os.path.join(output_root, category))

print("✅ Processing complete!")

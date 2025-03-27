import os
import cv2
import random
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# Initialize face detector
mtcnn = MTCNN(image_size=224, margin=20, post_process=True)

# Transform pipeline
transform = A.Compose([
    A.GaussianBlur(p=0.2),
    A.GaussNoise(p=0.2),
    A.ImageCompression(quality_lower=30, quality_upper=90, p=0.5),  # Fix incorrect parameters
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# Channel Random Erasing
def channel_random_erasing(img_tensor, p=0.5):
    if random.random() < p:
        channel = random.randint(0, img_tensor.shape[0] - 1)
        img_tensor[channel, :, :] = 0
    return img_tensor

# Extract face from image
def extract_face(image_path):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)
    if face is not None:
        return face.permute(1, 2, 0).numpy()
    return None

# Process image and extract face
def process_image(image_path, save_dir):
    face = extract_face(image_path)
    if face is not None:
        augmented = transform(image=face.astype(np.uint8))['image']
        final_img = channel_random_erasing(augmented)

        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_img = (final_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(save_dir, output_img)

# Process all images and split into train/val
def process_all_images(input_dir, output_dir, split_ratio=0.2):
    for label in ['real', 'fake']:
        image_dir = os.path.join(input_dir, label)
        image_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Split dataset into train and validation
        train_files, val_files = train_test_split(image_files, test_size=split_ratio, random_state=42)

        for dataset_type, files in zip(["train", "val"], [train_files, val_files]):
            save_label_path = os.path.join(output_dir, dataset_type, label)
            os.makedirs(save_label_path, exist_ok=True)

            for image_file in tqdm(files, desc=f"Processing {label} images for {dataset_type}"):
                output_path = os.path.join(save_label_path, os.path.basename(image_file))
                process_image(image_file, output_path)

# === MAIN ===
if __name__ == "__main__":
    input_folder = "../image_data/dataset"
    output_folder = "../image_data/image-dataset-2"
    process_all_images(input_folder, output_folder)

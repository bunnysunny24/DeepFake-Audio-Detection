import os
import cv2
import torch
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

# Extract frames from video
def extract_frames(video_path, save_dir, frame_interval=5):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(save_dir, f"{os.path.basename(video_path)}_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frames += 1
        frame_count += 1

    cap.release()
    return saved_frames

# Extract face from image
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

# Process dataset (Extract frames → Detect Faces → Augment → Save)
def process_videos(video_folder, output_folder, split_ratio=0.2):
    if not os.path.exists(video_folder):
        print(f"❌ Error: Folder {video_folder} does not exist!")
        return

    video_files = [os.path.join(video_folder, vid) for vid in os.listdir(video_folder) if vid.endswith(".mp4")]
    if not video_files:
        print(f"❌ No videos found in {video_folder}")
        return

    print(f"✅ Found {len(video_files)} videos in {video_folder}")

    # Temporary directory for extracted frames
    frame_dir = os.path.join(output_folder, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    # Extract frames from videos
    print("📽️ Extracting frames from videos...")
    for video in tqdm(video_files, desc="Extracting Frames"):
        extract_frames(video, frame_dir)

    # Process extracted frames
    image_files = [os.path.join(frame_dir, img) for img in os.listdir(frame_dir) if img.endswith(".jpg")]
    train_files, val_files = train_test_split(image_files, test_size=split_ratio, random_state=42)

    for dataset_type, files in zip(["train", "val"], [train_files, val_files]):
        save_path = os.path.join(output_folder, dataset_type)
        os.makedirs(save_path, exist_ok=True)

        print(f"⚡ Processing {dataset_type} images ({len(files)} files)...")
        for image_file in tqdm(files, desc=f"Processing {dataset_type} images"):
            process_image(image_file, save_path)

# Paths
celeb_df_root = "../image_data/Celeb_DF"
output_root = "../image_data/image_dataset-4"

# Process Real and Synthesis videos separately
for category in ["Celeb-real", "Celeb-synthesis"]:
    print(f"🚀 Processing {category} dataset...")
    process_videos(os.path.join(celeb_df_root, category), os.path.join(output_root, category))

print("✅ Processing complete!")

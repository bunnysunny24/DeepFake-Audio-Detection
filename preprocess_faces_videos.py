import os
import cv2
import random
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Initialize face detector
mtcnn = MTCNN(image_size=224, margin=20, post_process=True)

# Transform pipeline
transform = A.Compose([
    A.GaussianBlur(p=0.2),
    A.GaussNoise(p=0.2),
    A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# Channel Random Erasing
def channel_random_erasing(img_tensor, p=0.5):
    if random.random() < p:
        channel = random.randint(0, img_tensor.shape[0] - 1)
        img_tensor[channel, :, :] = 0
    return img_tensor

# Extract face from frame
def extract_face(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = mtcnn(img)
    if face is not None:
        return face.permute(1, 2, 0).numpy()
    return None

# Process video and extract faces
def process_video(video_path, save_dir, label, frame_skip=15):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            face = extract_face(frame)
            if face is not None:
                augmented = transform(image=face.astype(np.uint8))['image']
                final_img = channel_random_erasing(augmented)

                save_path = os.path.join(save_dir, label)
                os.makedirs(save_path, exist_ok=True)
                filename = os.path.splitext(os.path.basename(video_path))[0]
                output_img = (final_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_file = f"{filename}_{saved_count}.jpg"
                cv2.imwrite(os.path.join(save_path, img_file), output_img)
                saved_count += 1

        frame_count += 1
    cap.release()

# Process all videos in folder
def process_all_videos(input_dir, output_dir):
    for label in ['real', 'fake']:
        video_dir = os.path.join(input_dir, label)
        for video_file in tqdm(os.listdir(video_dir), desc=f"Processing {label} videos"):
            video_path = os.path.join(video_dir, video_file)
            if video_file.endswith('.mp4'):
                process_video(video_path, os.path.join(output_dir), label)

# === MAIN ===
if __name__ == "__main__":
    input_folder = "../image_data/FF++"
    output_folder = "../image_data/image-dataset-1/train"
    process_all_videos(input_folder, output_folder)

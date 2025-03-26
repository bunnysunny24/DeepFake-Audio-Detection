import os
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Paths from your previous code
input_folder = "../image_data/FF++"
output_folder = "../image_data/image-dataset-5/train"

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=224, margin=20, post_process=True, device="cuda" if torch.cuda.is_available() else "cpu")

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
    if torch.rand(1).item() < p and img_tensor.shape[0] == 3:  # Ensure RGB
        channel = torch.randint(0, 3, (1,)).item()  # Choose R, G, or B channel
        img_tensor[channel, :, :] = 0
    return img_tensor

# Extract face from a frame
def extract_face(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = mtcnn(img)
    if face is None:
        print("[INFO] No face detected in frame.")
        return None
    return face.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)

# Process a single video
def process_single_video(video_path, save_dir, frame_skip=15):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            face = extract_face(frame)
            if face is not None:
                # Save original face for debugging
                cv2.imwrite("debug_original.jpg", (face * 255).astype(np.uint8))

                # Apply augmentation
                augmented = transform(image=face.astype(np.uint8))["image"]
                final_img = channel_random_erasing(augmented)

                # Save augmented face for debugging
                debug_img = final_img.permute(1, 2, 0).cpu().numpy() * 255  # Convert to (H, W, C)
                cv2.imwrite("debug_augmented.jpg", debug_img.astype(np.uint8))

                # Save final processed image
                save_path = os.path.join(save_dir, "debug_output")
                os.makedirs(save_path, exist_ok=True)
                filename = os.path.basename(video_path).split(".")[0]
                img_file = f"{filename}_{saved_count}.jpg"
                cv2.imwrite(os.path.join(save_path, img_file), debug_img.astype(np.uint8))
                saved_count += 1

        frame_count += 1

    cap.release()
    print("[INFO] Processing completed.")

# Pick one video from the dataset
def get_one_video():
    for label in ["real", "fake"]:
        video_dir = os.path.join(input_folder, label)
        if os.path.exists(video_dir):
            video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
            if video_files:
                return os.path.join(video_dir, video_files[0])  # Pick the first video
    print("[ERROR] No videos found in dataset.")
    return None

# === Run the Debugging Test ===
if __name__ == "__main__":
    test_video_path = get_one_video()
    if test_video_path:
        print(f"[INFO] Processing test video: {test_video_path}")
        process_single_video(test_video_path, output_folder)
    else:
        print("[ERROR] No video found in dataset.")

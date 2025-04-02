import os
import cv2
import random
from tqdm import tqdm

DATASET_PATH = "D:/Bunny/Deepfake/backend/image_data/FF++"
FAKE_PATH = os.path.join(DATASET_PATH, "fake")
REAL_PATH = os.path.join(DATASET_PATH, "real")
OUTPUT_PATH = "D:/Bunny/Deepfake/backend/image_data/image-dataset-7"
TRAIN_PATH = os.path.join(OUTPUT_PATH, "train")
VAL_PATH = os.path.join(OUTPUT_PATH, "validation")

for path in [TRAIN_PATH, VAL_PATH]:
    os.makedirs(os.path.join(path, "fake"), exist_ok=True)
    os.makedirs(os.path.join(path, "real"), exist_ok=True)

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_file = os.path.join(output_folder, f"{count:04d}.jpg")
        cv2.imwrite(frame_file, frame)
        count += 1
    cap.release()

def process_videos(source_folder, label):
    videos = [f for f in os.listdir(source_folder) if f.endswith('.mp4')]
    random.shuffle(videos)
    train_split = int(0.8 * len(videos))
    train_videos = videos[:train_split]
    val_videos = videos[train_split:]
    for phase, video_list in zip(["train", "validation"], [train_videos, val_videos]):
        for video in tqdm(video_list, desc=f"Processing {phase} - {label}"):
            video_path = os.path.join(source_folder, video)
            output_folder = os.path.join(OUTPUT_PATH, phase, label, video.replace(".mp4", ""))
            os.makedirs(output_folder, exist_ok=True)
            extract_frames(video_path, output_folder)

process_videos(FAKE_PATH, "fake")
process_videos(REAL_PATH, "real")
print("✅ All videos processed into frames and saved in image-dataset-7!")

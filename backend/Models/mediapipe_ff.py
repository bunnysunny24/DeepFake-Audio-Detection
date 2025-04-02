import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm

predictor_path = r"D:\Bunny\Deepfake\backend\Models\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
DATASET_PATH = r"D:\Bunny\Deepfake\backend\image_data\image-dataset-7"
HEATMAP_PATH = r"D:\Bunny\Deepfake\backend\image_data\landmark_heatmaps_7"
os.makedirs(HEATMAP_PATH, exist_ok=True)

def process_images(image_folder, heatmap_folder):
    if not os.path.exists(image_folder):
        print(f"⚠️ Skipping {image_folder} (folder not found)")
        return
    for root, _, files in os.walk(image_folder):  
        for image_file in tqdm(files, desc=f"Processing {root}"):
            if not image_file.endswith(".jpg"):
                continue
            image_path = os.path.join(root, image_file)
            relative_path = os.path.relpath(image_path, DATASET_PATH)
            heatmap_path = os.path.join(HEATMAP_PATH, relative_path.replace(".jpg", "_heatmap.jpg"))
            print(f"🔍 Reading: {image_path}") 
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"❌ ERROR: Cannot read {image_path}")
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_frame)
            if len(faces) == 0:
                print(f"⚠️ No faces detected in {image_path}")
                continue  # Skip if no faces

            heatmap = np.zeros_like(frame)
            for face in faces:
                landmarks = predictor(gray_frame, face)
                for i in range(68):  # 68 facial landmarks
                    x, y = landmarks.part(i).x, landmarks.part(i).y
                    cv2.circle(heatmap, (x, y), 1, (255, 255, 255), -1)
            os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
            cv2.imwrite(heatmap_path, heatmap)
            print(f"✅ Heatmap saved: {heatmap_path}")

for phase in ["train", "validation"]:
    for label in ["fake", "real"]:
        image_folder = os.path.join(DATASET_PATH, phase, label)
        heatmap_folder = os.path.join(HEATMAP_PATH, phase, label)
        process_images(image_folder, heatmap_folder)

print("✅ Landmark heatmaps generated and saved successfully!")
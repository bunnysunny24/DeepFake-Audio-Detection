import os
import cv2
import numpy as np
import torch
import dlib
from torchvision import transforms

# Paths to your dataset
dataset_path = "../image_data/image-dataset-1"
train_path = os.path.join(dataset_path, "../image_data/image-dataset-1/train")
val_path = os.path.join(dataset_path, "../image_data/image-dataset-1/validation")

# Initialize dlib face detector & landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to generate heatmap from landmarks
def generate_heatmap(image, landmarks, img_size=(224, 224), sigma=5):
    heatmap = np.zeros((img_size[0], img_size[1]))
    for (x, y) in landmarks:
        for i in range(-sigma, sigma):
            for j in range(-sigma, sigma):
                if 0 <= x+i < img_size[0] and 0 <= y+j < img_size[1]:
                    heatmap[y+j, x+i] += np.exp(-(i**2 + j**2) / (2.0 * sigma**2))
    return heatmap

# Function to extract facial landmarks
def extract_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None, None
    
    landmarks = predictor(gray, faces[0])
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    return np.array(points), generate_heatmap(image, points)

# Preprocess images and save output
def preprocess_images(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    landmarks, heatmap = extract_landmarks(image)
    if landmarks is None:
        return None, None
    
    landmarks_vector = landmarks.flatten()  # Convert 68 points to 136D vector
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Save output in the same folder
    base_name = os.path.splitext(image_path)[0]
    np.save(base_name + "_landmarks.npy", landmarks_vector)  # Save landmarks as .npy
    heatmap_path = base_name + "_heatmap.jpg"
    cv2.imwrite(heatmap_path, (heatmap * 255).astype(np.uint8))  # Save heatmap
    
    return torch.tensor(landmarks_vector, dtype=torch.float32), torch.tensor(heatmap, dtype=torch.float32)

# Process dataset
train_data = []
for img_file in os.listdir(train_path):
    img_path = os.path.join(train_path, img_file)
    landmarks_vec, heatmap = preprocess_images(img_path)
    if landmarks_vec is not None:
        train_data.append((landmarks_vec, heatmap))

print(f"Processed {len(train_data)} training images with landmarks and heatmaps! Output saved in respective folders.")

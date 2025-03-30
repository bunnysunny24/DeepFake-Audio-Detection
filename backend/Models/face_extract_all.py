import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import sys
import os

# Add face-parsing.PyTorch to system path
sys.path.append(r"D:\Bunny\Deepfake\backend\face-parsing.PyTorch")

from model import BiSeNet  # Import the model

# Define classes to extract
FEATURES = {
    1: "skin", 2: "left_brow", 3: "right_brow", 4: "left_eye", 5: "right_eye",
    6: "eye_glass", 7: "left_ear", 8: "right_ear", 9: "ear_ring", 10: "nose",
    11: "mouth", 12: "upper_lip", 13: "lower_lip", 14: "neck", 15: "necklace",
    16: "cloth", 17: "hair", 18: "hat", 19: "background"
}

# Load the model
def load_model(model_path):
    print(f"[INFO] Loading model from: {model_path}")
    try:
        net = BiSeNet(n_classes=19)
        net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        net.eval()
        print("[INFO] Model loaded successfully.")
        return net
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

# Preprocess input image
def preprocess_image(image_path):
    try:
        print(f"[INFO] Preprocessing image: {image_path}")
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)
        
        print(f"[DEBUG] Image Tensor Shape: {img_tensor.shape}")
        print(f"[DEBUG] Image Tensor Min: {img_tensor.min()}, Max: {img_tensor.max()}")
        
        return img_tensor
    except Exception as e:
        print(f"[ERROR] Failed to preprocess image: {e}")
        return None

# Segment and save features
def segment_and_save(image_path, model, save_dir):
    print(f"[INFO] Processing image: {image_path}")
    img_tensor = preprocess_image(image_path)
    
    if img_tensor is None:
        print(f"[ERROR] Skipping {image_path} due to preprocessing failure.")
        return

    try:
        with torch.no_grad():
            output = model(img_tensor)[0].squeeze(0).cpu().numpy()
        
        print(f"[DEBUG] Model output shape: {output.shape}")  # Should be (19, 512, 512)
        segmentation_map = np.argmax(output, axis=0)
        unique_classes = np.unique(segmentation_map)
        print(f"[DEBUG] Unique class values in segmentation map: {unique_classes}")

        os.makedirs(save_dir, exist_ok=True)

        # Save individual feature masks
        for class_id, feature_name in FEATURES.items():
            mask = (segmentation_map == class_id).astype(np.uint8) * 255
            feature_path = os.path.join(save_dir, f"{feature_name}.png")
            cv2.imwrite(feature_path, mask)
            print(f"[INFO] Saved {feature_name} mask to: {feature_path}")

        # Save colorized segmentation map for visualization
        color_map = np.zeros((512, 512, 3), dtype=np.uint8)
        np.random.seed(42)  # Fix colors for consistency
        colors = {i: tuple(np.random.randint(0, 255, 3)) for i in FEATURES.keys()}

        for class_id, color in colors.items():
            color_map[segmentation_map == class_id] = color

        color_seg_path = os.path.join(save_dir, "segmentation_color.png")
        cv2.imwrite(color_seg_path, color_map)
        print(f"[INFO] Saved colorized segmentation map to: {color_seg_path}")

    except Exception as e:
        print(f"[ERROR] Failed to process image {image_path}: {e}")

# Run on dataset
def process_dataset(dataset_dir, model_path, output_dir):
    model = load_model(model_path)
    processed = 0  # Track processed images
    
    for root, _, files in os.walk(dataset_dir):
        for file_name in files:
            if file_name.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(root, dataset_dir)
                save_path = os.path.join(output_dir, relative_path, os.path.splitext(file_name)[0])
                os.makedirs(save_path, exist_ok=True)
                
                print(f"[INFO] Processing {file_name} ({processed + 1})")
                segment_and_save(image_path, model, save_path)
                
                processed += 1
                if processed % 5 == 0:  # Print update every 5 images
                    print(f"[INFO] Processed {processed} images so far...")

    print(f"[INFO] Processing complete! Total images processed: {processed}")

# Example usage
DATASET_DIR = "D:/Bunny/Deepfake/backend/image_data/image-dataset-7"
MODEL_PATH = "D:/Bunny/Deepfake/backend/face-parsing.PyTorch/res/79999_iter.pth"  # Path to pre-trained model weights
OUTPUT_DIR = "D:/Bunny/Deepfake/backend/image_data/segmented_output_chec1"

process_dataset(DATASET_DIR, MODEL_PATH, OUTPUT_DIR)

import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from tqdm import tqdm
import sys

# Add the path to face-parsing.PyTorch so Python can find model.py
sys.path.append(r"D:\Bunny\Deepfake\backend\face-parsing.PyTorch")

# Now you can import BiSeNet
from model import BiSeNet

# Paths
base_dir = r"D:\Bunny\Deepfake\backend\image_data"
image_dir = os.path.join(base_dir, "image-dataset-7")
face_dir = os.path.join(base_dir, "face_7")
blur_dir = os.path.join(base_dir, "blur_7")
gray_dir = os.path.join(base_dir, "gray_7")

# Create directories
os.makedirs(os.path.join(face_dir, "train", "real"), exist_ok=True)
os.makedirs(os.path.join(blur_dir, "train", "real"), exist_ok=True)
os.makedirs(os.path.join(gray_dir, "train", "real"), exist_ok=True)

# Load Pretrained BiSeNet Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_classes = 19  # CelebA-HQ has 19 face regions
model = BiSeNet(n_classes=n_classes).to(device)
model_path = r"D:\Bunny\Deepfake\backend\face-parsing.PyTorch\res\79999_iter.pth"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Extract and Process Faces
def process_faces():
    input_folder = os.path.join(image_dir, "train", "real")
    face_output_folder = os.path.join(face_dir, "train", "real")
    blur_output_folder = os.path.join(blur_dir, "train", "real")
    gray_output_folder = os.path.join(gray_dir, "train", "real")

    for file_name in tqdm(os.listdir(input_folder), desc="Processing Train/Real Faces"):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        input_file = os.path.join(input_folder, file_name)
        image = Image.open(input_file).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Get Face Segmentation
        with torch.no_grad():
            out = model(image_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

        # Convert Mask to Binary Mask (Extract Full Face)
        mask = np.isin(parsing, [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13])  # Face, ears, lips, nose, etc.
        mask = mask.astype(np.uint8) * 255

        # Apply Mask on Image
        image_np = np.array(image)
        segmented_face = cv2.bitwise_and(image_np, image_np, mask=mask)

        # Convert back to PIL
        face_image = Image.fromarray(segmented_face)

        # Save Extracted Face
        face_image.save(os.path.join(face_output_folder, file_name))

        # Apply Gaussian Blur
        blurred_face = face_image.filter(ImageFilter.GaussianBlur(5))
        blurred_face.save(os.path.join(blur_output_folder, file_name))

        # Convert to Grayscale
        gray_face = face_image.convert("L")
        gray_face.save(os.path.join(gray_output_folder, file_name))

# Run Processing
process_faces()
print("✅ Full Face Extraction, Blurring, and Grayscale Processing Completed!")

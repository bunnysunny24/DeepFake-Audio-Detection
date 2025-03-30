import torch
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Dataset paths
DATASET_PATH = r"D:\Bunny\Deepfake\backend\image_data\image-dataset-7"
TRAIN_IMAGE_DIR = os.path.join(DATASET_PATH, "train", "fake")
TRAIN_MASK_DIR = os.path.join(DATASET_PATH, "train", "fake_masks")
VAL_IMAGE_DIR = os.path.join(DATASET_PATH, "validation", "fake")
VAL_MASK_DIR = os.path.join(DATASET_PATH, "validation", "fake_masks")

# Feature Mapping
FEATURES = {0: "background", 1: "skin", 2: "left_brow", 3: "right_brow",
            4: "left_eye", 5: "right_eye", 6: "eye_glass", 7: "left_ear",
            8: "right_ear", 9: "ear_ring", 10: "nose", 11: "mouth",
            12: "upper_lip", 13: "lower_lip", 14: "neck", 15: "necklace",
            16: "cloth", 17: "hair", 18: "hat"}

class FaceSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx].replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask) * 18

        return image, mask.long()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

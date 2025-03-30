import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class HybridDataset(Dataset):
    def __init__(self, image_dir, heatmap_dir, ear_dir, optical_flow_dir,
                 transform_img=None, transform_heatmap=None, transform_optical=None):
        self.image_paths = []
        self.heatmap_paths = []
        self.ear_paths = []
        self.optical_flow_paths = []
        self.labels = []
        
        self.transform_img = transform_img
        self.transform_heatmap = transform_heatmap
        self.transform_optical = transform_optical
        self.label_map = {"real": 0, "fake": 1}  # Assign labels

        for label in ["real", "fake"]:
            img_folder = os.path.join(image_dir, label)
            heatmap_folder = os.path.join(heatmap_dir, label)
            ear_folder = os.path.join(ear_dir, label)
            optical_flow_folder = os.path.join(optical_flow_dir, label)

            for img_name in os.listdir(img_folder):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  
                    continue  

                img_path = os.path.join(img_folder, img_name)
                
                base_name, ext = os.path.splitext(img_name)
                heatmap_path = os.path.join(heatmap_folder, f"{base_name}_heatmap{ext}")
                ear_path = os.path.join(ear_folder, f"{base_name}_ear{ext}")
                optical_flow_path = os.path.join(optical_flow_folder, f"{base_name}_optical{ext}")

                if os.path.exists(heatmap_path) and os.path.exists(ear_path) and os.path.exists(optical_flow_path):
                    self.image_paths.append(img_path)
                    self.heatmap_paths.append(heatmap_path)
                    self.ear_paths.append(ear_path)
                    self.optical_flow_paths.append(optical_flow_path)
                    self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        heatmap = Image.open(self.heatmap_paths[idx]).convert("L")  # Grayscale
        ear = Image.open(self.ear_paths[idx]).convert("RGB")
        optical_flow = Image.open(self.optical_flow_paths[idx]).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform_img:
            image = self.transform_img(image)
            ear = self.transform_img(ear)  
            optical_flow = self.transform_optical(optical_flow)

        if self.transform_heatmap:
            heatmap = self.transform_heatmap(heatmap)

        return image, heatmap, ear, optical_flow, label

# Define Transformations
transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

transform_heatmap = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

# Optical flow transformation is the same as images
transform_optical = transform_img

def get_dataloaders(batch_size=32):
    train_dataset = HybridDataset(
        image_dir=r"D:\Bunny\Deepfake\backend\image_data\image-dataset-7\train",
        heatmap_dir=r"D:\Bunny\Deepfake\backend\image_data\landmark_heatmaps_7\train",
        ear_dir=r"D:\Bunny\Deepfake\backend\image_data\ear_7\train",
        optical_flow_dir=r"D:\Bunny\Deepfake\backend\image_data\optical_flow_7\train",
        transform_img=transform_img,
        transform_heatmap=transform_heatmap,
        transform_optical=transform_optical  
    )

    val_dataset = HybridDataset(
        image_dir=r"D:\Bunny\Deepfake\backend\image_data\image-dataset-7\validation",
        heatmap_dir=r"D:\Bunny\Deepfake\backend\image_data\landmark_heatmaps_7\validation",
        ear_dir=r"D:\Bunny\Deepfake\backend\image_data\ear_7\validation",
        optical_flow_dir=r"D:\Bunny\Deepfake\backend\image_data\optical_flow_7\validation",
        transform_img=transform_img,
        transform_heatmap=transform_heatmap,
        transform_optical=transform_optical  
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


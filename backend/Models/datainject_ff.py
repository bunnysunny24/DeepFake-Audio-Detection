import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class HybridDataset(Dataset):
    def __init__(self, image_dir, heatmap_dir, transform=None):
        self.image_paths = []
        self.heatmap_paths = []
        self.labels = []
        self.transform = transform
        self.label_map = {"real": 0, "fake": 1}  # Assign labels

        for label in ["real", "fake"]:
            img_folder = os.path.join(image_dir, label)
            heatmap_folder = os.path.join(heatmap_dir, label)

            for img_name in os.listdir(img_folder):
                img_path = os.path.join(img_folder, img_name)

                # Modify heatmap filename to match dataset structure
                base_name, ext = os.path.splitext(img_name)  # Get filename without extension
                heatmap_name = f"{base_name}_heatmap{ext}"  # Append "_heatmap" before extension
                heatmap_path = os.path.join(heatmap_folder, heatmap_name)

                if os.path.exists(heatmap_path):  # Only add matching pairs
                    self.image_paths.append(img_path)
                    self.heatmap_paths.append(heatmap_path)
                    self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        heatmap = Image.open(self.heatmap_paths[idx]).convert("L")  # Grayscale
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)
            heatmap = self.transform(heatmap)

        return image, heatmap, label

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
])

def get_dataloaders(batch_size=32):
    train_dataset = HybridDataset(
        image_dir=r"D:\Bunny\Deepfake\backend\image_data\image-dataset-7\train",
        heatmap_dir=r"D:\Bunny\Deepfake\backend\image_data\landmark_heatmaps_7\train",
        transform=transform
    )

    val_dataset = HybridDataset(
        image_dir=r"D:\Bunny\Deepfake\backend\image_data\image-dataset-7\validation",
        heatmap_dir=r"D:\Bunny\Deepfake\backend\image_data\landmark_heatmaps_7\validation",
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

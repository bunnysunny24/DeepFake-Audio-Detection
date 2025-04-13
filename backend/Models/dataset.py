import json
import torch
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        video_frames = torch.tensor(entry["frames"], dtype=torch.float32)  # Normalize if needed
        audio_features = torch.tensor(entry["audio"], dtype=torch.float32)
        label = torch.tensor(entry["label"], dtype=torch.float32)
        
        return {
            "video_frames": video_frames,
            "audio_features": audio_features,
            "labels": label
        }
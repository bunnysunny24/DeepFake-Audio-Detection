import json
import os
import torch
from dataset_loader import MultiModalDeepfakeDataset

def check_for_none_values(json_path, data_dir):
    dataset = MultiModalDeepfakeDataset(
        json_path=json_path,
        data_dir=data_dir,
        transform=None,
        audio_transform=None,
        logging=True
    )

    num_invalid_samples = 0
    invalid_samples = []

    for idx in range(len(dataset)):
        sample = dataset[idx]

        if sample is None:
            num_invalid_samples += 1
            invalid_samples.append(idx)
            print(f"Invalid sample at index {idx}: Missing video or audio.")

    if num_invalid_samples > 0:
        print(f"\nTotal invalid samples: {num_invalid_samples}")
        print(f"Indices of invalid samples: {invalid_samples}")
    else:
        print("No invalid samples found in the dataset.")

# Example usage
json_path = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF\metadata.json"
data_dir = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF"


check_for_none_values(json_path, data_dir)

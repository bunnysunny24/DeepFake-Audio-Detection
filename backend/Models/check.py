import os
from tqdm import tqdm
from collections import defaultdict
import torch
from dataset_loader import MultiModalDeepfakeDataset  # Replace with the correct file/module name

def check_dataset_integrity(json_path, data_dir, num_samples_to_check=100):
    dataset = MultiModalDeepfakeDataset(
        json_path=json_path,
        data_dir=data_dir,
        transform=None,
        audio_transform=None,
        logging=False
    )

    issues = defaultdict(list)
    total_checked = 0

    for idx in tqdm(range(min(len(dataset), num_samples_to_check)), desc="Checking samples"):
        sample = dataset[idx]
        total_checked += 1

        if sample is None:
            issues["None sample"].append(idx)
            continue

        # Check required fields for all samples
        if sample.get('video_frames') is None:
            issues["Missing video_frames"].append(idx)
        if sample.get('audio') is None:
            issues["Missing audio"].append(idx)
        if sample.get('label') is None:
            issues["Missing label"].append(idx)
        if not isinstance(sample.get('fake_mask'), torch.Tensor):
            issues["Missing or invalid fake_mask"].append(idx)
        if 'timestamps' not in sample or not isinstance(sample['timestamps'], list):
            issues["Missing or invalid timestamps"].append(idx)
        if 'transcript' not in sample or not isinstance(sample['transcript'], str):
            issues["Missing or invalid transcript"].append(idx)

        # Check original fields only for fake samples
        if sample.get('n_fakes', 0) > 0:  # Only for fake samples
            if sample.get('original_video_frames') is None:
                issues["Missing original_video_frames (for fake sample)"].append(idx)
            if sample.get('original_audio') is None:
                issues["Missing original_audio (for fake sample)"].append(idx)

    # Summary
    print(f"\n✅ Checked {total_checked} samples.\n")
    if not issues:
        print("🎉 All samples are valid!")
    else:
        print("⚠️ Issues found in the dataset:")
        for issue, indices in issues.items():
            print(f"- {issue}: {len(indices)} occurrences (e.g., {indices[:5]})")

# Example usage
check_dataset_integrity(
    json_path=r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF\metadata.json",
    data_dir=r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF",
    num_samples_to_check=200  # or len(dataset) if you want to check everything
)

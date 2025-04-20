import os
from dataset_loader import MultiModalDeepfakeDataset
from tqdm import tqdm

json_path = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF\metadata.json"
data_dir = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF"

# Create dataset instance and enable quick check mode
dataset = MultiModalDeepfakeDataset(
    json_path=json_path,
    data_dir=data_dir,
    logging=False
)
dataset.check_only = True  # <--- ✅ Fast check mode enabled

none_samples = []

print("\n⚡ Fast scan for missing files...")
for i in tqdm(range(len(dataset))):
    if dataset[i] is None:
        none_samples.append(i)

print(f"\n🚫 Total broken samples: {len(none_samples)}")
if none_samples:
    print("Problematic indices:")
    for i in none_samples:
        print(f"  Sample {i}")
else:
    print("✅ All samples seem fine.")

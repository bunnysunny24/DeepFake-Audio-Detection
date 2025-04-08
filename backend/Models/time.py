import json
import os

# Path to your metadata.json
metadata_path = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF\metadata.json"

# Time per file in seconds
time_per_file = 10

if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        data = json.load(f)
    
    total_files = len(data)
    total_time_seconds = total_files * time_per_file
    total_minutes = total_time_seconds / 60
    total_hours = total_minutes / 60

    print(f"Total files: {total_files}")
    print(f"Total processing time:")
    print(f"  - {total_time_seconds} seconds")
    print(f"  - {total_minutes:.2f} minutes")
    print(f"  - {total_hours:.2f} hours")
else:
    print(f"File not found at: {metadata_path}")

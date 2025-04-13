import json
import os
from collections import Counter

# Path to your metadata.json
metadata_path = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF\metadata.json"

def find_all_files(data):
    """Recursively find all filenames in the JSON data."""
    all_files = []
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, (dict, list)):
                all_files.extend(find_all_files(value))
            elif isinstance(value, str):
                all_files.append(value)
    elif isinstance(data, list):
        for item in data:
            all_files.extend(find_all_files(item))
    return all_files

def find_mp4_files(data):
    """Recursively find .mp4 filenames in the JSON data."""
    return [file for file in find_all_files(data) if file.lower().endswith(".mp4")]

if os.path.exists(metadata_path):
    try:
        with open(metadata_path, "r") as f:
            data = json.load(f)

        # Find all files and .mp4 files
        all_files = find_all_files(data)
        mp4_files = find_mp4_files(data)

        # Count duplicates
        file_counts = Counter(all_files)
        duplicates = [file for file, count in file_counts.items() if count > 1]

        # Output results
        print(f"Total files: {len(all_files)}")
        print(f"Total .mp4 files: {len(mp4_files)}")
        print(f"Duplicate files: {len(duplicates)}")
        if duplicates:
            print("Sample duplicate files:")
            print(duplicates[:5])  # Display first 5 duplicates as a sample
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print(f"File not found at: {metadata_path}")
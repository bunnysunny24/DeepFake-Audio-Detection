import json

# Path to the processed dataset
processed_data_path = r"D:\Bunny\Deepfake\backend\combined_data\processed_data_cleaned_v2.json"

def validate_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Check for missing or corrupted entries
    for i, entry in enumerate(data):
        if not entry.get("file") or not entry.get("frames") or not entry.get("audio"):
            print(f"Missing data in entry {i}: {entry}")
        if not isinstance(entry["label"], int):
            print(f"Invalid label in entry {i}: {entry}")
    
    print(f"Dataset validation complete. Total entries: {len(data)}")

if __name__ == "__main__":
    validate_dataset(processed_data_path)
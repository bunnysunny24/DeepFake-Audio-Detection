import json

def validate_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"File is valid. Total entries: {len(data)}")
        return data
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e.msg}")
        print(f"Line: {e.lineno}, Column: {e.colno}, Char position: {e.pos}")
        return None

def inspect_entries(data):
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            print(f"Invalid entry at index {i}: Not a dictionary")
        elif "file" not in entry or "frames" not in entry or "audio" not in entry:
            print(f"Missing required keys in entry at index {i}: {entry}")
        elif not entry["frames"] or not entry["audio"]:
            print(f"Empty frames or audio in entry at index {i}: {entry}")

if __name__ == "__main__":
    file_path = r"D:\Bunny\Deepfake\backend\combined_data\processed_data.json"
    data = validate_json(file_path)
    if data:
        inspect_entries(data)
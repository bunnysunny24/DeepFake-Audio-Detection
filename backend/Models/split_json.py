import json

# Path to the large JSON file
input_file = r"D:\Bunny\Deepfake\backend\combined_data\processed_data.json"
output_dir = r"D:\Bunny\Deepfake\backend\combined_data\chunks"

def split_json(file_path, output_dir, chunk_size=1000):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunk_path = f"{output_dir}/chunk_{i // chunk_size + 1}.json"
        with open(chunk_path, "w", encoding="utf-8") as chunk_file:
            json.dump(chunk, chunk_file, indent=4)

    print(f"JSON file split into chunks of {chunk_size} entries.")

if __name__ == "__main__":
    split_json(input_file, output_dir)
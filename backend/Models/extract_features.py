import os

# Paths to segmented image directories
segmented_dir = "D:/Bunny/Deepfake/backend/image_data/segmented_mediapipe/train/real"
original_dir = "D:/Bunny/Deepfake/backend/image_data/image-dataset-7/train/real"

# Mapping dictionary to match original filenames
file_map = {}

# Create a mapping of image ID to original filename
for filename in os.listdir(original_dir):
    if filename.endswith((".jpg", ".png")):
        name, ext = os.path.splitext(filename)
        file_map[name] = filename  # Store full name for later lookup

# Rename segmented images
for file in os.listdir(segmented_dir):
    if file.startswith("image-"):
        parts = file.split("_")
        image_id = parts[1]  # Extract the numeric ID (e.g., 100000)
        feature_name = "_".join(parts[2:])  # Extract the feature name
        feature_name = feature_name.replace(".png", ".jpg")  # Ensure JPG extension
        
        # Find matching original filename
        for key in file_map:
            if key.endswith(image_id):  # Match based on ID
                new_name = f"{file_map[key].replace('.jpg', '')}_seg_{feature_name}"
                old_path = os.path.join(segmented_dir, file)
                new_path = os.path.join(segmented_dir, new_name)

                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {file} -> {new_name}")
                except FileExistsError:
                    print(f"[ERROR] File already exists: {new_name}")

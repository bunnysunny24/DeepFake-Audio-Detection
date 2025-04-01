import os
import re

# Base directories
original_dir = "D:/Bunny/Deepfake/backend/image_data/image-dataset-7/train/fake"
segmented_dir = "D:/Bunny/Deepfake/backend/image_data/segmented_mediapipe/train/fake"

# Function to extract frame number from original filenames
def extract_frame_number(filename):
    match = re.search(r"_(\d{4})\.jpg$", filename)
    return int(match.group(1)) if match else None

# Get sorted list of original filenames
original_files = sorted(
    [f for f in os.listdir(original_dir) if f.endswith(".jpg")],
    key=lambda x: extract_frame_number(x) if extract_frame_number(x) is not None else float('inf')
)

# Create mapping of frame number to filename
frame_to_filename = {}
for filename in original_files:
    frame_number = extract_frame_number(filename)
    if frame_number is not None:
        frame_to_filename[frame_number] = filename  # Store mapping

# Debugging: Print available frame numbers
print("Extracted frame numbers:", list(frame_to_filename.keys()))  # Debug
print("Segmented image IDs detected:", set(re.findall(r"image-(\d+)", " ".join(os.listdir(segmented_dir)))))  # Debug

# Process segmented images
for file in os.listdir(segmented_dir):
    match = re.match(r"image-(\d+)_([a-zA-Z_]+)\.png$", file)
    if match:
        image_id = int(match.group(1))  # Extract numeric ID
        feature_name = match.group(2)  # Extract feature name

        # Check if image_id exists in frame_to_filename
        if image_id in frame_to_filename:
            original_filename = frame_to_filename[image_id]
            new_name = f"{original_filename.replace('.jpg', '')}_{feature_name}.png"

            old_path = os.path.join(segmented_dir, file)
            new_path = os.path.join(segmented_dir, new_name)

            # Rename the file
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"Renamed: {file} -> {new_name}")
            else:
                print(f"[ERROR] File already exists: {new_name}")
        else:
            print(f"[ERROR] No matching original filename for image-{image_id}")  # Debugging issue
    else:
        print(f"[ERROR] Failed to parse {file}")  # Debugging for unrecognized filenames

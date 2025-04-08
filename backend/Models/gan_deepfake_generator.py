import os
import json
import numpy as np
from PIL import Image

# Paths
generated_dir = 'generated_deepfakes'
existing_json_path = 'D:\\Bunny\\Deepfake\\backend\\combined_data\\processed_data.json'
updated_json_path = 'D:\\Bunny\\Deepfake\\backend\\combined_data\\updated_processed_data.json'

# Function to load images from the generated directory
def load_generated_images(generated_dir):
    generated_data = []
    for filename in os.listdir(generated_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(generated_dir, filename)
            image = Image.open(image_path)
            image_array = np.array(image)
            generated_data.append({
                "file": filename,
                "label": 1,
                "frames": [image_array.tolist()],
                "audio": None,
                "tags": {
                    "low_light": False,
                    "occlusion": False,
                    "extreme_pose": False,
                    "distorted_audio": False,
                    "eye_blinking_issue": False,
                    "lip_sync_issue": False,
                    "facial_inconsistency": False
                }
            })
    return generated_data

# Load existing JSON data
with open(existing_json_path, 'r', encoding='utf-8') as f:
    existing_data = json.load(f)

# Load generated images data
generated_data = load_generated_images(generated_dir)

# Integrate generated data into existing data
existing_data.extend(generated_data)

# Save the updated data to a new JSON file
with open(updated_json_path, 'w', encoding='utf-8') as f:
    json.dump(existing_data, f, indent=4)

print(f"Integrated {len(generated_data)} generated deepfake entries into the existing JSON data.")
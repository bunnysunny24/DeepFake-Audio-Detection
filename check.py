import os

DATASET_PATH = r"D:/Bunny/Deepfake/backend/image_data/image-dataset-7"

for phase in ["train", "validation"]:
    for label in ["fake", "real"]:
        folder = os.path.join(DATASET_PATH, phase, label)
        if os.path.exists(folder):
            images = [f for f in os.listdir(folder) if f.endswith('.jpg')]
            print(f"{folder} -> {len(images)} images found")
        else:
            print(f"⚠️ Folder not found: {folder}")

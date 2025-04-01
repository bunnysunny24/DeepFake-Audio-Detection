import os

# Define dataset base directory
base_dir = r"D:\Bunny\Deepfake\backend\image_data"
folders = {
    "images": os.path.join(base_dir, "image-dataset-7"),
    "heatmaps": os.path.join(base_dir, "landmark_heatmaps_7"),
    "ear": os.path.join(base_dir, "ear_7"),
    "optical_flow": os.path.join(base_dir, "optical_flow_7"),
    "segmented": os.path.join(base_dir, "segmented_mediapipe"),
    "features": os.path.join(base_dir, "extracted_features-2"),
}

# List of segmented parts to check
segmented_parts = [
    "background", "cloth", "ear_ring", "eye_glass", "hair", "hat", "left_brow", "left_ear", "left_eye",
    "lower_lip", "mouth", "neck", "necklace", "nose", "right_brow", "right_ear", "right_eye", "skin", "upper_lip"
]

# Function to check dataset consistency for ONE image
def check_single_example():
    print(f"\n🔍 Searching for an example image...\n")

    for split in ["train", "validation"]:
        for label in ["real", "fake"]:
            img_folder = os.path.join(folders["images"], split, label)
            if os.path.exists(img_folder):
                image_files = [f for f in os.listdir(img_folder) if f.endswith((".png", ".jpg", ".jpeg"))]
                
                if image_files:
                    example_img = image_files[0]  # Select the first image
                    base_name, ext = os.path.splitext(example_img)

                    print(f"✅ Checking {example_img} in {split}/{label}...\n")
                    missing_files = []

                    # Construct expected file paths
                    paths_to_check = {
                        "heatmap": os.path.join(folders["heatmaps"], split, label, f"{base_name}_heatmap.jpg"),
                        "ear": os.path.join(folders["ear"], split, label, f"{base_name}_ear.jpg"),
                        "optical_flow": os.path.join(folders["optical_flow"], split, label, f"{base_name}_flow.jpg"),
                        "features": os.path.join(folders["features"], split, label, f"{base_name}_hair.png"),
                    }

                    # Add segmented Mediapipe checks
                    for part in segmented_parts:
                        paths_to_check[part] = os.path.join(folders["segmented"], split, label, f"{base_name}_{part}.png")

                    # Check if all expected files exist
                    for key, path in paths_to_check.items():
                        if not os.path.exists(path):
                            missing_files.append(f"{key}: {path}")

                    if missing_files:
                        print(f"⚠️ Missing files for {example_img}:")
                        for mf in missing_files:
                            print(f"   - {mf}")
                    else:
                        print(f"✅ {example_img} has all required files.")

                    return  # Stop after checking one example image

    print("❌ No example image found in the dataset!")

# Run the check for one image
check_single_example()

import os
from PIL import Image, ImageFilter
from tqdm import tqdm

# Define base paths
base_dir = r"D:\Bunny\Deepfake\backend\image_data"
image_dir = os.path.join(base_dir, "image-dataset-7")  # Your main dataset
ear_dir = os.path.join(base_dir, "ear_7")  # Folder for ear frames
optical_flow_dir = os.path.join(base_dir, "optical_flow_7")  # Folder for optical flow frames

# Create necessary directories for train/fake only
os.makedirs(os.path.join(ear_dir, "train", "fake"), exist_ok=True)
os.makedirs(os.path.join(optical_flow_dir, "train", "fake"), exist_ok=True)

# Function to generate frames (ONLY for train/fake)
def generate_train_fake_frames():
    input_folder = os.path.join(image_dir, "train", "fake")
    ear_output_folder = os.path.join(ear_dir, "train", "fake")
    optical_output_folder = os.path.join(optical_flow_dir, "train", "fake")

    for file_name in tqdm(os.listdir(input_folder), desc="Processing Train/Fake Images"):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Skip non-image files
            continue
        
        input_file = os.path.join(input_folder, file_name)
        base_name, ext = os.path.splitext(file_name)

        # Load image
        image = Image.open(input_file).convert("RGB")

        # Generate Ear Image (Grayscale)
        ear_image = image.convert("L")  # Convert to grayscale
        ear_image.save(os.path.join(ear_output_folder, f"{base_name}_ear{ext}"))

        # Generate Optical Flow Image (Blurred)
        optical_flow_image = image.filter(ImageFilter.BLUR)  # Simple blur as placeholder
        optical_flow_image.save(os.path.join(optical_output_folder, f"{base_name}_flow{ext}"))

# Process only train/fake images
generate_train_fake_frames()

print("✅ Train/Fake Ear and Optical Flow frames generated successfully!")

import os
from PIL import Image, ImageFilter
from tqdm import tqdm

# Define base paths
base_dir = r"D:\Bunny\Deepfake\backend\image_data"
image_dir = os.path.join(base_dir, "image-dataset-7")  # Your main dataset
ear_dir = os.path.join(base_dir, "ear_7")  # Folder for ear frames
optical_flow_dir = os.path.join(base_dir, "optical_flow_7")  # Folder for optical flow frames

# Create necessary directories for validation/real only
os.makedirs(os.path.join(ear_dir, "validation", "real"), exist_ok=True)
os.makedirs(os.path.join(optical_flow_dir, "validation", "real"), exist_ok=True)

# Function to generate frames (ONLY for validation/real)
def generate_validation_real_frames():
    input_folder = os.path.join(image_dir, "validation", "real")
    ear_output_folder = os.path.join(ear_dir, "validation", "real")
    optical_output_folder = os.path.join(optical_flow_dir, "validation", "real")

    for file_name in tqdm(os.listdir(input_folder), desc="Processing Validation/Real Images"):
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

# Process only validation/real images
generate_validation_real_frames()

print("✅ Validation/Real Ear and Optical Flow frames generated successfully!")

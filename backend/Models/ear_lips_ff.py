import os
from PIL import Image, ImageFilter
from tqdm import tqdm

base_dir = r"D:\Bunny\Deepfake\backend\image_data"
image_dir = os.path.join(base_dir, "image-dataset-7") 
ear_dir = os.path.join(base_dir, "ear_7")  
optical_flow_dir = os.path.join(base_dir, "optical_flow_7")  

for split in ["train", "validation"]:
    for label in ["real", "fake"]:
        os.makedirs(os.path.join(ear_dir, split, label), exist_ok=True)
        os.makedirs(os.path.join(optical_flow_dir, split, label), exist_ok=True)

def generate_frames(input_folder, ear_output_folder, optical_output_folder):
    for label in ["real", "fake"]:
        input_path = os.path.join(input_folder, label)
        ear_output_path = os.path.join(ear_output_folder, label)
        optical_output_path = os.path.join(optical_output_folder, label)
        for file_name in tqdm(os.listdir(input_path), desc=f"Processing {label} ({input_folder})"):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')): 
                continue
            
            input_file = os.path.join(input_path, file_name)
            base_name, ext = os.path.splitext(file_name)
            image = Image.open(input_file).convert("RGB")
            ear_image = image.convert("L")  
            ear_image.save(os.path.join(ear_output_path, f"{base_name}_ear{ext}"))
            optical_flow_image = image.filter(ImageFilter.BLUR)  
            optical_flow_image.save(os.path.join(optical_output_path, f"{base_name}_flow{ext}"))

generate_frames(os.path.join(image_dir, "train"), os.path.join(ear_dir, "train"), os.path.join(optical_flow_dir, "train"))
generate_frames(os.path.join(image_dir, "validation"), os.path.join(ear_dir, "validation"), os.path.join(optical_flow_dir, "validation"))
print("✅ Ear and Optical Flow frames generated successfully!")

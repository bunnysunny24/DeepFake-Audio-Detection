import os
import sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Add the cloned repository to the system path
sys.path.append(r'D:\Bunny\Deepfake\backend\models\stylegan2-pytorch')

from model import Generator

# Force the use of CPU
device = torch.device('cpu')

# Initialize the StyleGAN2 generator
gan_model = Generator(size=1024, style_dim=512, n_mlp=8).to(device)

# Attempt to load the PyTorch-pretrained weights
try:
    checkpoint_path = r'D:\Bunny\Deepfake\backend\combined_data\stylegan2-horse-config-f.pt'  # Change to a PyTorch-compatible model file
    gan_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Checkpoint file not found at {checkpoint_path}. Please provide a valid PyTorch model checkpoint (.pt file).")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

gan_model.eval()

# Directory setup for saving generated images
output_dir = 'generated_deepfakes'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to generate deepfake images
def generate_deepfake_images(num_images=100, image_size=256):
    z_dim = 512  # Latent space dimension
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    for i in range(num_images):
        z = torch.randn(1, z_dim).to(device)  # Generate random latent vector
        with torch.no_grad():
            # Generate an image using the model
            img_tensor = gan_model([z])[0].clamp(-1, 1).cpu()
            img_tensor = (img_tensor + 1) / 2  # Normalize to [0, 1]
            img_array = (img_tensor.numpy() * 255).astype(np.uint8)
            
            # Convert to PIL Image and save
            img = Image.fromarray(img_array.transpose(1, 2, 0), mode='RGB')
            img.save(os.path.join(output_dir, f'deepfake_{i}.png'))
        print(f'Generated {i+1}/{num_images} images.')

if __name__ == "__main__":
    num_images = 100  # Number of deepfake images to generate
    generate_deepfake_images(num_images)
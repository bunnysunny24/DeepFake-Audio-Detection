import os
import sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Add the cloned repository to the system path
sys.path.append('D:\\Bunny\\Deepfake\\backend\\models\\stylegan2-pytorch')

from model import Generator

# Force the use of CPU
device = torch.device('cpu')
gan_model = Generator(size=1024, style_dim=512, n_mlp=8).to(device)
gan_model.load_state_dict(torch.load('D:\\Bunny\\Deepfake\\backend\\combined_data\\stylegan2-horse-config-f.pkl', map_location=device))
gan_model.eval()

# Directory setup
output_dir = 'generated_deepfakes'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to generate deepfake images
def generate_deepfake_images(num_images=100, image_size=256):
    z_dim = 512
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    for i in range(num_images):
        z = torch.randn(1, z_dim).to(device)
        with torch.no_grad():
            img_tensor = gan_model([z])[0]
            img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
            img_array = (img_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array.transpose(1, 2, 0))
            img.save(os.path.join(output_dir, f'deepfake_{i}.png'))
        print(f'Generated {i+1}/{num_images} images.')

if __name__ == "__main__":
    num_images = 100  # Number of deepfake images to generate
    generate_deepfake_images(num_images)
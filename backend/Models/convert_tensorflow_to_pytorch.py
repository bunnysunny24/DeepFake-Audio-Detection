import sys
import os
import pickle
import torch
import numpy as np
from collections import OrderedDict
import tensorflow as tf

# Add the path to the directory containing dnnlib
sys.path.append(os.path.abspath(r"D:\Bunny\Deepfake\backend\Models\stylegan2"))

# Disable GPU by setting the CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Add the path to the stylegan2-pytorch directory
sys.path.append(os.path.abspath(r"D:\Bunny\Deepfake\backend\Models\stylegan2-pytorch"))

# Import the PyTorch Generator model from the rosinality implementation
from model import Generator

# Import dnnlib for TensorFlow session initialization
import dnnlib
import dnnlib.tflib as tflib

# Paths to the TensorFlow and PyTorch models
tensorflow_model_path = r"D:\Bunny\Deepfake\backend\combined_data\stylegan2-horse-config-f.pkl"
pytorch_model_path = r"D:\Bunny\Deepfake\backend\combined_data\stylegan2-horse-config-f.pt"

def validate_paths(tensorflow_path, pytorch_path):
    """Validate file paths for TensorFlow and PyTorch models."""
    if not os.path.isfile(tensorflow_path):
        raise FileNotFoundError(f"TensorFlow model file not found: {tensorflow_path}")
    if not os.path.isdir(os.path.dirname(pytorch_path)):
        raise FileNotFoundError(f"Directory for PyTorch model does not exist: {os.path.dirname(pytorch_path)}")

def initialize_tensorflow():
    """Initialize TensorFlow using dnnlib.tflib."""
    print("Initializing TensorFlow...")
    tflib.init_tf()

def load_tensorflow_model(tensorflow_path):
    """Load TensorFlow model from a .pkl file."""
    print("Loading TensorFlow model...")
    with open(tensorflow_path, 'rb') as f:
        data = pickle.load(f)
    if 'Gs' not in data:
        raise KeyError("The loaded TensorFlow model does not contain 'Gs' key.")
    return data['Gs']

def initialize_pytorch_model():
    """Initialize the PyTorch StyleGAN2 generator."""
    print("Initializing PyTorch model...")
    pytorch_generator = Generator(size=1024, style_dim=512, n_mlp=8)
    pytorch_generator.eval()
    return pytorch_generator

def convert_weights(tf_weights, pytorch_model):
    """Convert TensorFlow weights to PyTorch weights."""
    print("Converting TensorFlow weights to PyTorch format...")
    state_dict = OrderedDict()
    for tf_name, tf_weight in tf_weights.items():
        # Adjust the TensorFlow weight name to match the PyTorch model's state_dict
        pt_name = tf_name.replace('/', '.').replace('weight', 'weight').replace('bias', 'bias')
        weight = torch.from_numpy(tf_weight).float()

        # Check the shape and transpose if necessary (for convolutional layers)
        if len(weight.shape) == 4:  # TensorFlow uses [H, W, In, Out], PyTorch uses [Out, In, H, W]
            weight = weight.permute(3, 2, 0, 1)
        elif len(weight.shape) == 2:  # Fully connected layers
            weight = weight.permute(1, 0)

        state_dict[pt_name] = weight

    # Load the converted state_dict into the PyTorch model
    pytorch_model.load_state_dict(state_dict, strict=False)
    return pytorch_model

def extract_tensorflow_weights(generator):
    """Extract TensorFlow weights into a dictionary."""
    print("Extracting TensorFlow weights...")
    tf_weights = {}
    for var in generator.trainable_variables:
        tf_weights[var.name] = var.numpy()
    return tf_weights

def save_pytorch_model(pytorch_model, pytorch_path):
    """Save the PyTorch model to a file."""
    print(f"Saving PyTorch model to {pytorch_path}...")
    torch.save(pytorch_model.state_dict(), pytorch_path)
    print("Conversion complete!")

def main():
    try:
        # Validate paths
        validate_paths(tensorflow_model_path, pytorch_model_path)

        # Initialize TensorFlow
        initialize_tensorflow()

        # Load TensorFlow model
        Gs = load_tensorflow_model(tensorflow_model_path)

        # Initialize PyTorch model
        pytorch_generator = initialize_pytorch_model()

        # Extract TensorFlow weights
        tf_weights = extract_tensorflow_weights(Gs)

        # Convert and load weights into PyTorch model
        pytorch_generator = convert_weights(tf_weights, pytorch_generator)

        # Save the PyTorch model
        save_pytorch_model(pytorch_generator, pytorch_model_path)

    except Exception as e:
        print(f"Error: {e}")
        print("Conversion failed!")

if __name__ == "__main__":
    main()
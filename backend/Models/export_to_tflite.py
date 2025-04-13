import torch
import tensorflow as tf
from model import MultiModalDeepfakeDetector

# Load trained PyTorch model
model = MultiModalDeepfakeDetector()
model.load_state_dict(torch.load("path/to/trained_model.pth"))
model.eval()

# Convert to ONNX
dummy_input_video = torch.randn(1, 3, 224, 224)  # Example video input
dummy_input_audio = torch.randn(1, 16000)  # Example audio input
torch.onnx.export(model, (dummy_input_video, dummy_input_audio), "model.onnx", export_params=True)

# Convert ONNX to TFLite (using tf converter)
converter = tf.lite.TFLiteConverter.from_saved_model("model.onnx")
tflite_model = converter.convert()

# Save TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
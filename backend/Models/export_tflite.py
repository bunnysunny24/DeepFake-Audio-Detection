import torch
import tensorflow as tf
from model import MultiModalDeepfakeDetector

# Paths
model_path = r"D:\Bunny\Deepfake\backend\model\deepfake_detector.pth"
tflite_path = r"D:\Bunny\Deepfake\backend\model\deepfake_detector.tflite"

# Load Model
model = MultiModalDeepfakeDetector()
model.load_state_dict(torch.load(model_path))
model.eval()

# Convert to ONNX
dummy_input_video = torch.randn(1, 3, 224, 224)  # Example video input
dummy_input_audio = torch.randn(1, 16000)  # Example audio input
torch.onnx.export(model, (dummy_input_video, dummy_input_audio), "model.onnx", export_params=True)

# Convert ONNX to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("model.onnx")
tflite_model = converter.convert()

# Save TFLite model
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"Model exported to {tflite_path}")
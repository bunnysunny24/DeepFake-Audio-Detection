import torch
from multi_modal_model import MultiModalDeepfakeModel
from dataset_loader import preprocess_video, preprocess_audio  # Implement your own preprocessing functions
import os
import cv2
import numpy as np

def load_model(model_path, device):
    model = MultiModalDeepfakeModel(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

def predict_combined(video_path, audio_path, model, device):
    # Preprocess video frames
    video_frames = preprocess_video(video_path)  # Implement video frame extraction and preprocessing
    video_frames = torch.tensor(video_frames).to(device)

    # Preprocess audio
    audio = preprocess_audio(audio_path)  # Implement audio preprocessing (e.g., feature extraction from audio)
    audio = torch.tensor(audio).to(device)

    # Run prediction
    with torch.no_grad():
        outputs, deepfake_check = model(video_frames, audio)

    # Get the predicted label (fake or real)
    predicted_label = torch.argmax(outputs, dim=1).item()
    return predicted_label, deepfake_check.item()

if __name__ == "__main__":
    model_save_path = r"D:\Bunny\Deepfake\backend\Models\saved_models\best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = load_model(model_save_path, device)

    # Specify paths to the video and audio you want to predict
    video_path = "path_to_video.mp4"
    audio_path = "path_to_audio.wav"

    # Get prediction
    predicted_label, deepfake_check = predict_combined(video_path, audio_path, model, device)

    # Output the result
    if predicted_label == 0:
        print("Prediction: Fake Video and Audio")
    else:
        print("Prediction: Real Video and Audio")

    print(f"Deepfake Check: {deepfake_check}")

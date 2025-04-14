import torch
from multi_modal_model import MultiModalDeepfakeModel
from dataset_loader import video_transform, audio_transform
import cv2
import numpy as np
import librosa
import moviepy.editor as mp
import os

# Load the model
model = MultiModalDeepfakeModel(num_classes=2)
model.load_state_dict(torch.load("saved_model.pth", map_location=torch.device('cpu')))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def extract_audio_from_video(video_path, output_wav_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(output_wav_path, codec='pcm_s16le')

def preprocess_video_audio(video_path, max_frames=32, audio_length=16000):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    video_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = video_transform(frame)
        video_frames.append(frame)
    cap.release()

    if len(video_frames) == 0:
        raise Exception("No valid frames extracted from video.")

    video_tensor = torch.stack(video_frames).unsqueeze(0).to(device)  # Shape: (1, N, C, H, W)

    # Extract and load audio
    audio_path = video_path.replace('.mp4', '.wav')
    extract_audio_from_video(video_path, audio_path)

    audio, _ = librosa.load(audio_path, sr=16000)
    if len(audio) > audio_length:
        audio = audio[:audio_length]
    else:
        audio = np.pad(audio, (0, audio_length - len(audio)), mode='constant')

    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, audio_length)

    return video_tensor, audio_tensor

def predict_fake_or_real(video_path):
    video_tensor, audio_tensor = preprocess_video_audio(video_path)
    with torch.no_grad():
        output, _ = model(video_tensor, audio_tensor)
        prediction = torch.argmax(output, dim=1).item()
        label = "Fake" if prediction == 1 else "Real"
        return label

# ==== Example ====
video_path = "example_video.mp4"  # <- Your test video here
result = predict_fake_or_real(video_path)
print(f"Prediction for '{video_path}': {result}")

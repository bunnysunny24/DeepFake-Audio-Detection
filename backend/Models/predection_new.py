import cv2
import torch
import numpy as np
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch.nn.functional as F
import torchaudio  # For audio processing
import warnings
from multi_modal_model import MultiModalDeepfakeModel
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Function to extract frames from the video
def extract_frames(video_path, frame_interval=30):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {video_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

# Function to extract audio from the video
def extract_audio(video_path, output_audio_path='audio.wav'):
    audio = AudioSegment.from_file(video_path)
    audio.export(output_audio_path, format='wav')
    return output_audio_path

# Load the Wav2Vec 2.0 model for audio feature extraction
def load_audio_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
    return processor, model

# Preprocess video frames (normalize, resize, and adjust channels as per your model's requirement)
def preprocess_video(frames):
    # Resize the frames to 224x224 (or the model's input size)
    frames_resized = [cv2.resize(frame, (224, 224)) for frame in frames]
    
    # Normalize to [0, 1]
    frames_resized = np.array(frames_resized, dtype=np.float32) / 255.0
    
    # Convert to PyTorch tensor and rearrange dimensions to (batch_size, channels, height, width)
    frames_tensor = torch.tensor(frames_resized).permute(0, 3, 1, 2).float()
    return frames_tensor.unsqueeze(0)  # Add batch dimension

# Preprocess audio (convert audio file to suitable format for the Wav2Vec2 model)
def preprocess_audio(audio_path, processor, target_sampling_rate=16000):
    """
    Preprocess audio by loading it, resampling to the target sampling rate, and preparing it for the Wav2Vec2 model.
    """
    # Load audio using torchaudio
    audio, sample_rate = torchaudio.load(audio_path)

    # Resample audio if sample rate is not equal to the target sampling rate
    if sample_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sampling_rate)
        audio = resampler(audio)

    # Convert audio to mono by taking the mean across channels (if stereo)
    if audio.size(0) > 1:  # Check if audio has multiple channels
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Remove channel dimension (squeeze) to match the required input shape
    audio = audio.squeeze(0)

    # Use the processor to prepare the audio for the Wav2Vec2 model
    inputs = processor(audio, sampling_rate=target_sampling_rate, return_tensors="pt", padding=True)
    return inputs.input_values

# Predict whether the video is fake or real
def predict_video(video_path, model, device):
    # Extract frames and audio
    print("Extracting video frames...")
    frames = extract_frames(video_path)
    print("Extracting audio...")
    audio_path = extract_audio(video_path)

    # Preprocess the video frames
    print("Preprocessing video frames...")
    video_frames = preprocess_video(frames).to(device)

    # Preprocess the audio
    print("Preprocessing audio...")
    audio_processor, audio_model = load_audio_model()
    audio_input = preprocess_audio(audio_path, audio_processor).to(device)

    # Predict using the multimodal model
    print("Making predictions...")
    model.eval()
    with torch.no_grad():
        outputs, _ = model(video_frames, audio_input)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()
        confidence = probabilities.max(dim=1).values.item()

    # Return prediction and confidence
    return predicted_class, confidence

# Main function
if __name__ == "__main__":
    # Path to the video file
    video_path = r"C:\Users\Bhavashesh\Downloads\WhatsApp Video 2025-04-14 at 22.40.10.mp4"  # Replace with your video file path

    # Initialize the multimodal deepfake detection model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalDeepfakeModel(num_classes=2)
    model.load_state_dict(torch.load(r"D:\Bunny\Deepfake\backend\Models\best_model_2.pth", map_location=device))
    model.to(device)

    # Predict the video
    try:
        predicted_class, confidence = predict_video(video_path, model, device)
        print(f"Prediction: {'FAKE' if predicted_class == 1 else 'REAL'} with confidence {confidence:.2f}")
    except Exception as e:
        print(f"An error occurred: {e}")
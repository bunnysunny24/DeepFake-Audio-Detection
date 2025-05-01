import torch
import torchaudio
import cv2
import os
import numpy as np
from PIL import Image  # Import PIL for image conversion
from multi_modal_model import MultiModalDeepfakeModel
from transformers import Wav2Vec2Processor
from torchvision import transforms
from moviepy.editor import VideoFileClip

# Paths
model_path = r"D:\Bunny\Deepfake\backend\Models\saved_models\best_model.pth"
video_path = r"D:\Bunny\Deepfake\backend\combined_data\fake_2.mp4"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Video frame transform (resize and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize frames to 224x224 for the model input
    transforms.ToTensor()           # Convert to tensor
])

# Load model
model = MultiModalDeepfakeModel(num_classes=2)  # Assuming the model has 2 output classes (fake/real)
model.load_state_dict(torch.load(model_path, map_location=device))  # Load saved model weights
model.to(device)
model.eval()  # Set model to evaluation mode

# Audio processor for Wav2Vec2
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def extract_video_frames(video_path, num_frames=16):
    """
    Extract frames from a video file.
    :param video_path: Path to the video.
    :param num_frames: Number of frames to extract.
    :return: Tensor of video frames with consistent batch size.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)  # Select frames evenly

    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:  # Only select the specified frames
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame_pil = Image.fromarray(frame_rgb)  # Convert NumPy array to PIL Image
            frame_tensor = transform(frame_pil)  # Apply transformations (resize, to tensor)
            frames.append(frame_tensor)
    cap.release()

    video_tensor = torch.stack(frames)  # Stack frames into a single tensor (num_frames, 3, H, W)
    video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension (1, num_frames, 3, H, W)

    print(f"Video frames shape after processing: {video_tensor.shape}")
    return video_tensor

def extract_audio_waveform(video_path):
    """
    Extract the audio waveform from the video.
    :param video_path: Path to the video.
    :return: Audio waveform tensor with the correct shape.
    """
    audio_clip = VideoFileClip(video_path).audio
    temp_audio_path = "temp_audio.wav"
    audio_clip.write_audiofile(temp_audio_path, verbose=False, logger=None)  # Save audio as temporary WAV file

    waveform, sample_rate = torchaudio.load(temp_audio_path)  # Load the audio waveform
    os.remove(temp_audio_path)  # Delete the temporary file

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Ensure the tensor has the correct shape
    if waveform.dim() > 2 or waveform.size(0) > 1:  # If multi-channel audio, average across channels
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Remove unnecessary dimensions to match [batch_size, sequence_length]
    waveform = waveform.squeeze(0)  # Remove channel dimension if present

    # Add batch dimension to match the expected input shape [batch_size, sequence_length]
    waveform = waveform.unsqueeze(0)

    print(f"Audio waveform shape after processing: {waveform.shape}")
    return waveform

def predict(video_path):
    """
    Predict whether the video is real or fake.
    :param video_path: Path to the video.
    """
    # Preprocess video and audio
    video_frames = extract_video_frames(video_path).to(device)  # Shape: (1, num_frames, 3, 224, 224)
    audio_waveform = extract_audio_waveform(video_path).to(device)  # Shape: (1, sequence_length)

    # Debugging: Ensure batch sizes match
    print(f"Video frames batch size: {video_frames.size(0)}")
    print(f"Audio waveform batch size: {audio_waveform.size(0)}")

    # Perform prediction
    with torch.no_grad():
        outputs, _ = model(video_frames, audio_waveform)
        probs = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        pred = torch.argmax(probs).item()  # Get predicted class index
        confidence = probs[0][pred].item()  # Get confidence for the prediction

    # Output prediction result
    label = "FAKE" if pred == 1 else "REAL"
    print(f"🎬 Video is predicted to be: {label} (Confidence: {confidence:.2f})")

# Run prediction
if __name__ == "__main__":
    predict(video_path)
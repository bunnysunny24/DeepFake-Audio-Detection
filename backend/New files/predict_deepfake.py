import torch
import os
import cv2
import numpy as np
from multi_modal_model import MultiModalDeepfakeModel

def extract_video_frames(video_path, num_frames=32, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(torch.tensor(frame).permute(2, 0, 1).float() / 255.)  # [C, H, W]
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No frames extracted from video.")
    return torch.stack(frames)  # [num_frames, C, H, W]

def extract_audio_tensor(video_path, audio_length=8000):
    """
    Extract audio from video file using ffmpeg and librosa.
    """
    import subprocess
    import tempfile
    import librosa

    # Create a temporary wav file
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav:
        # Use ffmpeg to extract audio
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-vn", temp_wav.name
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            audio, sample_rate = librosa.load(temp_wav.name, sr=16000, mono=True)
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio with ffmpeg: {e}")

    # Center crop or pad
    if len(audio) > audio_length:
        start = (len(audio) - audio_length) // 2
        audio = audio[start:start + audio_length]
    else:
        audio = np.pad(audio, (0, audio_length - len(audio)), mode='constant')
    return torch.tensor(audio, dtype=torch.float32)  # [audio_length]

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        config = checkpoint.get('config', {})
        model = MultiModalDeepfakeModel(
            num_classes=config.get('num_classes', 2),
            video_feature_dim=config.get('video_feature_dim', 1024),
            audio_feature_dim=config.get('audio_feature_dim', 8000),
            transformer_dim=config.get('transformer_dim', 768),
            num_transformer_layers=config.get('num_transformer_layers', 4),
            enable_face_mesh=config.get('enable_face_mesh', True),
            enable_explainability=config.get('enable_explainability', True),
            fusion_type=config.get('fusion_type', 'attention'),
            backbone_visual=config.get('backbone_visual', 'efficientnet'),
            backbone_audio=config.get('backbone_audio', 'wav2vec2'),
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model = MultiModalDeepfakeModel()
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model

def main(model_path, video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    video_frames = extract_video_frames(video_path)  # [num_frames, C, H, W]
    audio_tensor = extract_audio_tensor(video_path)  # [audio_length]
    inputs = {
        "video_frames": video_frames.unsqueeze(0).to(device),  # [1, num_frames, C, H, W]
        "audio": audio_tensor.unsqueeze(0).to(device),         # [1, audio_length]
    }
    with torch.no_grad():
        output, results = model(inputs)
        pred = torch.softmax(output, dim=-1) if output.shape[-1] > 1 else torch.sigmoid(output)
        print("Raw Output:", output)
        print("Prediction:", pred)
        print("Explainability/Results:", results)

        # Determine label and confidence
        if pred.shape[-1] > 1:
            fake_prob = float(pred[0, 1])
            if fake_prob >= 0.5:
                label = "Fake"
                confidence_value = fake_prob
            else:
                label = "Real"
                confidence_value = 1 - fake_prob
        else:
            raw_confidence = float(pred.item())
            if raw_confidence >= 0.5:
                label = "Fake"
                confidence_value = raw_confidence
            else:
                label = "Real"
                confidence_value = 1 - raw_confidence
        print(f"Prediction: {label} (Confidence: {confidence_value:.2f})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python predict_deepfake.py <model.pth> <video.mp4>")
        exit(1)
    main(sys.argv[1], sys.argv[2])

import os
import json
import torchaudio
import cv2
import torch
import numpy as np
from tqdm import tqdm

def validate_dataset(json_path, data_dir, max_frames=32, audio_length=16000, transform=None, audio_transform=None):
    """
    Validates the dataset by checking all components (video, audio, metadata, etc.).
    
    Args:
    - json_path (str): Path to the JSON metadata file.
    - data_dir (str): Directory containing video and audio files.
    - max_frames (int): Maximum number of video frames to load.
    - audio_length (int): Length of the audio tensor to pad or trim.
    - transform: Video frame transformation function.
    - audio_transform: Audio transformation function.
    """
    # Check if the JSON file exists
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at: {json_path}")
    
    # Load the metadata
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"🔍 Validating dataset with {len(data)} samples in '{data_dir}'...")
    issues = []

    for idx, sample in enumerate(tqdm(data, desc="Validating samples")):
        try:
            # Construct file paths
            video_path = os.path.join(data_dir, sample['file'])
            audio_path = video_path.replace('.mp4', '.wav')
            original_video_path = (
                os.path.join(data_dir, sample['original'])
                if 'original' in sample and sample['original']
                else None
            )
            original_audio_path = (
                original_video_path.replace('.mp4', '.wav')
                if original_video_path
                else None
            )

            # Check video file
            if not os.path.exists(video_path):
                issues.append((idx, "Missing video file", video_path))
            else:
                video_frames = load_video(video_path, max_frames, transform)
                if video_frames is None:
                    issues.append((idx, "Failed to load video", video_path))

            # Check audio file
            if not os.path.exists(audio_path):
                issues.append((idx, "Missing audio file", audio_path))
            else:
                audio_tensor = load_audio(audio_path, audio_length, audio_transform)
                if audio_tensor is None:
                    issues.append((idx, "Failed to load audio", audio_path))

            # Validate original video and audio based on flags
            if sample.get('modify_video', False) and sample.get('modify_audio', False):
                # Validate both original video and audio
                if original_video_path:
                    if not os.path.exists(original_video_path):
                        issues.append((idx, "Missing original video file", original_video_path))
                if original_audio_path:
                    if not os.path.exists(original_audio_path):
                        issues.append((idx, "Missing original audio file", original_audio_path))
            elif sample.get('modify_video', False):
                # Validate only original video
                if original_video_path:
                    if not os.path.exists(original_video_path):
                        issues.append((idx, "Missing original video file", original_video_path))
            elif sample.get('modify_audio', False):
                # Validate only original audio
                if original_audio_path:
                    if not os.path.exists(original_audio_path):
                        issues.append((idx, "Missing original audio file", original_audio_path))

            # Check metadata fields
            if 'n_fakes' not in sample:
                issues.append((idx, "Missing 'n_fakes' field", None))
            if 'timestamps' not in sample:
                issues.append((idx, "Missing 'timestamps' field", None))
            if 'fake_periods' not in sample:
                issues.append((idx, "Missing 'fake_periods' field", None))

        except Exception as e:
            issues.append((idx, "Exception during validation", str(e)))

    # Print summary
    if issues:
        print("\n⚠️ Issues Found:")
        for issue in issues:
            print(f" - Sample {issue[0]}: {issue[1]} ({issue[2]})")
        print("\nPlease review the issues and fix the dataset.")
    else:
        print("\n✅ All samples passed validation.")

def load_video(path, max_frames, transform=None):
    """
    Loads video frames from a given path.
    """
    if not os.path.exists(path):
        return None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    video_frames = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if transform:
            frame = transform(frame)
        else:
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
        video_frames.append(frame)
    cap.release()

    if not video_frames:
        return None
    return torch.stack(video_frames)

def load_audio(path, audio_length, audio_transform=None):
    """
    Loads audio waveform from a given path.
    """
    if not os.path.exists(path):
        return None
    try:
        audio, sample_rate = torchaudio.load(path)
        audio = audio.squeeze(0).numpy()

        if len(audio) > audio_length:
            audio = audio[:audio_length]
        else:
            audio = np.pad(audio, (0, audio_length - len(audio)), mode='constant')

        if audio_transform:
            audio = audio_transform(samples=audio, sample_rate=sample_rate)

        return torch.tensor(audio, dtype=torch.float32)
    except Exception:
        return None

if __name__ == "__main__":
    # Paths for the dataset
    json_path = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF\metadata.json"
    data_dir = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF"

    # Run validation
    validate_dataset(json_path, data_dir)
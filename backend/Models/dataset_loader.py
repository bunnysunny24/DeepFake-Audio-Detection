import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import torchaudio
import cv2
import warnings
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Shift

class MultiModalDeepfakeDataset(Dataset):
    def __init__(self, json_path, data_dir, max_frames=32, audio_length=16000, transform=None, audio_transform=None, logging=False):
        """
        Args:
            json_path (str): Path to the processed JSON file.
            data_dir (str): Base directory containing video and audio data.
            max_frames (int): Maximum number of frames to use per video.
            audio_length (int): Fixed length for audio samples (e.g., 16000 samples).
            transform (callable, optional): Optional transform to be applied on video frames.
            audio_transform (callable, optional): Optional transform to be applied on audio features.
            logging (bool): Flag to enable logging for debugging.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.data_dir = data_dir
        self.max_frames = max_frames
        self.audio_length = audio_length
        self.transform = transform
        self.audio_transform = audio_transform
        self.logging = logging

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            sample = self.data[idx]
            video_path = os.path.join(self.data_dir, sample['file'])
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Load video frames
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Failed to open video file: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"No frames found in video: {video_path}")

            frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
            video_frames = []

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    warnings.warn(f"Failed to read frame {frame_idx} from {video_path}. Skipping frame.")
                    continue
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                else:
                    frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
                video_frames.append(frame)
            cap.release()

            if len(video_frames) == 0:
                raise ValueError(f"No valid frames extracted from {video_path}")
            video_frames = torch.stack(video_frames)

            # Load raw audio
            audio_path = os.path.join(self.data_dir, sample['file'].replace('.mp4', '.wav'))
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            audio, sample_rate = torchaudio.load(audio_path)
            audio = audio.squeeze(0).numpy()  # Convert to numpy for augmentation

            # Ensure audio is of fixed length (e.g., pad or truncate to 16000 samples)
            if len(audio) > self.audio_length:
                audio = audio[:self.audio_length]
            else:
                audio = np.pad(audio, (0, self.audio_length - len(audio)), mode='constant')

            audio_tensor = torch.tensor(audio, dtype=torch.float32)

            # Apply audio augmentation if available
            if self.audio_transform:
                audio_tensor = self.audio_transform(samples=audio_tensor.numpy(), sample_rate=sample_rate)
                audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)

            # Dynamically generate the label
            label = 1 if sample['n_fakes'] > 0 else 0  # Fake if n_fakes > 0, else Real
            label = torch.tensor(label, dtype=torch.long)

            # Logging for debugging
            if self.logging:
                print(f"Loaded sample {idx}: Video shape = {video_frames.shape}, Audio shape = {audio_tensor.shape}, Label = {label.item()}")

            return {
                'video_frames': video_frames,  # (N, C, H, W)
                'audio': audio_tensor,         # (audio_length)
                'label': label                 # 0 or 1
            }

        except Exception as e:
            raise RuntimeError(f"Error loading sample at index {idx}: {e}")

# Define video augmentation transformations
video_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomGrayscale(p=0.1),  # Added grayscale for diversity
])

# Define audio augmentation transformations using audiomentations
audio_transform = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),  # Updated arguments
    # Optionally add Reverb, EQ, or Bandpass filters here
])

# DataLoader function 
def get_data_loaders(json_path, data_dir, batch_size=8, validation_split=0.2, shuffle=True, transform=None, audio_transform=None, num_workers=4):
    dataset = MultiModalDeepfakeDataset(
        json_path=json_path,
        data_dir=data_dir,
        transform=transform,
        audio_transform=audio_transform
    )

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_indices = indices[split:]
    val_indices = indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
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
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.data_dir = data_dir
        self.max_frames = max_frames
        self.audio_length = audio_length
        self.transform = transform
        self.audio_transform = audio_transform
        self.logging = logging  # Enable or disable detailed logging
        
        # Pre-validate the dataset to filter out problematic entries
        self.valid_indices = self._validate_dataset()
        print(f"Dataset initialized with {len(self.valid_indices)} valid samples out of {len(self.data)} total.")

    def _validate_dataset(self):
        """Pre-validate all samples in the dataset to identify valid ones."""
        valid_indices = []
        for idx in range(len(self.data)):
            sample = self.data[idx]
            video_path = os.path.join(self.data_dir, sample['file'])
            audio_path = video_path.replace('.mp4', '.wav')
            
            # Check if files exist
            if not os.path.exists(video_path):
                if self.logging:
                    print(f"⚠️ Video file missing for index {idx}: {video_path}")
                continue
                
            if not os.path.exists(audio_path):
                if self.logging:
                    print(f"⚠️ Audio file missing for index {idx}: {audio_path}")
                continue
                
            # For deepfake samples, check if original files exist when needed
            if sample['n_fakes'] > 0:
                if sample.get('modify_video', False) and 'original' in sample and sample['original']:
                    original_video_path = os.path.join(self.data_dir, sample['original'])
                    if not os.path.exists(original_video_path):
                        if self.logging:
                            print(f"⚠️ Original video file missing for index {idx}: {original_video_path}")
                        continue
                
                if sample.get('modify_audio', False) and 'original' in sample and sample['original']:
                    original_audio_path = os.path.join(self.data_dir, sample['original']).replace('.mp4', '.wav')
                    if not os.path.exists(original_audio_path):
                        if self.logging:
                            print(f"⚠️ Original audio file missing for index {idx}: {original_audio_path}")
                        continue
            
            # If we've reached here, the sample is valid
            valid_indices.append(idx)
            
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Get a sample by its index in the valid indices list."""
        # Map the provided index to the actual valid index in the dataset
        actual_idx = self.valid_indices[idx]
        sample = self.data[actual_idx]
        
        try:
            video_path = os.path.join(self.data_dir, sample['file'])
            audio_path = video_path.replace('.mp4', '.wav')

            original_video_path = (
                os.path.join(self.data_dir, sample['original'])
                if 'original' in sample and sample['original']
                else None
            )
            original_audio_path = (
                original_video_path.replace('.mp4', '.wav')
                if original_video_path
                else None
            )

            # Load video/audio with proper error handling
            video_frames = self._load_video(video_path)
            if video_frames is None:
                raise ValueError(f"Video loading failed for sample {actual_idx}. Path: {video_path}")
                
            audio_tensor = self._load_audio(audio_path)
            if audio_tensor is None:
                raise ValueError(f"Audio loading failed for sample {actual_idx}. Path: {audio_path}")

            # Load original video/audio if needed
            original_video_frames = None
            original_audio_tensor = None
            if sample['n_fakes'] > 0:
                if sample.get('modify_video', False) and original_video_path:
                    original_video_frames = self._load_video(original_video_path)
                    if original_video_frames is None and self.logging:
                        print(f"⚠️ Warning: Original video loading failed for sample {actual_idx}. Path: {original_video_path}")
                        
                if sample.get('modify_audio', False) and original_audio_path:
                    original_audio_tensor = self._load_audio(original_audio_path)
                    if original_audio_tensor is None and self.logging:
                        print(f"⚠️ Warning: Original audio loading failed for sample {actual_idx}. Path: {original_audio_path}")

            # Create fake mask
            timestamps = sample.get('timestamps', [])
            fake_mask = torch.zeros(len(timestamps))
            for i, (_, start, end) in enumerate(timestamps):
                for f_start, f_end in sample.get('fake_periods', []):
                    if start < f_end and end > f_start:
                        fake_mask[i] = 1
                        break

            label = torch.tensor(1 if sample['n_fakes'] > 0 else 0, dtype=torch.long)

            # Debugging: Log successful data loading
            if self.logging:
                print(f"✅ Successfully loaded sample at index {actual_idx}")

            return {
                'video_frames': video_frames,
                'audio': audio_tensor,
                'label': label,
                'original_video_frames': original_video_frames,
                'original_audio': original_audio_tensor,
                'fake_periods': sample.get('fake_periods', []),
                'timestamps': timestamps,
                'transcript': sample.get('transcript', ''),
                'fake_mask': fake_mask
            }

        except Exception as e:
            if self.logging:
                print(f"❌ Error in __getitem__ for index {actual_idx}: {e}")
            # Re-raise the exception to be caught by the DataLoader
            raise

    def _load_video(self, path):
        if not path or not os.path.exists(path):
            if self.logging:
                warnings.warn(f"⚠️ Video file not found: {path}")
            return None

        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                if self.logging:
                    warnings.warn(f"⚠️ Failed to open video: {path}")
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                if self.logging:
                    warnings.warn(f"⚠️ Video has no frames: {path}")
                return None
                
            # Create sampling indices for frames
            frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
            video_frames = []

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    if self.logging:
                        warnings.warn(f"⚠️ Failed to read frame {frame_idx} from {path}")
                    continue
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                else:
                    frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
                video_frames.append(frame)
            cap.release()

            if not video_frames:
                if self.logging:
                    warnings.warn(f"⚠️ No valid frames extracted from video: {path}")
                return None

            # Stack frames into a tensor
            return torch.stack(video_frames)

        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error loading video file: {path}. Error: {e}")
            return None

    def _load_audio(self, path):
        if not path or not os.path.exists(path):
            if self.logging:
                warnings.warn(f"⚠️ Audio file not found: {path}")
            return None
        try:
            audio, sample_rate = torchaudio.load(path)
            audio = audio.squeeze(0).numpy()

            if len(audio) > self.audio_length:
                audio = audio[:self.audio_length]
            else:
                audio = np.pad(audio, (0, self.audio_length - len(audio)), mode='constant')

            if self.audio_transform:
                try:
                    audio = self.audio_transform(samples=audio, sample_rate=sample_rate)
                except Exception as audio_transform_error:
                    if self.logging:
                        warnings.warn(f"⚠️ Audio transform error for file {path}. Error: {audio_transform_error}")

            return torch.tensor(audio, dtype=torch.float32)
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error loading audio file: {path}. Error: {e}")
            return None


def get_data_loaders(
    json_path, data_dir, batch_size=8, validation_split=0.2, shuffle=True, transform=None, audio_transform=None, num_workers=4, max_samples=None
):
    """
    Load data loaders with an option to restrict the maximum number of samples.
    
    Parameters:
        json_path (str): Path to the dataset metadata JSON file.
        data_dir (str): Directory containing video and audio files.
        batch_size (int): Batch size for the data loaders.
        validation_split (float): Fraction of the dataset to use for validation.
        shuffle (bool): Whether to shuffle the dataset.
        transform (callable): Transformations for video frames.
        audio_transform (callable): Transformations for audio samples.
        num_workers (int): Number of worker threads for loading data.
        max_samples (int, optional): Maximum number of samples to load from the dataset.
    
    Returns:
        tuple: Training and validation data loaders.
    """
    dataset = MultiModalDeepfakeDataset(
        json_path=json_path,
        data_dir=data_dir,
        transform=transform,
        audio_transform=audio_transform,
        logging=True  # Enable logging for debugging
    )
    
    # Get total number of valid samples
    num_samples = len(dataset)
    if num_samples == 0:
        raise ValueError("No valid samples found in the dataset!")
    
    # Restrict dataset size if max_samples is specified
    if max_samples is not None and max_samples < num_samples:
        indices = list(range(num_samples))
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(indices)
        indices = indices[:max_samples]
        num_samples = max_samples
    else:
        indices = list(range(num_samples))
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(indices)
    
    # Split indices into training and validation sets
    split = int(np.floor(validation_split * num_samples))
    train_indices = indices[split:]
    val_indices = indices[:split]

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create data loaders
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False,  # Keep all batches, even if they're smaller than batch_size
        collate_fn=collate_fn  # Custom collate function to handle None values
    )
    
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn  # Custom collate function to handle None values
    )

    print(f"✅ Dataset loaded with {len(train_indices)} training samples and {len(val_indices)} validation samples.")
    return train_loader, val_loader


def collate_fn(batch):
    """
    Custom collate function to handle None values and variable-sized tensors in the batch.
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if not batch:
        raise ValueError("All items in batch are None!")
    
    # Create a dictionary to hold the batched data
    result = {}
    
    # Get all keys from the first sample
    keys = batch[0].keys()
    
    # Batch each key
    for key in keys:
        if key == 'label':
            # Stack labels as a tensor
            result[key] = torch.stack([item[key] for item in batch])
        elif key == 'video_frames' or key == 'audio':
            # Stack tensors
            result[key] = torch.stack([item[key] for item in batch])
        elif key == 'original_video_frames' or key == 'original_audio':
            # Handle potentially missing original data
            values = [item[key] for item in batch if item[key] is not None]
            if values:
                result[key] = torch.stack(values)
            else:
                result[key] = None
        elif key == 'fake_periods' or key == 'timestamps':
            # List of lists, don't stack
            result[key] = [item[key] for item in batch]
        elif key == 'transcript':
            # List of strings
            result[key] = [item[key] for item in batch]
        elif key == 'fake_mask':
            # Don't stack fake_masks of different sizes, keep as list
            result[key] = [item[key] for item in batch]
        else:
            # Handle other types if needed
            result[key] = [item[key] for item in batch]
    
    return result

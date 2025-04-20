import json
import os
import torch
from torch.utils.data import Dataset
import cv2
import torchaudio
import logging

class MultiModalDeepfakeDataset(Dataset):
    def __init__(self, json_path, data_dir, max_frames=32, audio_length=16000, transform=None, audio_transform=None, logging=False):
        # Initialize logging if required
        self.logger = None
        if logging:
            self.logger = logging.getLogger('DeepfakeDatasetLogger')
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.data_dir = data_dir
        self.max_frames = max_frames
        self.audio_length = audio_length
        self.transform = transform
        self.audio_transform = audio_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        video_path = os.path.join(self.data_dir, sample['file'])
        audio_path = video_path.replace('.mp4', '.wav')

        video_frames = self._load_video(video_path)
        audio_tensor = self._load_audio(audio_path)

        # Check for None values in video and audio
        if video_frames is None or audio_tensor is None:
            if self.logger:
                self.logger.error(f"Sample {idx} caused None value: Video = {video_frames}, Audio = {audio_tensor}")
            return None  # Skip this sample if either video or audio is None

        label = 1 if sample['n_fakes'] > 0 else 0
        label = torch.tensor(label, dtype=torch.long)

        return {
            'video_frames': video_frames,
            'audio': audio_tensor,
            'label': label,
        }

    def _load_video(self, path):
        if not path or not os.path.exists(path):
            if self.logger:
                self.logger.error(f"Video file not found: {path}")
            return None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            if self.logger:
                self.logger.error(f"Failed to open video file: {path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            if self.logger:
                self.logger.error(f"No frames found in video: {path}")
            return None

        video_frames = []
        for frame_idx in range(min(total_frames, self.max_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                if self.logger:
                    self.logger.error(f"Failed to read frame {frame_idx} from {path}. Skipping frame.")
                continue
            video_frames.append(frame)
        cap.release()

        if len(video_frames) == 0:
            if self.logger:
                self.logger.error(f"No valid frames extracted from video: {path}")
            return None
        return torch.tensor(video_frames)

    def _load_audio(self, path):
        if not path or not os.path.exists(path):
            if self.logger:
                self.logger.error(f"Audio file not found: {path}")
            return None

        try:
            audio, sample_rate = torchaudio.load(path)
            audio = audio.squeeze(0).numpy()

            if len(audio) > self.audio_length:
                audio = audio[:self.audio_length]
            else:
                audio = np.pad(audio, (0, self.audio_length - len(audio)), mode='constant')

            return torch.tensor(audio, dtype=torch.float32)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading audio file {path}: {e}")
            return None


# Function to check and log where None values come from
def check_none_values(json_path, data_dir):
    dataset = MultiModalDeepfakeDataset(json_path=json_path, data_dir=data_dir, logging=True)
    print(f"Total samples in dataset: {len(dataset)}")

    # Check each sample
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is None:
            print(f"Invalid sample at index {idx}.")

# Example usage
json_path = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF\metadata.json"
data_dir = r"D:\Bunny\Deepfake\backend\combined_data\LAV-DF"

check_none_values(json_path, data_dir)

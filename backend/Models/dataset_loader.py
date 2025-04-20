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
import albumentations as A
from facenet_pytorch import MTCNN
from scipy.signal import spectrogram
import librosa
from PIL import Image
import random
import math


class MultiModalDeepfakeDataset(Dataset):
    def __init__(self, json_path, data_dir, max_frames=32, audio_length=16000, transform=None, audio_transform=None, 
                 logging=False, phase='train', detect_faces=True, compute_spectrograms=True, temporal_features=True):
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
        self.phase = phase  # 'train', 'val', or 'test'
        self.detect_faces = detect_faces
        self.compute_spectrograms = compute_spectrograms
        self.temporal_features = temporal_features
        
        # Optional face detector for more focused analysis
        if self.detect_faces:
            try:
                self.face_detector = MTCNN(
                    image_size=224, margin=40, min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
            except Exception as e:
                print(f"Warning: Could not initialize face detector: {e}")
                self.detect_faces = False
                
        # Pre-validate the dataset to filter out problematic entries
        self.valid_indices = self._validate_dataset()
        self.class_counts = self._count_classes()
        
        # Calculate class weights for imbalanced datasets
        self.class_weights = self._calculate_class_weights()
        
        print(f"Dataset initialized with {len(self.valid_indices)} valid samples out of {len(self.data)} total.")
        print(f"Class distribution: {self.class_counts}")

    def _validate_dataset(self):
        """Pre-validate all samples in the dataset to identify valid ones."""
        print("Starting dataset validation...")
        print("⚠️ LIMITING VALIDATION TO FIRST 100 SAMPLES ONLY")
        
        # Limit to first 100 samples for faster processing
        max_samples = 100
        max_to_validate = min(max_samples, len(self.data))
        print(f"Using first {max_to_validate} samples out of {len(self.data)} total.")
        
        valid_indices = []
        
        # Add progress indicators for validation
        progress_interval = max(1, max_to_validate // 10)  # Show progress ~10 times
        
        for idx in range(max_to_validate):
            if idx % progress_interval == 0 or idx == max_to_validate - 1:
                print(f"Validating sample {idx+1}/{max_to_validate} ({(idx+1)/max_to_validate*100:.1f}%)...")
            
            sample = self.data[idx]
            video_path = os.path.join(self.data_dir, sample['file'])
            audio_path = video_path.replace('.mp4', '.wav')
            
            # Simple validation - just check if files exist
            if os.path.exists(video_path) and os.path.exists(audio_path):
                valid_indices.append(idx)
        
        print(f"Found {len(valid_indices)} valid samples out of {max_to_validate} checked.")
        return valid_indices

    def _count_classes(self):
        """Count the number of real/fake samples for balancing."""
        real_count = 0
        fake_count = 0
        
        for idx in self.valid_indices:
            sample = self.data[idx]
            if sample.get('n_fakes', 0) > 0:
                fake_count += 1
            else:
                real_count += 1
                
        return {'real': real_count, 'fake': fake_count}
    
    def _calculate_class_weights(self):
        """Calculate class weights for handling imbalanced data."""
        if self.class_counts['real'] == 0 or self.class_counts['fake'] == 0:
            return torch.tensor([1.0, 1.0])
            
        total = self.class_counts['real'] + self.class_counts['fake']
        weight_real = total / (2.0 * self.class_counts['real'])
        weight_fake = total / (2.0 * self.class_counts['fake'])
        
        return torch.tensor([weight_real, weight_fake])

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
            video_frames, face_embeddings, temporal_consistency = self._load_video(video_path)
            if video_frames is None:
                raise ValueError(f"Video loading failed for sample {actual_idx}. Path: {video_path}")
                
            audio_tensor, audio_spectrogram = self._load_audio(audio_path)
            if audio_tensor is None:
                raise ValueError(f"Audio loading failed for sample {actual_idx}. Path: {audio_path}")

            # Load original video/audio if needed
            original_video_frames = None
            original_audio_tensor = None
            audio_visual_sync_features = None
            
            if sample.get('n_fakes', 0) > 0:
                if sample.get('modify_video', False) and original_video_path:
                    original_video_frames, _, _ = self._load_video(original_video_path)
                    if original_video_frames is None and self.logging:
                        print(f"⚠️ Warning: Original video loading failed for sample {actual_idx}. Path: {original_video_path}")
                        
                if sample.get('modify_audio', False) and original_audio_path:
                    original_audio_tensor, _ = self._load_audio(original_audio_path)
                    if original_audio_tensor is None and self.logging:
                        print(f"⚠️ Warning: Original audio loading failed for sample {actual_idx}. Path: {original_audio_path}")
                
                # Extract audio-visual synchronization features
                if video_frames is not None and audio_tensor is not None:
                    audio_visual_sync_features = self._extract_av_sync_features(video_frames, audio_tensor)

            # Create fake mask
            timestamps = sample.get('timestamps', [])
            fake_mask = torch.zeros(len(timestamps))
            for i, (_, start, end) in enumerate(timestamps):
                for f_start, f_end in sample.get('fake_periods', []):
                    if start < f_end and end > f_start:
                        fake_mask[i] = 1
                        break

            # Extract metadata features (e.g., compression artifacts, noise patterns)
            metadata_features = self._extract_metadata_features(video_path)
            
            # Calculate ELA (Error Level Analysis) for forgery detection
            ela_features = self._extract_ela_features(video_frames) if video_frames is not None else None
            
            label = torch.tensor(1 if sample.get('n_fakes', 0) > 0 else 0, dtype=torch.long)
            
            # Fine-grained deepfake type if available
            deepfake_type = sample.get('deepfake_type', 'unknown')
            deepfake_type_id = self._get_deepfake_type_id(deepfake_type)

            # Debugging: Log successful data loading
            if self.logging:
                print(f"✅ Successfully loaded sample at index {actual_idx}")

            return {
                'video_frames': video_frames,
                'audio': audio_tensor,
                'audio_spectrogram': audio_spectrogram,
                'label': label,
                'deepfake_type': deepfake_type_id,
                'original_video_frames': original_video_frames,
                'original_audio': original_audio_tensor,
                'fake_periods': sample.get('fake_periods', []),
                'timestamps': timestamps,
                'transcript': sample.get('transcript', ''),
                'fake_mask': fake_mask,
                'face_embeddings': face_embeddings,
                'temporal_consistency': temporal_consistency,
                'metadata_features': metadata_features,
                'ela_features': ela_features,
                'audio_visual_sync': audio_visual_sync_features,
                'file_path': video_path  # For explainability and error analysis
            }

        except Exception as e:
            if self.logging:
                print(f"❌ Error in __getitem__ for index {actual_idx}: {e}")
            # Return a placeholder sample instead of raising to avoid training crashes
            return self._get_placeholder_sample()

    def _get_deepfake_type_id(self, deepfake_type):
        """Map deepfake type to an ID for fine-grained classification."""
        deepfake_types = {
            'unknown': 0,
            'face_swap': 1,
            'face_reenactment': 2,
            'lip_sync': 3, 
            'audio_only': 4,
            'entire_synthesis': 5,
            'attribute_manipulation': 6
        }
        return deepfake_types.get(deepfake_type, 0)
        
    def _get_placeholder_sample(self):
        """Generate a placeholder sample when an error occurs."""
        # Create a blank tensor with appropriate dimensions
        video_frames = torch.zeros((self.max_frames, 3, 224, 224))
        audio_tensor = torch.zeros(self.audio_length)
        audio_spectrogram = torch.zeros((128, 128))
        label = torch.tensor(0, dtype=torch.long)  # Assume real by default
        
        return {
            'video_frames': video_frames,
            'audio': audio_tensor,
            'audio_spectrogram': audio_spectrogram,
            'label': label,
            'deepfake_type': 0,
            'original_video_frames': None,
            'original_audio': None,
            'fake_periods': [],
            'timestamps': [],
            'transcript': '',
            'fake_mask': torch.zeros(1),
            'face_embeddings': torch.zeros((1, 512)),
            'temporal_consistency': torch.tensor(1.0),
            'metadata_features': torch.zeros(10),
            'ela_features': torch.zeros((224, 224)),
            'audio_visual_sync': torch.zeros(5),
            'file_path': 'placeholder'
        }

    def _load_video(self, path):
        if not path or not os.path.exists(path):
            if self.logging:
                warnings.warn(f"⚠️ Video file not found: {path}")
            return None, None, None

        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                if self.logging:
                    warnings.warn(f"⚠️ Failed to open video: {path}")
                return None, None, None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if total_frames <= 0:
                if self.logging:
                    warnings.warn(f"⚠️ Video has no frames: {path}")
                return None, None, None
                
            # Create sampling indices for frames - more intelligent sampling strategy
            if self.phase == 'train':
                # Random sampling during training for data augmentation
                if total_frames <= self.max_frames:
                    frame_indices = list(range(total_frames))
                    # Repeat frames if not enough
                    if len(frame_indices) < self.max_frames:
                        frame_indices = frame_indices * math.ceil(self.max_frames / len(frame_indices))
                        frame_indices = frame_indices[:self.max_frames]
                else:
                    # Random frame sampling during training
                    frame_indices = sorted(random.sample(range(total_frames), self.max_frames))
            else:
                # Evenly distributed frames for validation/testing
                frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
                
            video_frames = []
            face_crops = []
            prev_face_locs = None

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    if self.logging:
                        warnings.warn(f"⚠️ Failed to read frame {frame_idx} from {path}")
                    continue
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply face detection if enabled
                face_crop = None
                face_detected = False
                consistency_score = 1.0  # Default - perfect consistency
                
                if self.detect_faces:
                    try:
                        # Convert to PIL for face detector
                        pil_img = Image.fromarray(frame_rgb)
                        
                        # Detect faces
                        boxes, probs = self.face_detector.detect(pil_img)
                        
                        if boxes is not None and len(boxes) > 0:
                            # Take the face with highest probability
                            box = boxes[0]
                            face_detected = True
                            
                            # Check temporal consistency with previous frame
                            if self.temporal_features and prev_face_locs is not None:
                                # Calculate movement between consecutive frames
                                movement = np.mean(np.abs(box - prev_face_locs))
                                
                                # Normalize by frame size and convert to a consistency score
                                # Lower movement = higher consistency
                                frame_diag = np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
                                consistency_score = 1.0 - min(1.0, movement / (frame_diag * 0.1))
                                
                            prev_face_locs = box
                            
                            # Crop and resize face
                            x1, y1, x2, y2 = box.astype(int)
                            face_crop = frame_rgb[max(0, y1):min(frame.shape[0], y2), 
                                               max(0, x1):min(frame.shape[1], x2)]
                                               
                            if face_crop.size != 0:
                                face_crop = cv2.resize(face_crop, (224, 224))
                                face_crops.append(face_crop)
                    except Exception as e:
                        if self.logging:
                            print(f"Face detection error on frame {frame_idx}: {e}")
                
                # Resize frame
                frame_rgb = cv2.resize(frame_rgb, (224, 224))
                
                # Apply transformations
                if self.transform:
                    if self.phase == 'train':  # Apply augmentation only during training
                        frame_rgb = self.transform(frame_rgb)
                    else:
                        # Just normalize for validation/testing
                        frame_rgb = self.transform(frame_rgb)
                else:
                    frame_rgb = torch.tensor(frame_rgb).permute(2, 0, 1).float() / 255.0
                
                video_frames.append(frame_rgb)
            cap.release()

            if not video_frames:
                if self.logging:
                    warnings.warn(f"⚠️ No valid frames extracted from video: {path}")
                return None, None, None

            # Stack frames into a tensor
            video_tensor = torch.stack(video_frames)
            
            # Process face crops if available
            face_embeddings = None
            if face_crops:
                # Stack face crops
                face_crops_tensor = torch.stack([
                    torch.tensor(crop).permute(2, 0, 1).float() / 255.0
                    for crop in face_crops
                ])
                
                # Create simple face embeddings (in a real model, you would use a face recognition network here)
                # This is just a placeholder - in practice use a pre-trained face embedding network
                face_embeddings = torch.mean(face_crops_tensor.reshape(face_crops_tensor.size(0), -1), dim=1)
            else:
                # No faces detected, use zeros as placeholder
                face_embeddings = torch.zeros((1, 512))
                
            # Temporal consistency feature
            temporal_consistency = torch.tensor(consistency_score).float()

            return video_tensor, face_embeddings, temporal_consistency

        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error loading video file: {path}. Error: {e}")
            return None, None, None

    def _load_audio(self, path):
        if not path or not os.path.exists(path):
            if self.logging:
                warnings.warn(f"⚠️ Audio file not found: {path}")
            return None, None
        try:
            # Load audio with torchaudio
            audio, sample_rate = torchaudio.load(path)
            audio = audio.squeeze(0).numpy()

            # Process audio length
            if len(audio) > self.audio_length:
                if self.phase == 'train':
                    # Random crop during training
                    start = random.randint(0, len(audio) - self.audio_length)
                    audio = audio[start:start + self.audio_length]
                else:
                    # Center crop during validation/testing
                    start = (len(audio) - self.audio_length) // 2
                    audio = audio[start:start + self.audio_length]
            else:
                # Pad if too short
                audio = np.pad(audio, (0, self.audio_length - len(audio)), mode='constant')

            # Apply audio augmentation if provided
            if self.audio_transform and self.phase == 'train':
                try:
                    audio = self.audio_transform(samples=audio, sample_rate=sample_rate)
                except Exception as audio_transform_error:
                    if self.logging:
                        warnings.warn(f"⚠️ Audio transform error for file {path}. Error: {audio_transform_error}")

            # Compute mel spectrogram for additional audio features
            audio_spec = None
            if self.compute_spectrograms:
                try:
                    # Compute mel spectrogram
                    mel_spec = librosa.feature.melspectrogram(
                        y=audio, 
                        sr=sample_rate,
                        n_mels=128,
                        hop_length=512,
                        n_fft=2048
                    )
                    
                    # Convert to dB scale
                    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    # Resize to 128x128
                    mel_spec = cv2.resize(mel_spec, (128, 128))
                    
                    # Normalize
                    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
                    
                    audio_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
                except Exception as e:
                    if self.logging:
                        warnings.warn(f"⚠️ Error computing spectrogram: {e}")
                    audio_spec = torch.zeros((1, 128, 128), dtype=torch.float32)
            else:
                audio_spec = torch.zeros((1, 128, 128), dtype=torch.float32)

            return torch.tensor(audio, dtype=torch.float32), audio_spec
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error loading audio file: {path}. Error: {e}")
            return None, None
            
    def _extract_metadata_features(self, video_path):
        """Extract metadata features like compression artifacts."""
        try:
            # Get video metadata using OpenCV
            cap = cv2.VideoCapture(video_path)
            
            # Extract basic metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Get file size in MB
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            
            # Calculate bitrate estimate
            duration = frame_count / fps if fps > 0 else 0
            bitrate = file_size_mb * 8 / duration if duration > 0 else 0
            
            # Extract noise level from first few frames
            noise_level = 0
            frames_to_check = min(10, frame_count)
            frame_diffs = []
            
            ret, prev_frame = cap.read()
            if ret:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                
                for _ in range(frames_to_check - 1):
                    ret, curr_frame = cap.read()
                    if not ret:
                        break
                        
                    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(prev_gray, curr_gray)
                    noise_level += np.mean(diff)
                    frame_diffs.append(np.mean(diff))
                    prev_gray = curr_gray
                    
                noise_level /= max(1, len(frame_diffs))
                noise_std = np.std(frame_diffs) if frame_diffs else 0
            
            cap.release()
            
            # Check for quantization artifacts
            quantization_metric = bitrate / (width * height) if width and height else 0
            
            # Compile features into tensor
            metadata_features = torch.tensor([
                fps / 30.0,  # Normalize fps
                min(1.0, file_size_mb / 10.0),  # Normalized file size
                min(1.0, bitrate / 5000.0),  # Normalized bitrate
                noise_level / 10.0,  # Normalized noise level
                noise_std / 10.0,  # Normalized noise std
                min(1.0, quantization_metric * 10),  # Normalized quantization
                width / 1920.0,  # Normalized width
                height / 1080.0,  # Normalized height
                min(1.0, (width * height) / (1920 * 1080)),  # Normalized resolution
                1.0 if file_size_mb < 0.5 else 0.0  # Small file flag (potential compression sign)
            ], dtype=torch.float32)
            
            return metadata_features
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting metadata features: {e}")
            return torch.zeros(10, dtype=torch.float32)
    
    def _extract_ela_features(self, video_frames):
        """Extract Error Level Analysis features for the first frame."""
        try:
            if video_frames is None or len(video_frames) == 0:
                return torch.zeros((224, 224), dtype=torch.float32)
                
            # Use first frame for ELA
            first_frame = video_frames[0].permute(1, 2, 0).cpu().numpy()
            
            # Convert to uint8
            first_frame = (first_frame * 255).astype(np.uint8)
            
            # Convert to PIL Image
            img = Image.fromarray(first_frame)
            
            # Save with a specific quality
            quality = 90
            import uuid
            import time
            # Create unique filename with process ID and time
            temp_filename = f"temp_ela_{os.getpid()}_{uuid.uuid4()}.jpg"
            img.save(temp_filename, 'JPEG', quality=quality)
            
            # Read back the saved image
            saved_img = np.array(Image.open(temp_filename))
            
            # Calculate absolute difference
            ela = np.abs(first_frame.astype(np.float32) - saved_img.astype(np.float32))
            
            # Use grayscale ELA 
            ela_gray = np.mean(ela, axis=2)
            
            # Resize to 224x224
            ela_resized = cv2.resize(ela_gray, (224, 224))
            
            # Normalize
            ela_normalized = ela_resized / ela_resized.max() if ela_resized.max() > 0 else ela_resized
            
            # Clean up
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception:
                    # If file is still being used, don't crash
                    pass

            return torch.tensor(ela_normalized, dtype=torch.float32)
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting ELA features: {e}")
            return torch.zeros((224, 224), dtype=torch.float32)
    
    def _extract_av_sync_features(self, video_frames, audio_tensor):
        """Extract features for audio-visual synchronization analysis."""
        try:
            # Calculate visual temporal features
            if video_frames.shape[0] < 2:
                return torch.zeros(5, dtype=torch.float32)
                
            # Calculate frame differences to detect motion
            frame_diffs = torch.mean(torch.abs(video_frames[1:] - video_frames[:-1]), dim=[1, 2, 3])
            
            # Calculate audio energy
            audio_chunks = audio_tensor.unfold(0, self.audio_length // self.max_frames, 
                                            self.audio_length // self.max_frames)
            audio_energy = torch.mean(audio_chunks**2, dim=1)
            
            # Ensure same lengths
            min_length = min(frame_diffs.shape[0], audio_energy.shape[0])
            frame_diffs = frame_diffs[:min_length]
            audio_energy = audio_energy[:min_length]
            
            # Calculate correlation
            if min_length > 1:
                # Normalize
                frame_diffs = (frame_diffs - frame_diffs.mean()) / (frame_diffs.std() + 1e-8)
                audio_energy = (audio_energy - audio_energy.mean()) / (audio_energy.std() + 1e-8)
                
                # Correlation
                correlation = torch.mean(frame_diffs * audio_energy)
                
                # Calculate lag by trying different offsets
                max_lag = min(5, min_length-1)
                best_lag = 0
                best_corr = correlation
                
                for lag in range(1, max_lag+1):
                    # Positive lag
                    corr_pos = torch.mean(frame_diffs[lag:] * audio_energy[:-lag]) if min_length > lag else torch.tensor(0.)
                    # Negative lag
                    corr_neg = torch.mean(frame_diffs[:-lag] * audio_energy[lag:]) if min_length > lag else torch.tensor(0.)
                    
                    if corr_pos > best_corr:
                        best_corr = corr_pos
                        best_lag = lag
                    if corr_neg > best_corr:
                        best_corr = corr_neg
                        best_lag = -lag
                
                # Features
                sync_features = torch.tensor([
                    correlation,  # Base correlation
                    float(best_lag),  # Best lag
                    best_corr,  # Best correlation
                    torch.std(frame_diffs),  # Motion consistency
                    torch.std(audio_energy)  # Audio consistency
                ], dtype=torch.float32)
                
                return sync_features
            else:
                return torch.zeros(5, dtype=torch.float32)
                
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting A/V sync features: {e}")
            return torch.zeros(5, dtype=torch.float32)


def get_transforms(phase='train'):
    """Get appropriate transforms for the dataset phase."""
    if phase == 'train':
        # Training transforms with augmentation
        video_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        audio_transform = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
        ])
    else:
        # Validation/test transforms
        video_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        audio_transform = None
        
    return video_transform, audio_transform


def get_data_loaders(
    json_path, data_dir, batch_size=8, validation_split=0.2, test_split=0.1,
    shuffle=True, num_workers=4, max_samples=None, detect_faces=True,
    compute_spectrograms=True, temporal_features=True
):
    """
    Load data loaders with an option to restrict the maximum number of samples.
    
    Parameters:
        json_path (str): Path to the dataset metadata JSON file.
        data_dir (str): Directory containing video and audio files.
        batch_size (int): Batch size for the data loaders.
        validation_split (float): Fraction of the dataset to use for validation.
        test_split (float): Fraction of the dataset to use for testing.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of worker threads for loading data.
        max_samples (int, optional): Maximum number of samples to load from the dataset.
        detect_faces (bool): Whether to detect and extract facial features.
        compute_spectrograms (bool): Whether to compute audio spectrograms.
        temporal_features (bool): Whether to compute temporal consistency features.
    
    Returns:
        tuple: Training, validation, and test data loaders.
    """
    # Get transforms for training and validation
    train_video_transform, train_audio_transform = get_transforms('train')
    val_video_transform, val_audio_transform = get_transforms('val')
    
    # Create training dataset
    train_dataset = MultiModalDeepfakeDataset(
        json_path=json_path,
        data_dir=data_dir,
        transform=train_video_transform,
        audio_transform=train_audio_transform,
        logging=True,  # Enable logging for debugging
        phase='train',
        detect_faces=detect_faces,
        compute_spectrograms=compute_spectrograms,
        temporal_features=temporal_features
    )
    
    # Create validation dataset
    val_dataset = MultiModalDeepfakeDataset(
        json_path=json_path,
        data_dir=data_dir,
        transform=val_video_transform,
        audio_transform=val_audio_transform,
        logging=True,  # Enable logging for debugging
        phase='val',
        detect_faces=detect_faces,
        compute_spectrograms=compute_spectrograms,
        temporal_features=temporal_features
    )
    
    # Create test dataset
    test_dataset = MultiModalDeepfakeDataset(
        json_path=json_path,
        data_dir=data_dir,
        transform=val_video_transform,
        audio_transform=val_audio_transform,
        logging=True,
        phase='test',
        detect_faces=detect_faces,
        compute_spectrograms=compute_spectrograms,
        temporal_features=temporal_features
    )
    
    # Get total number of valid samples
    num_samples = len(train_dataset)
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
    
    # Calculate split sizes
    val_count = int(np.floor(validation_split * num_samples))
    test_count = int(np.floor(test_split * num_samples))
    train_count = num_samples - val_count - test_count
    
    # Split indices
    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count+val_count]
    test_indices = indices[train_count+val_count:]

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Get class weights for weighted sampling
    class_weights = train_dataset.class_weights

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        sampler=test_sampler,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(f"✅ Dataset loaded with {len(train_indices)} training, {len(val_indices)} validation, and {len(test_indices)} test samples.")
    return train_loader, val_loader, test_loader, class_weights


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
        if key in ['label', 'deepfake_type']:
            # Make sure we're stacking tensors, not ints
            values = [item[key] if isinstance(item[key], torch.Tensor) else torch.tensor(item[key]) for item in batch]
            result[key] = torch.stack(values)
        elif key in ['video_frames', 'audio', 'audio_spectrogram', 'metadata_features', 'temporal_consistency']:
            # Stack tensors
            values = [item[key] for item in batch if item[key] is not None]
            if values:
                try:
                    result[key] = torch.stack(values)
                except:
                    # If can't stack, convert to tensors and stack
                    try:
                        values = [v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in values]
                        result[key] = torch.stack(values)
                    except:
                        # If still can't stack, store as list
                        result[key] = values
            else:
                result[key] = None
        elif key in ['original_video_frames', 'original_audio', 'ela_features', 'audio_visual_sync', 'face_embeddings']:
            # Handle potentially missing data
            values = [item[key] for item in batch if item[key] is not None]
            if values and all(v is not None and isinstance(v, torch.Tensor) for v in values):
                try:
                    result[key] = torch.stack(values)
                except:
                    # If can't stack (different sizes), store as list
                    result[key] = values
            else:
                # Try to convert non-tensor values to tensors
                try:
                    tensor_values = []
                    for v in values:
                        if isinstance(v, torch.Tensor):
                            tensor_values.append(v)
                        else:
                            try:
                                tensor_values.append(torch.tensor(v))
                            except:
                                pass  # Skip if can't convert to tensor
                    
                    if tensor_values:
                        result[key] = torch.stack(tensor_values)
                    else:
                        result[key] = None
                except:
                    result[key] = None
        elif key in ['fake_periods', 'timestamps']:
            # List of lists, don't stack
            result[key] = [item[key] for item in batch]
        elif key in ['transcript', 'file_path']:
            # List of strings
            result[key] = [item[key] for item in batch]
        elif key == 'fake_mask':
            # Don't stack fake_masks of different sizes, keep as list
            result[key] = [item[key] for item in batch]
        else:
            # Handle other types if needed
            result[key] = [item[key] for item in batch]
    
    return result
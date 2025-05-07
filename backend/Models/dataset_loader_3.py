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
            
            # Extract physiological signals
            physiological_features = self._extract_physiological_signals(face_embeddings)
            
            # Extract ocular features
            ocular_features = self._extract_ocular_features(face_embeddings)
            
            # Extract lip-audio sync features
            lip_audio_sync_features = self._extract_lip_audio_sync_features(video_frames, audio_tensor)
            
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
                'physiological_features': physiological_features,
                'ocular_features': ocular_features,
                'lip_audio_sync_features': lip_audio_sync_features,
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
            'physiological_features': torch.zeros(8),
            'ocular_features': torch.zeros(10),
            'lip_audio_sync_features': torch.zeros(7),
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

    def _extract_physiological_signals(self, face_regions):
        """
        Extract physiological signals like heartbeat (rPPG) from facial regions.
        This is a baseline implementation - production systems would use more 
        sophisticated algorithms.
        """
        if face_regions is None or len(face_regions) < 10:  # Need enough frames
            return torch.zeros(8, dtype=torch.float32)
            
        try:
            # Convert tensor to numpy
            if isinstance(face_regions, torch.Tensor):
                # Use CPU tensor for numpy operations
                face_np = face_regions.cpu().numpy()
            else:
                face_np = face_regions
                
            # Extract green channel (most sensitive to blood flow changes)
            batch_size, seq_len = face_np.shape[:2]
            
            # Extract regions of interest (forehead and cheeks) where blood flow is visible
            h, w = face_np.shape[2:4] if len(face_np.shape) >= 4 else (0, 0)
            
            # Skin regions: forehead (upper 1/3), cheeks (middle sides)
            # These regions typically show skin color variations due to blood flow
            forehead_region = face_np[:, :, :int(h/3), int(w/4):int(3*w/4), 1] if h > 0 and w > 0 else None
            left_cheek = face_np[:, :, int(h/3):int(2*h/3), :int(w/3), 1] if h > 0 and w > 0 else None
            right_cheek = face_np[:, :, int(h/3):int(2*h/3), int(2*w/3):, 1] if h > 0 and w > 0 else None
            
            # Combine ROIs
            signal_regions = []
            if forehead_region is not None:
                signal_regions.append(np.mean(forehead_region, axis=(2, 3)))
            if left_cheek is not None:
                signal_regions.append(np.mean(left_cheek, axis=(2, 3)))
            if right_cheek is not None:
                signal_regions.append(np.mean(right_cheek, axis=(2, 3)))
                
            if not signal_regions:
                return torch.zeros(8, dtype=torch.float32)
                
            # Average signal across all ROIs
            raw_signal = np.mean(signal_regions, axis=0)
            
            # Detrend signal (remove low frequency components)
            from scipy import signal as scipy_signal
            detrended = scipy_signal.detrend(raw_signal, axis=1)
            
            # Normalize
            normalized = (detrended - np.mean(detrended, axis=1, keepdims=True)) / (np.std(detrended, axis=1, keepdims=True) + 1e-6)
            
            # Simple heart rate estimation using FFT
            heart_rates = []
            heart_rate_snrs = []  # Signal-to-noise ratio
            
            for b in range(batch_size):
                # Apply FFT to get frequency components
                # This estimates the heart rate from skin color variations
                fft_data = np.abs(np.fft.rfft(normalized[b]))
                freqs = np.fft.rfftfreq(seq_len) * seq_len  # Assume 30fps
                
                # Focus on frequencies in human heart rate range (0.75-3.0 Hz or 45-180 BPM)
                valid_range = (freqs >= 0.75) & (freqs <= 3.0)
                if np.any(valid_range):
                    valid_freqs = freqs[valid_range]
                    valid_fft = fft_data[valid_range]
                    
                    # Find the dominant frequency (heart rate)
                    if len(valid_fft) > 0:
                        peak_idx = np.argmax(valid_fft)
                        heart_rate = valid_freqs[peak_idx] * 60  # Convert Hz to BPM
                        
                        # Calculate SNR
                        peak_power = valid_fft[peak_idx]
                        avg_power = np.mean(valid_fft)
                        snr = peak_power / (avg_power + 1e-8)
                        
                        heart_rates.append(heart_rate)
                        heart_rate_snrs.append(snr)
                    else:
                        heart_rates.append(75.0)  # Default
                        heart_rate_snrs.append(1.0)
                else:
                    heart_rates.append(75.0)  # Default
                    heart_rate_snrs.append(1.0)
            
            # Calculate heart rate variability
            if len(heart_rates) > 1:
                hrv = np.std(heart_rates)
            else:
                hrv = 0.0
                
            # Breathing rate estimation (lower frequency component of the signal)
            breathing_rates = []
            for b in range(batch_size):
                # We look for frequencies between 0.1-0.5 Hz (6-30 breaths per minute)
                fft_data = np.abs(np.fft.rfft(normalized[b]))
                freqs = np.fft.rfftfreq(seq_len) * seq_len  # Assume 30fps
                
                breath_range = (freqs >= 0.1) & (freqs <= 0.5)
                if np.any(breath_range):
                    breath_freqs = freqs[breath_range]
                    breath_fft = fft_data[breath_range]
                    
                    if len(breath_fft) > 0:
                        peak_idx = np.argmax(breath_fft)
                        breathing_rate = breath_freqs[peak_idx] * 60  # Convert Hz to breaths per minute
                        breathing_rates.append(breathing_rate)
                    else:
                        breathing_rates.append(15.0)  # Default
                else:
                    breathing_rates.append(15.0)  # Default
                    
            # Skin color variation metrics
            color_variations = []
            naturality_scores = []
            
            # Analyze color distributions in skin regions
            for b in range(batch_size):
                # Extract all color channels for this batch item
                if h > 0 and w > 0 and len(face_np.shape) >= 5:
                    # Calculate standard deviation of skin regions over time (natural faces show variations)
                    skin_regions = np.concatenate([
                        face_np[b, :, :int(h/3), int(w/4):int(3*w/4), :].reshape(-1, 3),  # forehead
                        face_np[b, :, int(h/3):int(2*h/3), :int(w/3), :].reshape(-1, 3),  # left cheek
                        face_np[b, :, int(h/3):int(2*h/3), int(2*w/3):, :].reshape(-1, 3)  # right cheek
                    ], axis=0)
                    
                    # Calculate color variation over time
                    color_var = np.std(skin_regions, axis=0).mean()
                    color_variations.append(color_var)
                    
                    # Calculate "naturality" score based on color distribution
                    # Natural faces show specific patterns in RGB distribution
                    r_g_ratio = np.mean(skin_regions[:, 0]) / (np.mean(skin_regions[:, 1]) + 1e-8)
                    g_b_ratio = np.mean(skin_regions[:, 1]) / (np.mean(skin_regions[:, 2]) + 1e-8)
                    
                    # Natural face has these approximate ratios
                    r_g_natural = 1.1
                    g_b_natural = 1.15
                    
                    naturality = 1.0 - (abs(r_g_ratio - r_g_natural) / r_g_natural + 
                                       abs(g_b_ratio - g_b_natural) / g_b_natural) / 2
                    naturality_scores.append(max(0, min(1, naturality)))
                else:
                    color_variations.append(0.02)  # Default
                    naturality_scores.append(0.5)  # Default
            
            # Compile features
            avg_heart_rate = np.mean(heart_rates) if heart_rates else 75.0
            avg_heart_rate_snr = np.mean(heart_rate_snrs) if heart_rate_snrs else 1.0
            avg_breathing_rate = np.mean(breathing_rates) if breathing_rates else 15.0
            avg_color_variation = np.mean(color_variations) if color_variations else 0.02
            avg_naturality = np.mean(naturality_scores) if naturality_scores else 0.5
            
            # Create feature tensor
            physio_features = torch.tensor([
                avg_heart_rate / 150.0,  # Normalize heart rate
                hrv / 20.0,  # Normalize HRV
                avg_heart_rate_snr,  # SNR already normalized
                avg_breathing_rate / 30.0,  # Normalize breathing rate
                avg_color_variation / 0.05,  # Normalize color variation
                avg_naturality,  # Already normalized
                1.0 if avg_heart_rate > 40 and avg_heart_rate < 180 else 0.0,  # Valid heart rate flag
                1.0 if avg_breathing_rate > 8 and avg_breathing_rate < 25 else 0.0  # Valid breathing rate flag
            ], dtype=torch.float32)
            
            return physio_features
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting physiological signals: {e}")
            return torch.zeros(8, dtype=torch.float32)

    def _extract_ocular_features(self, face_regions):
        """
        Extract ocular features like eye movements, blink patterns, and
        pupil dilation from facial video.
        """
        if face_regions is None or len(face_regions) < 5:  # Need multiple frames
            return torch.zeros(10, dtype=torch.float32)
            
        try:
            # Convert tensor to numpy if needed
            if isinstance(face_regions, torch.Tensor):
                face_np = face_regions.cpu().numpy()
            else:
                face_np = face_regions
                
            batch_size, seq_len = face_np.shape[:2]
            h, w = face_np.shape[2:4] if len(face_np.shape) >= 4 else (0, 0)
            
            # Extract eye regions
            # This is a simplified version - a real system would use facial landmarks
            left_eye_region = face_np[:, :, int(h/4):int(h/2), int(w/5):int(2*w/5)] if h > 0 and w > 0 else None
            right_eye_region = face_np[:, :, int(h/4):int(h/2), int(3*w/5):int(4*w/5)] if h > 0 and w > 0 else None
            
            if left_eye_region is None or right_eye_region is None:
                return torch.zeros(10, dtype=torch.float32)
                
            # Calculate eye movements by tracking changes in eye regions
            eye_movements = []
            for b in range(batch_size):
                movement_scores = []
                
                # Process sequential frames
                for t in range(1, seq_len):
                    # Calculate abs difference between consecutive frames in eye regions
                    left_diff = np.mean(np.abs(left_eye_region[b, t] - left_eye_region[b, t-1]))
                    right_diff = np.mean(np.abs(right_eye_region[b, t] - right_eye_region[b, t-1]))
                    
                    # Average eye movement
                    avg_movement = (left_diff + right_diff) / 2
                    movement_scores.append(avg_movement)
                
                if movement_scores:
                    eye_movements.append(movement_scores)
            
            # Calculate blink detection
            blink_patterns = []
            for b in range(batch_size):
                blinks = []
                
                # Convert eye regions to grayscale
                left_gray = np.mean(left_eye_region[b], axis=3) if len(left_eye_region[b].shape) > 3 else left_eye_region[b]
                right_gray = np.mean(right_eye_region[b], axis=3) if len(right_eye_region[b].shape) > 3 else right_eye_region[b]
                
                # Calculate average eye openness over time
                for t in range(seq_len):
                    # Use simple edge detection as proxy for eye openness
                    # In a real system, you'd use more sophisticated methods
                    left_edges = np.mean(np.abs(np.diff(left_gray[t], axis=0))) + np.mean(np.abs(np.diff(left_gray[t], axis=1)))
                    right_edges = np.mean(np.abs(np.diff(right_gray[t], axis=0))) + np.mean(np.abs(np.diff(right_gray[t], axis=1)))
                    
                    openness = (left_edges + right_edges) / 2
                    blinks.append(openness)
                
                # Detect blinks by finding dips in openness
                if len(blinks) > 2:
                    # Normalize
                    blinks = np.array(blinks)
                    if np.max(blinks) > np.min(blinks):
                        blinks = (blinks - np.min(blinks)) / (np.max(blinks) - np.min(blinks))
                        
                    # Detect when eye openness drops below threshold
                    threshold = 0.4
                    blink_candidates = np.where(blinks < threshold)[0]
                    
                    # Group consecutive frames to find blink events
                    if len(blink_candidates) > 0:
                        blink_events = []
                        current_blink = [blink_candidates[0]]
                        
                        for i in range(1, len(blink_candidates)):
                            if blink_candidates[i] - blink_candidates[i-1] <= 2:  # consecutive or 1 frame gap
                                current_blink.append(blink_candidates[i])
                            else:
                                blink_events.append(current_blink)
                                current_blink = [blink_candidates[i]]
                                
                        if current_blink:
                            blink_events.append(current_blink)
                            
                        # Count valid blinks (duration between 2-10 frames)
                        valid_blinks = [b for b in blink_events if 2 <= len(b) <= 10]
                        blink_patterns.append(len(valid_blinks))
                    else:
                        blink_patterns.append(0)
                else:
                    blink_patterns.append(0)
            
            # Calculate saccades (quick eye movements)
            saccade_features = []
            for b in range(batch_size):
                if len(eye_movements) > b and len(eye_movements[b]) > 3:
                    # Saccades are rapid movements followed by fixation periods
                    movements = np.array(eye_movements[b])
                    
                    # Identify potential saccades (large movements)
                    threshold = np.mean(movements) + 1.5 * np.std(movements)
                    saccade_candidates = movements > threshold
                    
                    # Count saccades
                    saccade_count = 0
                    in_saccade = False
                    
                    for i in range(len(saccade_candidates)):
                        if saccade_candidates[i] and not in_saccade:
                            in_saccade = True
                            saccade_count += 1
                        elif not saccade_candidates[i]:
                            in_saccade = False
                    
                    # Calculate saccade rate (per second, assuming 30fps)
                    saccade_rate = saccade_count / (len(movements) / 30)
                    saccade_features.append(saccade_rate)
                else:
                    saccade_features.append(0.0)
            
            # Pupil dilation analysis 
            pupil_features = []
            for b in range(batch_size):
                # Simple pupil detection using thresholding on eye regions
                # This is simplified - production systems would use more robust methods
                pupil_sizes = []
                
                for t in range(seq_len):
                    # Convert to grayscale
                    left_eye = np.mean(left_eye_region[b, t], axis=2) if len(left_eye_region[b, t].shape) >= 3 else left_eye_region[b, t]
                    right_eye = np.mean(right_eye_region[b, t], axis=2) if len(right_eye_region[b, t].shape) >= 3 else right_eye_region[b, t]
                    
                    # Threshold to estimate pupil
                    try:
                        if np.max(left_eye) > np.min(left_eye):
                            left_threshold = np.min(left_eye) + 0.3 * (np.max(left_eye) - np.min(left_eye))
                            left_pupil = np.sum(left_eye < left_threshold) / left_eye.size
                        else:
                            left_pupil = 0.3  # Default
                            
                        if np.max(right_eye) > np.min(right_eye):
                            right_threshold = np.min(right_eye) + 0.3 * (np.max(right_eye) - np.min(right_eye))
                            right_pupil = np.sum(right_eye < right_threshold) / right_eye.size
                        else:
                            right_pupil = 0.3  # Default
                            
                        # Average pupil size
                        pupil_size = (left_pupil + right_pupil) / 2
                        pupil_sizes.append(pupil_size)
                    except:
                        pupil_sizes.append(0.3)  # Default
                
                # Calculate pupil variation metrics
                if len(pupil_sizes) > 1:
                    # Normalize
                    pupil_array = np.array(pupil_sizes)
                    pupil_mean = np.mean(pupil_array)
                    pupil_std = np.std(pupil_array)
                    
                    # Autocorrelation indicates reactivity
                    autocorr = np.correlate(pupil_array - pupil_mean, pupil_array - pupil_mean, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    autocorr = autocorr / autocorr[0]
                    
                    # Calculate reaction metrics
                    if len(autocorr) > 1:
                        pupil_reactivity = 1.0 - autocorr[1]  # Higher value means more reactivity
                    else:
                        pupil_reactivity = 0.5  # Default
                    
                    pupil_features.append((pupil_mean, pupil_std, pupil_reactivity))
                else:
                    pupil_features.append((0.3, 0.05, 0.5))  # Default values
            
            # Micro-expression analysis
            # Simplified version that looks for brief local motions in face regions
            micro_expr_scores = []
            
            for b in range(batch_size):
                micro_movements = []
                
                # Look at facial regions where micro-expressions are common
                if h > 0 and w > 0:
                    # Define regions - in real implementation, use precise facial landmarks
                    mouth_region = face_np[b, :, int(2*h/3):int(5*h/6), int(w/3):int(2*w/3)]
                    brow_region = face_np[b, :, int(h/6):int(h/4), int(w/4):int(3*w/4)]
                    
                    # Calculate frame-by-frame differences
                    for t in range(1, seq_len):
                        mouth_diff = np.mean(np.abs(mouth_region[t] - mouth_region[t-1]))
                        brow_diff = np.mean(np.abs(brow_region[t] - brow_region[t-1]))
                        
                        # Combine differences
                        micro_movement = (mouth_diff + brow_diff) / 2
                        micro_movements.append(micro_movement)
                    
                    # Analyze pattern of movements for micro-expression signs
                    if len(micro_movements) > 2:
                        movements = np.array(micro_movements)
                        
                        # Micro-expressions are brief - look for short, isolated movements
                        # Normalize
                        if np.max(movements) > np.min(movements):
                            movements = (movements - np.min(movements)) / (np.max(movements) - np.min(movements))
                        
                        # Look for peaks in movement
                        threshold = np.mean(movements) + 1.0 * np.std(movements)
                        peak_candidates = movements > threshold
                        
                        # Count isolated peaks (potential micro-expressions)
                        isolated_peaks = 0
                        for i in range(1, len(peak_candidates)-1):
                            if peak_candidates[i] and not peak_candidates[i-1] and not peak_candidates[i+1]:
                                isolated_peaks += 1
                        
                        # Calculate micro-expression score
                        micro_expr = isolated_peaks / (len(movements) / 30)  # normalize to per second
                        micro_expr_scores.append(micro_expr)
                    else:
                        micro_expr_scores.append(0.1)  # Default value
                else:
                    micro_expr_scores.append(0.1)  # Default value
            
            # Compile all metrics into feature vector
            avg_blink_count = np.mean(blink_patterns) if blink_patterns else 0.3
            avg_saccade_rate = np.mean(saccade_features) if saccade_features else 0.5
            
            # Average pupil features
            avg_pupil_size = np.mean([p[0] for p in pupil_features]) if pupil_features else 0.3
            avg_pupil_variation = np.mean([p[1] for p in pupil_features]) if pupil_features else 0.05
            avg_pupil_reactivity = np.mean([p[2] for p in pupil_features]) if pupil_features else 0.5
            
            # Average micro-expression score
            avg_micro_expr = np.mean(micro_expr_scores) if micro_expr_scores else 0.1
            
            # Movement regularity (more regular in deepfakes)
            movement_regularity = 0.5
            if eye_movements and len(eye_movements[0]) > 1:
                # Calculate coefficient of variation for movement patterns
                movement_array = np.array(eye_movements[0])
                movement_regularity = np.std(movement_array) / (np.mean(movement_array) + 1e-6)
                movement_regularity = 1.0 - min(1.0, movement_regularity / 0.5)  # Convert to 0-1 scale
            
            # Calculate if blink rate is in natural range (15-30 blinks/min)
            blink_rate_per_min = avg_blink_count * (60 / (seq_len / 30))
            normal_blink_rate = 1.0 if 10 <= blink_rate_per_min <= 30 else (
                1.0 - min(1.0, abs(blink_rate_per_min - 20) / 20)
            )
            
            # Create feature tensor
            ocular_features = torch.tensor([
                avg_blink_count / 3.0,  # Normalize blink count
                avg_saccade_rate / 3.0,  # Normalize saccade rate
                avg_pupil_size,  # Already normalized
                avg_pupil_variation * 10,  # Scale up variation
                avg_pupil_reactivity,  # Already normalized
                avg_micro_expr / 0.5,  # Normalize micro-expression score
                movement_regularity,  # Already normalized
                normal_blink_rate,  # Already normalized
                min(1.0, np.mean([m[0] for m in eye_movements]) * 50) if eye_movements else 0.5,  # Overall movement level
                1.0 if avg_blink_count > 0 else 0.0  # Has blinks flag
            ], dtype=torch.float32)
            
            return ocular_features
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting ocular features: {e}")
            return torch.zeros(10, dtype=torch.float32)

    def _extract_lip_audio_sync_features(self, video_frames, audio_tensor):
        """
        Extract features measuring synchronization between lip movements and audio.
        Detects inconsistencies in lip-audio timing and phoneme-viseme mismatches.
        """
        if video_frames is None or audio_tensor is None:
            return torch.zeros(7, dtype=torch.float32)
            
        try:
            # Convert tensors to numpy if needed
            if isinstance(video_frames, torch.Tensor):
                video_np = video_frames.cpu().numpy()
            else:
                video_np = video_frames
                
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = audio_tensor
                
            batch_size, seq_len = video_np.shape[:2]
            h, w = video_np.shape[2:4] if len(video_np.shape) >= 4 else (0, 0)
            
            # Extract lip region
            # In a real system, use precise facial landmarks
            lip_regions = video_np[:, :, int(2*h/3):int(5*h/6), int(w/3):int(2*w/3)] if h > 0 and w > 0 else None
            
            if lip_regions is None:
                return torch.zeros(7, dtype=torch.float32)
                
            # Features to extract
            sync_scores = []
            temporal_offsets = []
            speech_consistency = []
            phoneme_match = []
            articulation_naturalness = []
            movement_naturalness = []
            style_consistency = []
            
            # Process each batch item
            for b in range(batch_size):
                # Extract lip movement signal
                lip_movement = []
                
                # Convert lip region to grayscale and calculate frame differences
                lip_gray = np.mean(lip_regions[b], axis=3) if len(lip_regions[b].shape) > 3 else lip_regions[b]
                
                for t in range(1, seq_len):
                    # Calculate absolute difference between consecutive frames
                    lip_diff = np.mean(np.abs(lip_gray[t] - lip_gray[t-1]))
                    lip_movement.append(lip_diff)
                
                # Get audio energy - reshape audio to match video sequence
                # This assumes audio is sampled at 16kHz and video at 30fps
                samples_per_frame = len(audio_np) // seq_len
                audio_frames = [audio_np[i:i+samples_per_frame] for i in range(0, len(audio_np), samples_per_frame)][:seq_len]
                
                audio_energy = [np.sqrt(np.mean(frame**2)) for frame in audio_frames if len(frame) > 0]
                
                # Make sure we have enough frames
                min_frames = min(len(lip_movement), len(audio_energy))
                if min_frames < 3:
                    sync_scores.append(0.5)
                    temporal_offsets.append(0.0)
                    speech_consistency.append(0.5)
                    phoneme_match.append(0.5)
                    articulation_naturalness.append(0.5)
                    movement_naturalness.append(0.5)
                    style_consistency.append(0.5)
                    continue
                    
                # Trim to same length
                lip_movement = lip_movement[:min_frames]
                audio_energy = audio_energy[:min_frames]
                
                # Normalize signals
                lip_movement = np.array(lip_movement)
                audio_energy = np.array(audio_energy)
                
                if np.max(lip_movement) > np.min(lip_movement):
                    lip_movement = (lip_movement - np.min(lip_movement)) / (np.max(lip_movement) - np.min(lip_movement))
                if np.max(audio_energy) > np.min(audio_energy):
                    audio_energy = (audio_energy - np.min(audio_energy)) / (np.max(audio_energy) - np.min(audio_energy))
                
                # Calculate cross-correlation to find sync offset
                corr = np.correlate(lip_movement, audio_energy, mode='full')
                corr = corr / max(np.max(corr), 1e-8)  # Normalize
                
                # Find max correlation and corresponding lag
                max_idx = np.argmax(corr)
                max_corr = corr[max_idx]
                
                # Convert to lag in frames
                lag = max_idx - (len(lip_movement) - 1)
                
                # Calculate absolute lag in seconds (assuming 30fps)
                abs_lag_sec = abs(lag) / 30.0
                
                # Sync score decreases as lag increases (we want low lag)
                sync_score = max(0, 1.0 - abs_lag_sec * 2)  # 0.5 sec lag = 0 sync
                
                # Normalized offset (-1 to 1)
                temporal_offset = min(1.0, max(-1.0, lag / 10.0))
                
                # Speech consistency - correlation between signals
                # Higher correlation = more consistency in timing
                speech_consist = max(0, max_corr)
                
                # Phoneme-viseme matching score
                # Here we use a simplified approach that looks at pattern similarity
                # In a real system, you would use a phoneme-viseme alignment model
                
                # Check if audio energy peaks coincide with lip movement peaks
                audio_peaks = scipy_signal.find_peaks(audio_energy)[0]
                lip_peaks = scipy_signal.find_peaks(lip_movement)[0]
                
                # Calculate temporal distance between each audio peak and nearest lip peak
                if len(audio_peaks) > 0 and len(lip_peaks) > 0:
                    distances = []
                    for ap in audio_peaks:
                        # Find closest lip peak
                        min_dist = min(abs(ap - lp) for lp in lip_peaks)
                        distances.append(min_dist)
                    
                    avg_distance = np.mean(distances)
                    # Convert to a score (lower distance = higher score)
                    phoneme_match_score = max(0, 1.0 - avg_distance / 5.0)
                else:
                    phoneme_match_score = 0.5  # Default
                
                # Articulation naturalness
                # Check if lip movement has natural variability
                lip_movement_std = np.std(lip_movement)
                articulation_score = min(1.0, lip_movement_std * 5.0)
                
                # Movement naturalness
                # Check if there's a natural decay in lip movement after audio peaks
                if len(audio_peaks) > 0:
                    decay_scores = []
                    for ap in audio_peaks:
                        if ap + 5 < len(lip_movement):
                            # Look at 5 frames after audio peak
                            movement_after_peak = lip_movement[ap:ap+5]
                            # Check if there's a natural decay pattern
                            is_decaying = all(movement_after_peak[i] >= movement_after_peak[i+1] for i in range(len(movement_after_peak)-1))
                            decay_scores.append(1.0 if is_decaying else 0.0)
                    
                    movement_natural_score = np.mean(decay_scores) if decay_scores else 0.5
                else:
                    movement_natural_score = 0.5  # Default
                    
                # Speaking style consistency
                # Check if the pattern of lip movements is consistent
                if len(lip_movement) > 10:
                    # Chunk the sequence
                    chunk_size = 5
                    chunks = [lip_movement[i:i+chunk_size] for i in range(0, len(lip_movement)-chunk_size, chunk_size)]
                    
                    # Calculate correlation between adjacent chunks
                    chunk_correlations = []
                    for i in range(len(chunks)-1):
                        corr = np.corrcoef(chunks[i], chunks[i+1])[0, 1]
                        if not np.isnan(corr):
                            chunk_correlations.append(corr)
                    
                    style_consistency_score = np.mean(chunk_correlations) if chunk_correlations else 0.5
                    # Convert to 0-1 range (higher correlation = more consistent)
                    style_consistency_score = (style_consistency_score + 1) / 2
                else:
                    style_consistency_score = 0.5  # Default
                
                # Store features
                sync_scores.append(sync_score)
                temporal_offsets.append(temporal_offset)
                speech_consistency.append(speech_consist)
                phoneme_match.append(phoneme_match_score)
                articulation_naturalness.append(articulation_score)
                movement_naturalness.append(movement_natural_score)
                style_consistency.append(style_consistency_score)
            
            # Average across batch
            avg_sync = np.mean(sync_scores) if sync_scores else 0.5
            avg_offset = np.mean(temporal_offsets) if temporal_offsets else 0.0
            avg_speech_consist = np.mean(speech_consistency) if speech_consistency else 0.5
            avg_phoneme = np.mean(phoneme_match) if phoneme_match else 0.5
            avg_articulation = np.mean(articulation_naturalness) if articulation_naturalness else 0.5
            avg_movement = np.mean(movement_naturalness) if movement_naturalness else 0.5
            avg_style = np.mean(style_consistency) if style_consistency else 0.5
            
            # Create feature tensor
            lip_sync_features = torch.tensor([
                avg_sync,
                avg_offset,
                avg_speech_consist,
                avg_phoneme,
                avg_articulation,
                avg_movement,
                avg_style
            ], dtype=torch.float32)
            
            return lip_sync_features
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting lip-audio sync features: {e}")
            return torch.zeros(7, dtype=torch.float32)


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
        elif key in ['original_video_frames', 'original_audio', 'ela_features', 'audio_visual_sync', 'face_embeddings',
                     'physiological_features', 'ocular_features', 'lip_audio_sync_features']:
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

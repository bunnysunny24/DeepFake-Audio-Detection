import torch.nn.functional as F
import scipy
from scipy import signal
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import torchaudio
import cv2
import warnings
import traceback
import random
import math
import librosa
import uuid
import multiprocessing
import uuid
import dlib
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Shift
import albumentations as A

# Import our improved augmentation techniques
try:
    from improved_augmentation import (
        get_advanced_video_transforms, 
        get_advanced_audio_transforms,
        TemporalConsistencyAugmenter,
        mix_up_augmentation
    )
except ImportError:
    print("Warning: improved_augmentation.py not found, falling back to standard augmentation")
try:
    from facenet_pytorch import MTCNN
except ImportError:
    print("Warning: facenet_pytorch not found, some face detection features will be limited")
from scipy.signal import spectrogram
import librosa
from PIL import Image
from PIL import Image
import random
import math
import scipy.ndimage as ndimage
import traceback
import uuid

class MultiModalDeepfakeDataset(Dataset):
    # NOTE: To maximize GPU usage, ensure that all output tensors (e.g., video_frames, audio, features)
    # are moved to the correct device (e.g., .to('cuda')) in your training loop or collate_fn.
    # This is typically handled in the training script, not the dataset itself, for best flexibility.
    def __init__(self, json_path, data_dir, max_frames=16, audio_length=8000, transform=None, audio_transform=None, 
                 logging=False, phase='train', detect_faces=True, compute_spectrograms=True, temporal_features=True,
                 enhanced_preprocessing=True):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at: {json_path}")
        
        # Load JSON with robust error handling
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✅ Successfully loaded JSON with {len(self.data)} entries")
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            print("🔧 Attempting to repair JSON file...")
            
            # Try to load partial JSON by reading line by line
            self.data = []
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Try to find complete JSON objects
                if content.startswith('[') and not content.endswith(']'):
                    # Likely truncated array, try to add closing bracket
                    content = content.rstrip().rstrip(',') + ']'
                    self.data = json.loads(content)
                    print(f"✅ Repaired JSON, loaded {len(self.data)} entries")
                else:
                    raise ValueError("Unable to repair JSON automatically")
                    
            except Exception as repair_error:
                print(f"❌ Failed to repair JSON: {repair_error}")
                raise ValueError(f"JSON file is corrupted and cannot be repaired: {json_path}")
        except Exception as e:
            raise ValueError(f"Error loading JSON file {json_path}: {e}")

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
        self.enhanced_preprocessing = enhanced_preprocessing
        
        # Initialize error counters early
        self.face_detection_error_count = 0
        self.max_face_detection_errors_to_print = 5  # Reduce error messages for cleaner output
        self._sample_count = 0  # Track sample count for periodic detailed logging
        
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
        
        # Initialize facial landmark detector if available (for enhanced facial analysis)
        if self.enhanced_preprocessing:
            try:
                import dlib
                # Try to load dlib's face detector and landmark predictor
                self.dlib_detector = dlib.get_frontal_face_detector()
                
                # Use relative path to the model file in the Models directory
                model_path = "shape_predictor_68_face_landmarks.dat"
                # Check if file exists in current directory
                if not os.path.exists(model_path):
                    # Try finding it in the script's directory
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")
                    
                if os.path.exists(model_path):
                    self.landmark_predictor = dlib.shape_predictor(model_path)
                    print("✅ Facial landmark predictor initialized successfully")
                else:
                    print(f"Warning: Facial landmark model not found at {model_path}")
                    self.landmark_predictor = None
            except Exception as e:
                print(f"Warning: Could not initialize facial landmark detector: {e}")
                self.dlib_detector = None
                self.landmark_predictor = None
        
        print(f"Dataset initialized with {len(self.valid_indices)} valid samples out of {len(self.data)} total.")
        print(f"Class distribution: {self.class_counts}")

    def _validate_dataset(self):
        """Pre-validate all samples in the dataset to identify valid ones."""
        print("Starting dataset validation...")
        
        # Use the entire dataset for production
        max_to_validate = len(self.data)
        print(f"✅ VALIDATING ALL {max_to_validate} SAMPLES IN THE DATASET")
        
        valid_indices = []
        
        # Add progress indicators for validation
        progress_interval = max(1, max_to_validate // 100)  # Show progress ~100 times
        
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
        
        # Log dataset distribution
        total = real_count + fake_count
        real_pct = (real_count / total * 100) if total > 0 else 0
        fake_pct = (fake_count / total * 100) if total > 0 else 0
        phase_name = getattr(self, 'phase', 'UNKNOWN').upper()
        print(f"\n{'='*60}")
        print(f"📊 DATASET CLASS DISTRIBUTION ({phase_name})")
        print(f"{'='*60}")
        print(f"  Real videos: {real_count:5d} ({real_pct:5.2f}%)")
        print(f"  Fake videos: {fake_count:5d} ({fake_pct:5.2f}%)")
        print(f"  Total:       {total:5d}")
        print(f"  Imbalance ratio (Fake:Real): {fake_count/max(real_count, 1):.2f}:1")
        print(f"{'='*60}\n")
                
        return {'real': real_count, 'fake': fake_count}
    
    def _calculate_class_weights(self):
        """Calculate class weights for handling imbalanced data."""
        if self.class_counts['real'] == 0 or self.class_counts['fake'] == 0:
            return torch.tensor([1.0, 1.0], dtype=torch.float32)
            
        total = self.class_counts['real'] + self.class_counts['fake']
        
        # Calculate imbalance ratio
        imbalance_ratio = self.class_counts['fake'] / max(self.class_counts['real'], 1)
        
        # For manual_extreme mode, use a very high weight for the real class
        # This is for cases with extreme class imbalance
        if hasattr(self, "class_weights_mode") and self.class_weights_mode == "manual_extreme":
            print("Using manual_extreme class weights with 10:1 ratio")
            # Set real:fake weight ratio to 10:1
            weight_real = 10.0  # Real class (minority)
            weight_fake = 1.0   # Fake class (majority)
        
        # For sqrt_balanced mode, explicitly use square root balanced weights
        elif hasattr(self, "class_weights_mode") and self.class_weights_mode == "sqrt_balanced":
            weight_real = float(np.sqrt(total / (2.0 * self.class_counts['real'])))
            weight_fake = float(np.sqrt(total / (2.0 * self.class_counts['fake'])))
            print(f"Using sqrt_balanced class weights (imbalance ratio {imbalance_ratio:.2f}:1)")
        
        # For severe imbalance (ratio > 2), use stronger default weights in balanced mode
        elif imbalance_ratio > 2.0:
            # Use square root of ratio to get stronger weighting
            weight_real = float(np.sqrt(total / (2.0 * self.class_counts['real'])))
            weight_fake = float(np.sqrt(total / (2.0 * self.class_counts['fake'])))
            print(f"⚠️  Severe class imbalance detected (ratio {imbalance_ratio:.2f}:1)")
            print(f"Using enhanced class weights (sqrt-balanced)")
        
        else:
            # Standard balanced weights
            weight_real = float(total / (2.0 * self.class_counts['real']))
            weight_fake = float(total / (2.0 * self.class_counts['fake']))
            print("Using standard balanced class weights")
            
        weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32)
        print(f"📊 Final class weights: Real={weight_real:.4f}, Fake={weight_fake:.4f}")
        print(f"   (Real class weight is {weight_real/weight_fake:.2f}x higher than Fake)\n")
        
        return weights

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
            video_frames, face_embeddings, temporal_consistency, facial_landmarks = self._load_video(video_path)
            if video_frames is None:
                raise ValueError(f"Video loading failed for sample {actual_idx}. Path: {video_path}")
                
            audio_tensor, audio_spectrogram, mfcc_features = self._load_audio(audio_path)
            if audio_tensor is None:
                raise ValueError(f"Audio loading failed for sample {actual_idx}. Path: {audio_path}")

            # Load original video/audio if needed
            original_video_frames = None
            original_audio_tensor = None
            audio_visual_sync_features = None
            
            if sample.get('n_fakes', 0) > 0:
                if sample.get('modify_video', False) and original_video_path:
                    original_video_frames, _, _, _ = self._load_video(original_video_path)
                    if original_video_frames is None and self.logging:
                        print(f"⚠️ Warning: Original video loading failed for sample {actual_idx}. Path: {original_video_path}")
                        
                if sample.get('modify_audio', False) and original_audio_path:
                    original_audio_tensor, _, _ = self._load_audio(original_audio_path)
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
            
            # NEW: Extract pulse signal features if enhanced preprocessing is enabled
            pulse_signal = None
            skin_color_variations = None
            if self.enhanced_preprocessing and video_frames is not None:
                pulse_signal = self._extract_pulse_signal(video_frames)
                skin_color_variations = self._extract_skin_color_variations(video_frames)
                
            # NEW: Extract head pose features
            head_pose_features = None
            if facial_landmarks is not None:
                head_pose_features = self._estimate_head_pose(facial_landmarks)
            
            # NEW: Extract eye blinking patterns
            eye_blink_features = self._extract_eye_blink_patterns(video_frames, facial_landmarks)
            
            # NEW: Extract frequency domain features
            frequency_features = self._extract_frequency_features(video_frames)
            
            # Label: 1 for fake, 0 for real
            label = torch.tensor(1 if sample.get('n_fakes', 0) > 0 else 0, dtype=torch.long)
            
            # Fine-grained deepfake type if available
            deepfake_type = sample.get('deepfake_type', 'unknown')
            deepfake_type_id = self._get_deepfake_type_id(deepfake_type)

            # Debugging: Log successful data loading with feature details
            if self.logging:
                # Increment sample counter
                self._sample_count = getattr(self, '_sample_count', 0) + 1
                
                # Build concise feature summary
                features_extracted = []
                if pulse_signal is not None:
                    features_extracted.append("pulse")
                if skin_color_variations is not None:
                    features_extracted.append("skin")
                if head_pose_features is not None:
                    features_extracted.append("pose")
                if eye_blink_features is not None:
                    features_extracted.append("blink")
                if frequency_features is not None:
                    features_extracted.append("freq")
                if facial_landmarks is not None and len(facial_landmarks) > 0:
                    features_extracted.append("landmarks")
                
                # Concise one-line output
                features_str = "+".join(features_extracted) if features_extracted else "basic"
                print(f"✅ Sample {actual_idx}: {features_str}")

            result = {
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
                'file_path': video_path,  # For explainability and error analysis
                'facial_landmarks': facial_landmarks,  # NEW
                'mfcc_features': mfcc_features,  # NEW
                'pulse_signal': pulse_signal,  # NEW
                'skin_color_variations': skin_color_variations,  # NEW
                'head_pose': head_pose_features,  # NEW
                'eye_blink_features': eye_blink_features,  # NEW
                'frequency_features': frequency_features  # NEW
            }

            return result

        except Exception as e:
            if self.logging:
                print(f"❌ Error in __getitem__ for index {actual_idx}: {e}")
                import traceback
                traceback.print_exc()
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
        # Create a blank tensor with appropriate dimensions (reduced for memory efficiency)
        video_frames = torch.zeros((self.max_frames, 3, 224, 224))
        audio_tensor = torch.zeros(self.audio_length)
        audio_spectrogram = torch.zeros((1, 64, 64))  # Reduced from 128x128
        label = torch.tensor(0, dtype=torch.long)  # Assume real by default
        facial_landmarks = torch.zeros((self.max_frames, 136))  # 68 landmarks with x,y coordinates
        
        # Additional placeholder features (reduced sizes)
        mfcc_features = torch.zeros((20, 50))  # Reduced from (40, 100)
        pulse_signal = torch.zeros(self.max_frames)
        skin_color_variations = torch.zeros((self.max_frames, 3))
        head_pose_features = torch.zeros((self.max_frames, 3))  # pitch, yaw, roll
        eye_blink_features = torch.zeros(self.max_frames)
        frequency_features = torch.zeros((1, 16, 16))  # Reduced from (1, 32, 32)
        
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
            'face_embeddings': torch.zeros((1, 256)),  # Reduced from 512
            'temporal_consistency': torch.tensor(1.0),
            'metadata_features': torch.zeros(10),
            'ela_features': torch.zeros((112, 112)),  # Reduced from (224, 224)
            'audio_visual_sync': torch.zeros(5),
            'file_path': 'placeholder',
            'facial_landmarks': facial_landmarks,
            'mfcc_features': mfcc_features,
            'pulse_signal': pulse_signal,
            'skin_color_variations': skin_color_variations,
            'head_pose': head_pose_features,
            'eye_blink_features': eye_blink_features,
            'frequency_features': frequency_features
        }

    def _load_video(self, path):
        if not path or not os.path.exists(path):
            if self.logging:
                warnings.warn(f"⚠️ Video file not found: {path}")
            return None, None, None, None

        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                if self.logging:
                    warnings.warn(f"⚠️ Failed to open video: {path}")
                return None, None, None, None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Only show detailed info every 100 samples
            show_details = self.logging and (hasattr(self, '_sample_count') and self._sample_count % 100 == 0)
            
            if show_details:
                print(f"📹 Loading video: {os.path.basename(path)}")
                print(f"   📐 Dimensions: {width}x{height}, {total_frames} frames @ {fps:.1f} FPS")
            
            if total_frames <= 0:
                if self.logging:
                    warnings.warn(f"⚠️ Video has no frames: {path}")
                return None, None, None, None
                
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
            
            if show_details:
                print(f"   🎯 Sampling {len(frame_indices)} frames: {frame_indices[:5]}{'...' if len(frame_indices) > 5 else ''}")
                
            video_frames = []
            face_crops = []
            all_landmarks = []
            prev_face_locs = None

            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    if self.logging:
                        warnings.warn(f"⚠️ Failed to read frame {frame_idx} from {path}")
                    continue
                
                # Validate frame before processing
                if frame is None or frame.size == 0:
                    if self.logging:
                        warnings.warn(f"⚠️ Empty frame at index {frame_idx}")
                    continue
                
                original_shape = frame.shape
                
                # Convert to RGB with error handling
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Validate converted frame
                    if frame_rgb is None or frame_rgb.size == 0 or len(frame_rgb.shape) != 3 or frame_rgb.shape[2] != 3:
                        if self.logging:
                            warnings.warn(f"⚠️ Invalid RGB conversion for frame {frame_idx}")
                        continue
                        
                except Exception as e:
                    if self.logging:
                        warnings.warn(f"⚠️ Color conversion error for frame {frame_idx}: {e}")
                    continue
                
                # Apply face detection if enabled and not too many errors
                face_crop = None
                face_detected = False
                consistency_score = 1.0  # Default - perfect consistency
                landmarks = []
                
                # Skip face detection if too many errors have occurred
                if self.detect_faces and self.face_detection_error_count < 50:
                    try:
                        # Skip face detection if frame has invalid dimensions or values
                        if frame_rgb.shape[2] != 3 or np.isnan(frame_rgb).any() or frame_rgb.size == 0:
                            raise ValueError("Invalid frame format - skipping face detection")
                        
                        # Convert to 8-bit RGB with robust error handling
                        try:
                            # Ensure frame is properly formatted before conversion
                            if frame_rgb.dtype != np.uint8:
                                # Check if values are in 0-1 range
                                if frame_rgb.max() <= 1.0 and frame_rgb.min() >= 0.0:
                                    frame_rgb_8bit = (frame_rgb * 255.0).astype(np.uint8)
                                else:
                                    # Values might be in 0-255 range but wrong dtype
                                    frame_rgb_8bit = np.clip(frame_rgb, 0, 255).astype(np.uint8)
                            else:
                                frame_rgb_8bit = frame_rgb.copy()
                            
                            # Ensure values are in valid range and correct shape
                            frame_rgb_8bit = np.clip(frame_rgb_8bit, 0, 255).astype(np.uint8)
                            
                            # Verify shape and data integrity
                            if len(frame_rgb_8bit.shape) != 3 or frame_rgb_8bit.shape[2] != 3 or frame_rgb_8bit.size == 0:
                                raise ValueError(f"Invalid image shape: {frame_rgb_8bit.shape}")
                            
                            # Additional validation - check for corrupted data
                            if np.any(np.isnan(frame_rgb_8bit)) or np.any(np.isinf(frame_rgb_8bit)):
                                raise ValueError("Frame contains NaN or inf values")
                            
                            # Ensure minimum size for face detection
                            if frame_rgb_8bit.shape[0] < 20 or frame_rgb_8bit.shape[1] < 20:
                                raise ValueError("Frame too small for face detection")
                            
                            # Convert to PIL with explicit mode and additional error handling
                            pil_img = Image.fromarray(frame_rgb_8bit, mode='RGB')
                            
                            # Verify PIL image was created successfully
                            if pil_img.size[0] == 0 or pil_img.size[1] == 0:
                                raise ValueError("PIL image has zero dimensions")
                                
                        except Exception as conversion_error:
                            raise ValueError(f"Frame conversion failed: {conversion_error}")
                            
                        
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
                                
                            # Extract facial landmarks if enhanced preprocessing is enabled
                            if self.enhanced_preprocessing and hasattr(self, 'dlib_detector') and self.dlib_detector is not None:
                                try:
                                    # Ensure we have the original frame in the right format for dlib
                                    # Use the original frame_rgb before any processing
                                    dlib_frame = frame_rgb.copy()
                                    
                                    # Ensure frame is in proper format and range
                                    if dlib_frame.dtype != np.uint8:
                                        if dlib_frame.max() <= 1.0:
                                            dlib_frame = (dlib_frame * 255).astype(np.uint8)
                                        else:
                                            dlib_frame = np.clip(dlib_frame, 0, 255).astype(np.uint8)
                                    
                                    # Ensure values are in valid range
                                    dlib_frame = np.clip(dlib_frame, 0, 255).astype(np.uint8)
                                    
                                    # Convert RGB to grayscale for dlib
                                    gray = cv2.cvtColor(dlib_frame, cv2.COLOR_RGB2GRAY)
                                    
                                    # Ensure gray image is 8-bit
                                    gray = gray.astype(np.uint8)
                                    
                                    # Detect faces with dlib
                                    dlib_faces = self.dlib_detector(gray)
                                    
                                    if dlib_faces and hasattr(self, 'landmark_predictor') and self.landmark_predictor is not None:
                                        # Get facial landmarks
                                        shape = self.landmark_predictor(gray, dlib_faces[0])
                                        
                                        # Convert landmarks to list of (x, y) coordinates
                                        landmarks = []
                                        for i in range(68):
                                            x = shape.part(i).x
                                            y = shape.part(i).y
                                            landmarks.extend([x, y])  # Flatten to [x1, y1, x2, y2, ...]
                                
                                except Exception as e:
                                    if self.logging and self.face_detection_error_count < self.max_face_detection_errors_to_print:
                                        print(f"Face detection error on frame {frame_idx}: {e}")
                                        self.face_detection_error_count += 1
                                    elif self.face_detection_error_count == self.max_face_detection_errors_to_print:
                                        print("Face detection error limit reached. Suppressing further error messages.")
                                        self.face_detection_error_count += 1
                                    landmarks = []  # Reset landmarks on error
                                    
                    except Exception as e:
                        # Handle MTCNN face detection errors
                        if self.logging and self.face_detection_error_count < self.max_face_detection_errors_to_print:
                            print(f"Face detection error on frame {frame_idx}: {e}")
                            self.face_detection_error_count += 1
                        elif self.face_detection_error_count == self.max_face_detection_errors_to_print:
                            print("Face detection error limit reached. Suppressing further error messages.")
                            self.face_detection_error_count += 1
                        else:
                            # Silent increment after limit reached
                            self.face_detection_error_count += 1
                            
                        # Disable face detection if too many consecutive errors
                        if self.face_detection_error_count > 50:
                            print("⚠️ Too many face detection errors. Disabling face detection for this video.")
                            self.detect_faces = False
                
                # Extract facial landmarks even if no face was detected (for consistency)
                all_landmarks.append(landmarks if landmarks else [0] * 136)  # 68 landmarks * 2 coordinates
                
                # Ensure frame is in correct format for transforms (uint8, 0-255 range)
                if frame_rgb.dtype != np.uint8:
                    if frame_rgb.max() <= 1.0:
                        frame_rgb = (frame_rgb * 255).astype(np.uint8)
                    else:
                        frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
                
                # Apply transformations
                if self.transform:
                    try:
                        # Check if using Albumentation transforms (require named arguments)
                        if hasattr(self.transform, 'processors'):
                            # Albumentation Compose object - use named argument
                            # Frame must be uint8 for Albumentation
                            transformed = self.transform(image=frame_rgb)
                            frame_rgb = transformed['image']
                        else:
                            # PyTorch transforms - resize first
                            frame_rgb = cv2.resize(frame_rgb, (224, 224))
                            frame_rgb = self.transform(frame_rgb)
                        
                        # Ensure result is a tensor
                        if not isinstance(frame_rgb, torch.Tensor):
                            frame_rgb = torch.tensor(frame_rgb).float()
                            
                        # Ensure proper shape [C, H, W]
                        if len(frame_rgb.shape) == 3 and frame_rgb.shape[0] not in [1, 3]:
                            frame_rgb = frame_rgb.permute(2, 0, 1)
                            
                    except Exception as transform_error:
                        if self.logging:
                            warnings.warn(f"⚠️ Transform error on frame {frame_idx}: {transform_error}")
                        # Fallback to manual conversion
                        try:
                            if frame_rgb.shape != (224, 224, 3):
                                if self.logging:
                                    warnings.warn(f"⚠️ Unexpected frame shape in fallback {frame_rgb.shape}, skipping frame {frame_idx}")
                                continue
                            frame_rgb = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
                        except Exception as fallback_error:
                            if self.logging:
                                warnings.warn(f"⚠️ Fallback conversion failed on frame {frame_idx}: {fallback_error}")
                            continue
                else:
                    # Manual conversion without transforms
                    try:
                        # Validate frame dimensions before permute
                        if frame_rgb.shape != (224, 224, 3):
                            if self.logging:
                                warnings.warn(f"⚠️ Unexpected frame shape {frame_rgb.shape}, skipping frame {frame_idx}")
                            continue
                            
                        frame_rgb = torch.tensor(frame_rgb, dtype=torch.float32)
                        
                        # Safe permute operation
                        if len(frame_rgb.shape) == 3:
                            frame_rgb = frame_rgb.permute(2, 0, 1) / 255.0
                        else:
                            if self.logging:
                                warnings.warn(f"⚠️ Invalid frame shape for permute: {frame_rgb.shape}")
                            continue
                            
                    except Exception as manual_error:
                        if self.logging:
                            warnings.warn(f"⚠️ Manual conversion error on frame {frame_idx}: {manual_error}")
                        continue  # Skip this frame
                
                video_frames.append(frame_rgb)
            cap.release()

            if not video_frames:
                if self.logging:
                    warnings.warn(f"⚠️ No valid frames extracted from video: {path}")
                return None, None, None, None

            # Stack frames into a tensor
            video_tensor = torch.stack(video_frames)
            
            # Process face crops if available
            face_embeddings = None
            if face_crops:
                try:
                    # Stack face crops with error handling for different tensor shapes
                    face_crops_tensors = []
                    for crop in face_crops:
                        try:
                            if len(crop.shape) == 3 and crop.shape[2] == 3:
                                # Convert HWC to CHW format
                                crop_tensor = torch.tensor(crop).permute(2, 0, 1).float() / 255.0
                            elif len(crop.shape) == 3 and crop.shape[0] == 3:
                                # Already in CHW format
                                crop_tensor = torch.tensor(crop).float() / 255.0
                            else:
                                # Handle unexpected formats
                                crop_tensor = torch.tensor(crop).float()
                                if crop_tensor.max() > 1.0:
                                    crop_tensor = crop_tensor / 255.0
                                # Ensure it has 3 channels and proper shape
                                if len(crop_tensor.shape) == 2:
                                    crop_tensor = crop_tensor.unsqueeze(0).repeat(3, 1, 1)
                                elif len(crop_tensor.shape) == 3 and crop_tensor.shape[0] == 1:
                                    crop_tensor = crop_tensor.repeat(3, 1, 1)
                                elif len(crop_tensor.shape) == 3 and crop_tensor.shape[0] != 3:
                                    crop_tensor = crop_tensor.permute(2, 0, 1)
                            
                            face_crops_tensors.append(crop_tensor)
                        except Exception as crop_error:
                            if self.logging:
                                warnings.warn(f"⚠️ Error processing face crop: {crop_error}")
                            continue
                    
                    if face_crops_tensors:
                        face_crops_tensor = torch.stack(face_crops_tensors)
                        
                        # Create simple face embeddings (in a real model, you would use a face recognition network here)
                        # This is just a placeholder - in practice use a pre-trained face embedding network
                        
                        # Flatten each face crop to create embeddings
                        batch_size, channels, height, width = face_crops_tensor.shape
                        flattened_crops = face_crops_tensor.view(batch_size, -1)
                        
                        # Take mean across all face crops to get a single embedding per sample
                        if batch_size > 1:
                            # Average multiple face detections to get single embedding
                            raw_embedding = torch.mean(flattened_crops, dim=0, keepdim=True)
                        else:
                            raw_embedding = flattened_crops
                        
                        # Ensure consistent dimensions: resize to fixed 256-dimensional embeddings
                        embedding_dim = raw_embedding.size(1)
                        if embedding_dim != 256:
                            # Create a simple projection to 256 dimensions
                            if embedding_dim > 256:
                                # Simple downsampling by taking every nth element
                                step = embedding_dim // 256
                                face_embeddings = raw_embedding[:, ::step][:, :256]
                            else:
                                # Pad with zeros
                                padding = torch.zeros(raw_embedding.size(0), 256 - embedding_dim)
                                face_embeddings = torch.cat([raw_embedding, padding], dim=1)
                        else:
                            face_embeddings = raw_embedding
                            
                        # Ensure we always have exactly one embedding per sample
                        if face_embeddings.size(0) != 1:
                            face_embeddings = face_embeddings[0:1]  # Take only first
                    else:
                        face_embeddings = torch.zeros((1, 256))
                except Exception as face_error:
                    if self.logging:
                        warnings.warn(f"⚠️ Error processing face embeddings: {face_error}")
                    face_embeddings = torch.zeros((1, 256))
            else:
                # No faces detected, use zeros as placeholder
                face_embeddings = torch.zeros((1, 256))  # Match placeholder size
                
            # Temporal consistency feature
            temporal_consistency = torch.tensor(consistency_score).float()
            
            # Convert facial landmarks to tensor
            facial_landmarks_tensor = torch.tensor(all_landmarks, dtype=torch.float32)

            return video_tensor, face_embeddings, temporal_consistency, facial_landmarks_tensor

        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error loading video file: {path}. Error: {e}")
            return None, None, None, None

    def _load_audio(self, path):
        if not path or not os.path.exists(path):
            if self.logging:
                warnings.warn(f"⚠️ Audio file not found: {path}")
            return None, None, None
        try:
            # Load audio with torchaudio
            audio, sample_rate = torchaudio.load(path)
            audio = audio.squeeze(0).numpy()
            
            original_length = len(audio)
            original_duration = original_length / sample_rate

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
                padding_needed = self.audio_length - len(audio)
                audio = np.pad(audio, (0, padding_needed), mode='constant')

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
                        n_mels=64,  # Reduced from 128
                        hop_length=512,
                        n_fft=2048
                    )
                    
                    # Convert to dB scale
                    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    # Resize to 64x64 (reduced from 128x128)
                    mel_spec = cv2.resize(mel_spec, (64, 64))
                    
                    # Normalize
                    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
                    
                    audio_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
                except Exception as e:
                    if self.logging:
                        warnings.warn(f"⚠️ Error computing spectrogram: {e}")
                    audio_spec = torch.zeros((1, 64, 64), dtype=torch.float32)  # Updated to match new size
            else:
                audio_spec = torch.zeros((1, 64, 64), dtype=torch.float32)  # Updated to match new size
            
            # NEW: Extract MFCC features
            mfcc_features = None
            try:
                # Extract MFCCs
                mfccs = librosa.feature.mfcc(
                    y=audio, 
                    sr=sample_rate, 
                    n_mfcc=20,  # Reduced from 40
                    hop_length=512,
                    n_fft=2048
                )
                # Normalize
                mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
                mfcc_features = torch.tensor(mfccs, dtype=torch.float32)
            except Exception as e:
                if self.logging:
                    warnings.warn(f"⚠️ Error computing MFCC features: {e}")
                mfcc_features = torch.zeros((20, 50), dtype=torch.float32)  # Reduced default shape

            return torch.tensor(audio, dtype=torch.float32), audio_spec, mfcc_features
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error loading audio file: {path}. Error: {e}")
            return None, None, None
            
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
                return torch.zeros((112, 112), dtype=torch.float32)
                
            # Use first frame for ELA
            first_frame = video_frames[0]
            
            # Ensure frame has correct dimensions for permute
            if len(first_frame.shape) == 3 and first_frame.shape[0] == 3:
                first_frame = first_frame.permute(1, 2, 0).cpu().numpy()
            else:
                # Handle unexpected frame shapes - convert to expected format
                first_frame = first_frame.cpu().numpy()
                if len(first_frame.shape) == 2:
                    # Grayscale frame, convert to RGB
                    first_frame = np.stack([first_frame, first_frame, first_frame], axis=2)
                elif len(first_frame.shape) == 3 and first_frame.shape[2] == 3:
                    # Already in HWC format
                    pass
                else:
                    # Unexpected format, return zeros
                    return torch.zeros((112, 112), dtype=torch.float32)
            
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
            
            # Resize to 112x112 (reduced from 224x224 for memory efficiency)
            ela_resized = cv2.resize(ela_gray, (112, 112))
            
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
            return torch.zeros((112, 112), dtype=torch.float32)
    
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
    
    # NEW METHODS FOR ENHANCED FEATURES
    
    def _extract_pulse_signal(self, video_frames):
        """Extract subtle color variations to detect pulse signal (rPPG)."""
        try:
            if video_frames is None or len(video_frames) < 2:
                return torch.zeros(self.max_frames, dtype=torch.float32)
            
            # Extract green channel from frames (most sensitive to blood flow changes)
            green_values = []
            
            for i in range(len(video_frames)):
                frame = video_frames[i]
                
                # Ensure frame has correct dimensions for permute
                if len(frame.shape) == 3 and frame.shape[0] == 3:
                    frame = frame.permute(1, 2, 0).cpu().numpy()
                elif len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Already in HWC format
                    frame = frame.cpu().numpy()
                else:
                    # Handle unexpected frame shapes
                    green_values.append(0.5)  # Default value
                    continue
                
                # Simple skin detection (very basic)
                r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
                skin_mask = (r > 0.4) & (g > 0.2) & (b > 0.2) & (r > g) & (r > b)
                
                # If skin detected, get mean green value of skin region
                if np.any(skin_mask):
                    green_mean = np.mean(g[skin_mask])
                else:
                    green_mean = np.mean(g)  # Fallback to full frame
                
                green_values.append(green_mean)
            
            # Convert to numpy array
            signal = np.array(green_values)
            
            # Simple signal processing: bandpass filter for heart rate range (0.7-4Hz, approx 40-240 BPM)
            if len(signal) > 30:  # Need enough points for filtering (increased minimum)
                try:
                    # Estimated frame rate: assume 30fps for simplicity
                    fps = 30
                    
                    # Design bandpass filter
                    nyquist = fps / 2
                    low = 0.7 / nyquist
                    high = 4.0 / nyquist
                    b, a = scipy.signal.butter(3, [low, high], btype='band')
                    
                    # Apply filter with padlen adjustment
                    padlen = min(len(signal) // 4, 10)  # Adaptive padlen
                    filtered_signal = scipy.signal.filtfilt(b, a, signal, padlen=padlen)
                    
                    # Normalize
                    filtered_signal = (filtered_signal - np.mean(filtered_signal)) / (np.std(filtered_signal) + 1e-8)
                except Exception as filter_error:
                    if self.logging:
                        warnings.warn(f"⚠️ Filtering failed, using raw signal: {filter_error}")
                    filtered_signal = signal
            else:
                # Not enough signal points for filtering, use raw signal
                filtered_signal = signal
            
            # Ensure correct length
            if len(filtered_signal) < self.max_frames:
                filtered_signal = np.pad(filtered_signal, 
                                        (0, self.max_frames - len(filtered_signal)), 
                                        mode='constant')
            else:
                filtered_signal = filtered_signal[:self.max_frames]
            
            return torch.tensor(filtered_signal, dtype=torch.float32)
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting pulse signal: {e}")
            return torch.zeros(self.max_frames, dtype=torch.float32)
    
    def _extract_skin_color_variations(self, video_frames):
        """Extract skin color variations over time to detect blood flow patterns."""
        try:
            if video_frames is None or len(video_frames) < 2:
                return torch.zeros((self.max_frames, 3), dtype=torch.float32)
            
            # Extract average skin color (RGB) from each frame
            skin_colors = []
            
            for i in range(len(video_frames)):
                frame = video_frames[i]
                
                # Ensure frame has correct dimensions for permute
                if len(frame.shape) == 3 and frame.shape[0] == 3:
                    frame = frame.permute(1, 2, 0).cpu().numpy()
                elif len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Already in HWC format
                    frame = frame.cpu().numpy()
                else:
                    # Handle unexpected frame shapes
                    skin_colors.append([0.5, 0.5, 0.5])  # Default values
                    continue
                
                # Simple skin detection
                r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
                skin_mask = (r > 0.4) & (g > 0.2) & (b > 0.2) & (r > g) & (r > b)
                
                # If skin detected, get mean RGB values of skin region
                if np.any(skin_mask):
                    r_mean = np.mean(r[skin_mask])
                    g_mean = np.mean(g[skin_mask])
                    b_mean = np.mean(b[skin_mask])
                else:
                    r_mean = np.mean(r)  # Fallback to full frame
                    g_mean = np.mean(g)
                    b_mean = np.mean(b)
                
                skin_colors.append([r_mean, g_mean, b_mean])
            
            # Convert to numpy array
            skin_colors = np.array(skin_colors)
            
            # Ensure correct length
            if len(skin_colors) < self.max_frames:
                skin_colors = np.pad(skin_colors, 
                                    ((0, self.max_frames - len(skin_colors)), (0, 0)), 
                                    mode='constant')
            else:
                skin_colors = skin_colors[:self.max_frames]
            
            return torch.tensor(skin_colors, dtype=torch.float32)
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting skin color variations: {e}")
            return torch.zeros((self.max_frames, 3), dtype=torch.float32)
    
    def _estimate_head_pose(self, facial_landmarks):
        """Estimate head pose (pitch, yaw, roll) from facial landmarks."""
        try:
            if facial_landmarks is None or facial_landmarks.size(0) == 0:
                return torch.zeros((self.max_frames, 3), dtype=torch.float32)
            
            # Simple head pose estimation from 2D landmarks
            # This is a very simplified approach - more sophisticated methods exist
            
            # Get number of frames
            num_frames = facial_landmarks.size(0)
            
            # Placeholder for head pose estimates
            head_poses = []
            
            for i in range(num_frames):
                landmarks = facial_landmarks[i]
                
                # Check if landmarks are valid (non-zero)
                if torch.sum(landmarks) < 1e-6:
                    head_poses.append([0, 0, 0])  # Default pose
                    continue
                
                # Extract key landmarks for pose estimation
                # These indices are based on the 68-point facial landmark model
                # Left eye center
                left_eye_x = (landmarks[2*36] + landmarks[2*39]) / 2
                left_eye_y = (landmarks[2*36+1] + landmarks[2*39+1]) / 2
                
                # Right eye center
                right_eye_x = (landmarks[2*42] + landmarks[2*45]) / 2
                right_eye_y = (landmarks[2*42+1] + landmarks[2*45+1]) / 2
                
                # Nose tip
                nose_x = landmarks[2*30]
                nose_y = landmarks[2*30+1]
                
                # Mouth center
                mouth_x = (landmarks[2*48] + landmarks[2*54]) / 2
                mouth_y = (landmarks[2*48+1] + landmarks[2*54+1]) / 2
                
                # Calculate simplified pose estimates
                # Yaw: horizontal head rotation (left-right)
                eye_diff_x = left_eye_x - right_eye_x
                yaw = eye_diff_x.item() if abs(eye_diff_x.item()) < 100 else 0
                
                # Pitch: vertical head rotation (up-down)
                eyes_center_y = (left_eye_y + right_eye_y) / 2
                nose_mouth_diff_y = nose_y - mouth_y
                pitch = (eyes_center_y - nose_y).item() if abs(eyes_center_y - nose_y) < 100 else 0
                
                # Roll: tilting of the head
                eye_diff_y = left_eye_y - right_eye_y
                roll = eye_diff_y.item() if abs(eye_diff_y.item()) < 100 else 0
                
                # Normalize and scale values to reasonable range
                yaw = yaw / 50.0
                pitch = pitch / 30.0
                roll = roll / 20.0
                
                head_poses.append([pitch, yaw, roll])
            
            # Convert to tensor
            head_poses = np.array(head_poses)
            
            # Ensure correct length
            if len(head_poses) < self.max_frames:
                head_poses = np.pad(head_poses, 
                                    ((0, self.max_frames - len(head_poses)), (0, 0)), 
                                    mode='constant')
            else:
                head_poses = head_poses[:self.max_frames]
            
            return torch.tensor(head_poses, dtype=torch.float32)
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error estimating head pose: {e}")
            return torch.zeros((self.max_frames, 3), dtype=torch.float32)
    
    def _extract_eye_blink_patterns(self, video_frames, facial_landmarks):
        """Extract eye blinking patterns from facial landmarks or direct frame analysis."""
        try:
            if video_frames is None or len(video_frames) < 2:
                return torch.zeros(self.max_frames, dtype=torch.float32)
            
            num_frames = len(video_frames)
            blink_scores = []
            
            if facial_landmarks is not None and facial_landmarks.size(0) > 0:
                # Extract eye aspect ratio from landmarks
                for i in range(min(num_frames, facial_landmarks.size(0))):
                    landmarks = facial_landmarks[i]
                    
                    # Check if landmarks are valid
                    if torch.sum(landmarks) < 1e-6:
                        blink_scores.append(0.5)  # Default value (undefined)
                        continue
                    
                    try:
                        # Extract eye landmarks (68-point model)
                        # Left eye: points 36-41, Right eye: points 42-47
                        left_eye_pts = []
                        right_eye_pts = []
                        
                        # Left eye landmarks - ensure indices are valid
                        for j in range(36, 42):
                            x_idx = j * 2
                            y_idx = j * 2 + 1
                            if x_idx < len(landmarks) and y_idx < len(landmarks):
                                left_eye_pts.append([landmarks[x_idx], landmarks[y_idx]])
                        
                        # Right eye landmarks - ensure indices are valid
                        for j in range(42, 48):
                            x_idx = j * 2
                            y_idx = j * 2 + 1
                            if x_idx < len(landmarks) and y_idx < len(landmarks):
                                right_eye_pts.append([landmarks[x_idx], landmarks[y_idx]])
                        
                        # Calculate eye aspect ratio if we have enough points
                        if len(left_eye_pts) >= 6 and len(right_eye_pts) >= 6:
                            def eye_aspect_ratio(eye):
                                # Ensure we have valid points
                                if len(eye) < 6:
                                    return torch.tensor(0.3)  # Default EAR value
                                
                                try:
                                    # Compute vertical distances
                                    v1 = torch.sqrt((eye[1][0] - eye[5][0])**2 + (eye[1][1] - eye[5][1])**2)
                                    v2 = torch.sqrt((eye[2][0] - eye[4][0])**2 + (eye[2][1] - eye[4][1])**2)
                                    
                                    # Compute horizontal distance
                                    h = torch.sqrt((eye[0][0] - eye[3][0])**2 + (eye[0][1] - eye[3][1])**2)
                                    
                                    # Return ratio
                                    return (v1 + v2) / (2.0 * h + 1e-6)
                                except Exception:
                                    return torch.tensor(0.3)  # Default EAR value
                            
                            left_ear = eye_aspect_ratio(left_eye_pts)
                            right_ear = eye_aspect_ratio(right_eye_pts)
                            
                            # Average EAR
                            ear = (left_ear + right_ear) / 2.0
                            
                            # Convert to blink score (lower EAR = more closed eyes)
                            # Typical threshold for blink detection is around 0.2
                            blink_score = 1.0 - min(1.0, max(0.0, ear * 3))  # Scale and invert
                            blink_scores.append(float(blink_score) if isinstance(blink_score, torch.Tensor) else blink_score)
                        else:
                            blink_scores.append(0.5)  # Default value if not enough landmarks
                            
                    except Exception as landmark_error:
                        blink_scores.append(0.5)  # Default value on error
            else:
                # Fallback to simpler detection directly from frames
                for i in range(num_frames):
                    try:
                        frame = video_frames[i]
                        # Ensure frame has correct dimensions for permute
                        if len(frame.shape) == 3 and frame.shape[0] == 3:
                            frame = frame.permute(1, 2, 0).cpu().numpy()
                        else:
                            # Handle unexpected frame shapes
                            blink_scores.append(0.5)  # Default value
                            continue
                        
                        # Very basic eye detection and scoring (placeholder)
                        # In a real implementation, this would be more sophisticated
                        blink_scores.append(0.5)  # Default (undefined)
                    except Exception as frame_error:
                        blink_scores.append(0.5)  # Default value on error
            
            # Ensure correct length
            if len(blink_scores) < self.max_frames:
                blink_scores = np.pad(blink_scores, (0, self.max_frames - len(blink_scores)), mode='constant')
            else:
                blink_scores = blink_scores[:self.max_frames]
            
            return torch.tensor(blink_scores, dtype=torch.float32)
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting eye blink patterns: {e}")
            return torch.zeros(self.max_frames, dtype=torch.float32)
    
    def _extract_frequency_features(self, video_frames):
        """Extract frequency domain features to detect artifacts from generative models."""
        try:
            if video_frames is None or len(video_frames) == 0:
                return torch.zeros((1, 16, 16), dtype=torch.float32)
            
            # Use first frame for frequency analysis
            first_frame = video_frames[0].cpu()
            
            # Convert to grayscale if it's RGB
            if first_frame.shape[0] == 3:
                gray_frame = 0.299 * first_frame[0] + 0.587 * first_frame[1] + 0.114 * first_frame[2]
            else:
                gray_frame = first_frame[0]
            
            # Apply 2D FFT
            freq_domain = torch.fft.fft2(gray_frame)
            
            # Shift zero frequency to center
            freq_domain_shifted = torch.fft.fftshift(freq_domain)
            
            # Get magnitude spectrum (log scale for better visualization)
            magnitude_spectrum = torch.log(torch.abs(freq_domain_shifted) + 1e-10)
            
            # Resize to fixed dimensions for consistent processing
            magnitude_spectrum = F.interpolate(
                magnitude_spectrum.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                size=(16, 16),  # Reduced from (32, 32)
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dim, keep channel dim
            
            # Normalize
            magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-8)
            
            return magnitude_spectrum
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting frequency features: {e}")
            return torch.zeros((1, 16, 16), dtype=torch.float32)


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


def get_transforms_enhanced(phase='train'):
    """Enhanced transforms with more diverse augmentations from improved_augmentation.py."""
    try:
        # Try to use the advanced augmentation techniques from improved_augmentation.py
        if phase == 'train':
            # Use our advanced video transforms with temporal consistency
            video_transform = get_advanced_video_transforms(train=True)
            
            # Use our advanced audio transforms 
            audio_transform = get_advanced_audio_transforms(train=True)
            
            # Add temporal consistency augmenter if available
            if 'TemporalConsistencyAugmenter' in globals():
                print("✅ Using Temporal Consistency Augmenter")
                # This will be applied at the batch level in the training loop
        else:
            # Validation/test transforms - use standard transforms for consistency
            video_transform = get_advanced_video_transforms(train=False)
            audio_transform = get_advanced_audio_transforms(train=False)
            
    except (NameError, ImportError) as e:
        # Fallback to standard enhanced transforms if advanced ones aren't available
        print(f"⚠️ Falling back to standard enhanced transforms: {e}")
        if phase == 'train':
            # Advanced training transforms with augmentation
            video_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            audio_transform = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.02, p=0.6),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.6),
                TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
                Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
            ])
        else:
            # Validation/test transforms (same as before for consistency)
            video_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            audio_transform = None
        
    return video_transform, audio_transform


def get_data_loaders(
    json_path, data_dir, batch_size=4, validation_split=0.2, test_split=0.1,
    shuffle=True, num_workers=2, max_samples=None, detect_faces=True,
    compute_spectrograms=True, temporal_features=True, enhanced_preprocessing=True,
    enhanced_augmentation=False, multiprocessing_context=None, 
    use_mixup=True, mixup_alpha=0.2, cutmix_prob=0.3
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
        enhanced_preprocessing (bool): Whether to enable enhanced preprocessing features.
        enhanced_augmentation (bool): Whether to use enhanced data augmentation techniques.
        multiprocessing_context (str, optional): Multiprocessing context method ('spawn', 'fork', etc.) for safety.
        use_mixup (bool): Whether to enable MixUp data augmentation (when enhanced_augmentation is True).
        mixup_alpha (float): Alpha parameter for MixUp augmentation.
        cutmix_prob (float): Probability of applying CutMix instead of MixUp.
    
    Returns:
        tuple: Training, validation, and test data loaders, plus class weights and optionally a mixup function.
    """
    # Store if we're using MixUp
    using_mixup = enhanced_augmentation and use_mixup
    mixup_fn = None
    
    # Get transforms for training and validation
    if enhanced_augmentation:
        train_video_transform, train_audio_transform = get_transforms_enhanced('train')
        val_video_transform, val_audio_transform = get_transforms_enhanced('val')
        
        # Set up MixUp/CutMix if we're using advanced augmentations
        if using_mixup:
            try:
                # Create MixUp augmentation function from improved_augmentation.py
                mixup_fn = mix_up_augmentation
                print("✅ Using MixUp/CutMix augmentation from improved_augmentation.py")
            except (NameError, AttributeError) as e:
                print(f"⚠️ MixUp augmentation not available: {e}")
                using_mixup = False
    else:
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
        temporal_features=temporal_features,
        enhanced_preprocessing=enhanced_preprocessing
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
        temporal_features=temporal_features,
        enhanced_preprocessing=enhanced_preprocessing
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
        temporal_features=temporal_features,
        enhanced_preprocessing=enhanced_preprocessing
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

    # Handle multiprocessing context for safety
    mp_context = None
    if num_workers > 0 and multiprocessing_context:
        import multiprocessing as mp
        try:
            mp_context = mp.get_context(multiprocessing_context)
        except Exception as e:
            print(f"⚠️ Warning: Could not set multiprocessing context '{multiprocessing_context}': {e}")
            mp_context = None

    # Create data loaders
    train_loader_kwargs = {
        'dataset': train_dataset,
        'batch_size': batch_size,
        'sampler': train_sampler,
        'num_workers': num_workers,
        'pin_memory': False,  # Disabled for Windows compatibility (causes "invalid device pointer" crash)
        'drop_last': False,
        'collate_fn': collate_fn,
        'persistent_workers': False,  # Disabled for Windows compatibility (num_workers > 0 can cause hangs)
    }
    if mp_context is not None:
        train_loader_kwargs['multiprocessing_context'] = mp_context
    
    train_loader = DataLoader(**train_loader_kwargs)
    
    val_loader_kwargs = {
        'dataset': val_dataset,
        'batch_size': batch_size,
        'sampler': val_sampler,
        'num_workers': num_workers,
        'pin_memory': False,  # Disabled for Windows compatibility (causes "invalid device pointer" crash)
        'drop_last': False,
        'collate_fn': collate_fn,
        'persistent_workers': False,  # Disabled for Windows compatibility (num_workers > 0 can cause hangs)
    }
    if mp_context is not None:
        val_loader_kwargs['multiprocessing_context'] = mp_context
    
    val_loader = DataLoader(**val_loader_kwargs)
    
    test_loader_kwargs = {
        'dataset': test_dataset,
        'batch_size': batch_size,
        'sampler': test_sampler,
        'num_workers': num_workers,
        'pin_memory': False,  # Disabled for Windows compatibility (causes "invalid device pointer" crash)
        'drop_last': False,
        'collate_fn': collate_fn,
        'persistent_workers': False,  # Disabled for Windows compatibility (num_workers > 0 can cause hangs)
    }
    if mp_context is not None:
        test_loader_kwargs['multiprocessing_context'] = mp_context
    
    test_loader = DataLoader(**test_loader_kwargs)

    print(f"✅ Dataset loaded with {len(train_indices)} training, {len(val_indices)} validation, and {len(test_indices)} test samples.")
    
    # Return mixup function if using enhanced augmentation
    if enhanced_augmentation and 'mixup_fn' in locals() and mixup_fn is not None:
        return train_loader, val_loader, test_loader, class_weights, mixup_fn
    else:
        return train_loader, val_loader, test_loader, class_weights


def collate_fn(batch):
    """
    Custom collate function that handles variable-sized data and ensures consistent batch dimensions.
    """
    result = {}
    
    # Process all possible keys that might be in batch
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())
    
    # Get the actual batch size
    actual_batch_size = len(batch)
    
    for key in all_keys:
        values = [item.get(key) for item in batch]
        
        # Filter out None values but keep track of positions
        valid_values = []
        valid_indices = []
        for i, v in enumerate(values):
            if v is not None:
                valid_values.append(v)
                valid_indices.append(i)
        
        if key in ['video_frames', 'audio', 'audio_spectrogram', 'facial_landmarks', 'mfcc_features', 'pulse_signal', 'skin_color_variations', 'head_pose', 'eye_blink_features', 'frequency_features']:
            # These are critical tensors that must have consistent batch dimensions
            if valid_values and all(isinstance(v, torch.Tensor) for v in valid_values):
                try:
                    # If we have the full batch, just stack
                    if len(valid_values) == actual_batch_size:
                        # Check if all tensors have the same shape
                        ref_shape = valid_values[0].shape
                        if all(v.shape == ref_shape for v in valid_values):
                            # Clone tensors to avoid memory sharing issues
                            cloned_tensors = [v.clone() for v in valid_values]
                            result[key] = torch.stack(cloned_tensors)
                        else:
                            # Handle different shapes by padding to max dimensions
                            max_shape = list(ref_shape)
                            for v in valid_values[1:]:
                                for j in range(len(v.shape)):
                                    max_shape[j] = max(max_shape[j], v.shape[j])
                            
                            # Pad all tensors to max shape
                            padded_tensors = []
                            for v in valid_values:
                                if list(v.shape) == max_shape:
                                    padded_tensors.append(v.clone())
                                else:
                                    # Create padding specification
                                    pad_spec = []
                                    for j in range(len(v.shape)):
                                        pad_amount = max_shape[j] - v.shape[j]
                                        pad_spec = [0, pad_amount] + pad_spec  # PyTorch padding is reversed
                                    
                                    padded = torch.nn.functional.pad(v, pad_spec)
                                    padded_tensors.append(padded.clone())
                            
                            result[key] = torch.stack(padded_tensors)
                    else:
                        # Missing some values, create full batch tensor with zeros
                        ref_shape = valid_values[0].shape
                        full_batch_tensor = torch.zeros(actual_batch_size, *ref_shape, dtype=valid_values[0].dtype)
                        
                        # Fill in the valid values at their correct positions
                        for i, valid_idx in enumerate(valid_indices):
                            if valid_idx < actual_batch_size:
                                full_batch_tensor[valid_idx] = valid_values[i].clone()
                        
                        result[key] = full_batch_tensor
                        
                except Exception as e:
                    print(f"[ERROR] Failed to collate {key}: {e}")
                    # Fallback: create empty tensor with correct batch size
                    if key == 'video_frames':
                        result[key] = torch.zeros(actual_batch_size, 16, 3, 224, 224)
                    elif key == 'audio':
                        result[key] = torch.zeros(actual_batch_size, 8000)
                    elif key == 'audio_spectrogram':
                        result[key] = torch.zeros(actual_batch_size, 1, 64, 64)
                    elif key == 'facial_landmarks':
                        result[key] = torch.zeros(actual_batch_size, 16, 136)
                    elif key == 'mfcc_features':
                        result[key] = torch.zeros(actual_batch_size, 20, 50)
                    elif key == 'pulse_signal':
                        result[key] = torch.zeros(actual_batch_size, 16)
                    elif key == 'skin_color_variations':
                        result[key] = torch.zeros(actual_batch_size, 16, 3)
                    elif key == 'head_pose':
                        result[key] = torch.zeros(actual_batch_size, 16, 3)
                    elif key == 'eye_blink_features':
                        result[key] = torch.zeros(actual_batch_size, 16)
                    elif key == 'frequency_features':
                        result[key] = torch.zeros(actual_batch_size, 1, 16, 16)
                    else:
                        result[key] = None
            else:
                # Create appropriate zero tensor for missing data
                if key == 'video_frames':
                    result[key] = torch.zeros(actual_batch_size, 16, 3, 224, 224)
                elif key == 'audio':
                    result[key] = torch.zeros(actual_batch_size, 8000)
                elif key == 'audio_spectrogram':
                    result[key] = torch.zeros(actual_batch_size, 1, 64, 64)
                elif key == 'facial_landmarks':
                    result[key] = torch.zeros(actual_batch_size, 16, 136)
                elif key == 'mfcc_features':
                    result[key] = torch.zeros(actual_batch_size, 20, 50)
                elif key == 'pulse_signal':
                    result[key] = torch.zeros(actual_batch_size, 16)
                elif key == 'skin_color_variations':
                    result[key] = torch.zeros(actual_batch_size, 16, 3)
                elif key == 'head_pose':
                    result[key] = torch.zeros(actual_batch_size, 16, 3)
                elif key == 'eye_blink_features':
                    result[key] = torch.zeros(actual_batch_size, 16)
                elif key == 'frequency_features':
                    result[key] = torch.zeros(actual_batch_size, 1, 16, 16)
                else:
                    result[key] = None
                    
        elif key in ['ela_features', 'metadata_features', 'face_embeddings', 'temporal_consistency']:
            # Optional tensor features
            if valid_values and all(isinstance(v, torch.Tensor) for v in valid_values):
                try:
                    # Stack valid tensors and pad batch if needed
                    if len(valid_values) == actual_batch_size:
                        # Check if all tensors have the same shape
                        ref_shape = valid_values[0].shape
                        if all(v.shape == ref_shape for v in valid_values):
                            result[key] = torch.stack(valid_values)
                        else:
                            # Handle different shapes by padding
                            max_shape = list(ref_shape)
                            for v in valid_values[1:]:
                                for j in range(len(v.shape)):
                                    max_shape[j] = max(max_shape[j], v.shape[j])
                            
                            # Pad all tensors to max shape
                            padded_tensors = []
                            for v in valid_values:
                                if list(v.shape) == max_shape:
                                    padded_tensors.append(v.clone())  # Clone to avoid memory sharing
                                else:
                                    # Create padding specification
                                    pad_spec = []
                                    for j in range(len(v.shape)):
                                        pad_amount = max_shape[j] - v.shape[j]
                                        pad_spec = [0, pad_amount] + pad_spec
                                    
                                    padded = torch.nn.functional.pad(v, pad_spec)
                                    padded_tensors.append(padded.clone())  # Clone to avoid memory sharing
                            
                            result[key] = torch.stack(padded_tensors)
                    else:
                        # Create tensor with zeros for missing samples
                        ref_shape = valid_values[0].shape
                        full_batch_tensor = torch.zeros(actual_batch_size, *ref_shape, dtype=valid_values[0].dtype)
                        
                        for i, valid_idx in enumerate(valid_indices):
                            if valid_idx < actual_batch_size:
                                full_batch_tensor[valid_idx] = valid_values[i].clone()  # Clone to avoid memory sharing
                        
                        result[key] = full_batch_tensor
                except Exception as e:
                    print(f"[WARNING] Failed to collate optional feature {key}: {e}")
                    result[key] = None
            else:
                result[key] = None
                
        elif key in ['original_video_frames', 'original_audio', 'audio_visual_sync']:
            # These may have different batch sizes, handle gracefully by creating zeros for missing samples
            if valid_values and all(isinstance(v, torch.Tensor) for v in valid_values):
                try:
                    # Check if we can create a consistent batch
                    ref_shape = valid_values[0].shape
                    compatible = all(v.shape[1:] == ref_shape[1:] for v in valid_values)
                    
                    if compatible and len(valid_values) == actual_batch_size:
                        result[key] = torch.stack(valid_values)
                    else:
                        # For mismatched batches, create full batch tensor with zeros for missing samples
                        if len(valid_values) > 0:
                            # Create full batch tensor filled with zeros
                            full_batch_tensor = torch.zeros(actual_batch_size, *ref_shape, dtype=valid_values[0].dtype)
                            
                            # Fill in the valid values at their correct positions
                            for i, valid_idx in enumerate(valid_indices):
                                if valid_idx < actual_batch_size and i < len(valid_values):
                                    full_batch_tensor[valid_idx] = valid_values[i].clone()
                            
                            result[key] = full_batch_tensor
                        else:
                            # Create appropriate zero tensor based on the key
                            if key == 'original_video_frames':
                                result[key] = torch.zeros(actual_batch_size, 16, 3, 224, 224)
                            elif key == 'original_audio':
                                result[key] = torch.zeros(actual_batch_size, 8000)
                            elif key == 'audio_visual_sync':
                                result[key] = torch.zeros(actual_batch_size, 5)
                            else:
                                result[key] = None
                except Exception as e:
                    # Create fallback tensors for failed collation
                    if key == 'original_video_frames':
                        result[key] = torch.zeros(actual_batch_size, 16, 3, 224, 224)
                    elif key == 'original_audio':
                        result[key] = torch.zeros(actual_batch_size, 8000)
                    elif key == 'audio_visual_sync':
                        result[key] = torch.zeros(actual_batch_size, 5)
                    else:
                        result[key] = None
            else:
                # Create appropriate zero tensor when no valid values
                if key == 'original_video_frames':
                    result[key] = torch.zeros(actual_batch_size, 16, 3, 224, 224)
                elif key == 'original_audio':
                    result[key] = torch.zeros(actual_batch_size, 8000)
                elif key == 'audio_visual_sync':
                    result[key] = torch.zeros(actual_batch_size, 5)
                else:
                    result[key] = None
                
        elif key in ['fake_periods', 'timestamps']:
            # List of lists, don't stack
            result[key] = [item.get(key, []) for item in batch]
        elif key in ['transcript', 'file_path']:
            # List of strings
            result[key] = [item.get(key, "") for item in batch]
        elif key == 'fake_mask':
            # Don't stack fake_masks of different sizes, keep as list
            result[key] = [item.get(key, []) for item in batch]
        elif key == 'label':
            # Labels should always be stackable
            if valid_values and all(isinstance(v, (int, torch.Tensor)) for v in valid_values):
                # Create full batch labels
                labels = torch.zeros(actual_batch_size, dtype=torch.long)
                for i, valid_idx in enumerate(valid_indices):
                    if isinstance(valid_values[i], torch.Tensor):
                        labels[valid_idx] = valid_values[i].clone()
                    else:
                        labels[valid_idx] = torch.tensor(valid_values[i], dtype=torch.long)
                result[key] = labels
            else:
                # Fallback to all zeros
                result[key] = torch.zeros(actual_batch_size, dtype=torch.long)
        else:
            # Handle other types if needed
            result[key] = [item.get(key) for item in batch]
    
    return result
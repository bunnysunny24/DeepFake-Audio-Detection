import torch.nn.functional as F
import scipy
from scipy import signal
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torchvision import transforms
import torchaudio
import cv2
import warnings
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Shift
import albumentations as A
try:
    from facenet_pytorch import MTCNN
except ImportError:
    print("Warning: facenet_pytorch not found, some face detection features will be limited")
from scipy.signal import spectrogram
import librosa
from PIL import Image
import random
import math
import scipy.ndimage as ndimage
from sklearn.utils import class_weight
import torch.nn as nn
import copy


class MultiModalDeepfakeDataset(Dataset):
    def __init__(self, json_path, data_dir, max_frames=32, audio_length=16000, transform=None, audio_transform=None, 
                 logging=False, phase='train', detect_faces=True, compute_spectrograms=True, temporal_features=True,
                 enhanced_preprocessing=True, adversarial_training=False, self_supervised_pretraining=False,
                 curriculum_learning=False, active_learning=False, domain_adaptation=False):
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
        self.phase = phase
        self.detect_faces = detect_faces
        self.compute_spectrograms = compute_spectrograms
        self.temporal_features = temporal_features
        self.enhanced_preprocessing = enhanced_preprocessing
        
        # NEW: Advanced training techniques
        self.adversarial_training = adversarial_training
        self.self_supervised_pretraining = self_supervised_pretraining
        self.curriculum_learning = curriculum_learning
        self.active_learning = active_learning
        self.domain_adaptation = domain_adaptation
        
        # Initialize curriculum learning parameters
        if self.curriculum_learning:
            self.curriculum_epoch = 0
            self.difficulty_scores = None
            self._compute_difficulty_scores()
        
        # Initialize active learning parameters
        if self.active_learning:
            self.sample_uncertainties = None
            self.selection_strategy = 'uncertainty'  # 'uncertainty', 'diversity', 'hybrid'
            
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
        
        # Initialize facial landmark detector if available
        if self.enhanced_preprocessing:
            try:
                import dlib
                self.dlib_detector = dlib.get_frontal_face_detector()
                model_path = "shape_predictor_68_face_landmarks.dat"
                if os.path.exists(model_path):
                    self.landmark_predictor = dlib.shape_predictor(model_path)
                    print("Facial landmark predictor initialized successfully")
                else:
                    print(f"Warning: Facial landmark model not found at {model_path}")
                    self.landmark_predictor = None
            except Exception as e:
                print(f"Warning: Could not initialize facial landmark detector: {e}")
                self.dlib_detector = None
                self.landmark_predictor = None
        
        # NEW: Initialize adversarial attack methods
        if self.adversarial_training:
            self.adversarial_methods = ['fgsm', 'pgd', 'gaussian_noise', 'jpeg_compression']
            self.adversarial_epsilon = 0.1
            self.adversarial_alpha = 0.01
            self.adversarial_iterations = 10
            
        # NEW: Initialize self-supervised learning components
        if self.self_supervised_pretraining:
            self.ssl_tasks = ['rotation', 'jigsaw', 'colorization', 'temporal_order']
            self.ssl_augmentations = self._get_ssl_augmentations()
            
        # NEW: Initialize domain adaptation components
        if self.domain_adaptation:
            self.domain_labels = self._assign_domain_labels()
            
        print(f"Dataset initialized with {len(self.valid_indices)} valid samples out of {len(self.data)} total.")
        print(f"Class distribution: {self.class_counts}")
        
        # NEW: Print advanced features status
        print(f"Advanced features enabled:")
        print(f"  - Adversarial training: {self.adversarial_training}")
        print(f"  - Self-supervised pretraining: {self.self_supervised_pretraining}")
        print(f"  - Curriculum learning: {self.curriculum_learning}")
        print(f"  - Active learning: {self.active_learning}")
        print(f"  - Domain adaptation: {self.domain_adaptation}")

    def _validate_dataset(self):
        """Pre-validate all samples in the dataset to identify valid ones."""
        print("Starting dataset validation...")
        
        # Use all samples for validation instead of limiting to 100
        max_to_validate = len(self.data)
        print(f"Validating all {max_to_validate} samples...")
        
        valid_indices = []
        progress_interval = max(1, max_to_validate // 100)
        
        for idx in range(max_to_validate):
            if idx % progress_interval == 0 or idx == max_to_validate - 1:
                print(f"Validating sample {idx+1}/{max_to_validate} ({(idx+1)/max_to_validate*100:.1f}%)...")
            
            sample = self.data[idx]
            video_path = os.path.join(self.data_dir, sample['file'])
            audio_path = video_path.replace('.mp4', '.wav')
            
            # Simple validation - check if files exist
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

    # NEW: Curriculum Learning Methods
    def _compute_difficulty_scores(self):
        """Compute difficulty scores for curriculum learning."""
        try:
            difficulty_scores = []
            
            for idx in self.valid_indices:
                sample = self.data[idx]
                
                # Compute difficulty based on multiple factors
                difficulty = 0.0
                
                # Factor 1: Video quality (lower quality = higher difficulty)
                video_path = os.path.join(self.data_dir, sample['file'])
                if os.path.exists(video_path):
                    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                    difficulty += max(0, 1.0 - file_size / 10.0)  # Normalize
                
                # Factor 2: Number of fake periods (more periods = higher difficulty)
                fake_periods = len(sample.get('fake_periods', []))
                difficulty += min(1.0, fake_periods / 5.0)
                
                # Factor 3: Deepfake type complexity
                deepfake_type = sample.get('deepfake_type', 'unknown')
                type_difficulty = {
                    'unknown': 0.1,
                    'audio_only': 0.3,
                    'face_swap': 0.5,
                    'lip_sync': 0.7,
                    'face_reenactment': 0.8,
                    'entire_synthesis': 0.9,
                    'attribute_manipulation': 0.6
                }
                difficulty += type_difficulty.get(deepfake_type, 0.5)
                
                # Normalize difficulty score
                difficulty = min(1.0, difficulty / 3.0)
                difficulty_scores.append(difficulty)
            
            self.difficulty_scores = np.array(difficulty_scores)
            print(f"Computed difficulty scores: mean={np.mean(self.difficulty_scores):.3f}, std={np.std(self.difficulty_scores):.3f}")
            
        except Exception as e:
            print(f"Warning: Could not compute difficulty scores: {e}")
            self.difficulty_scores = np.ones(len(self.valid_indices)) * 0.5
    
    def update_curriculum_epoch(self, epoch):
        """Update curriculum learning epoch."""
        self.curriculum_epoch = epoch
    
    def get_curriculum_samples(self, max_difficulty=None):
        """Get samples based on curriculum learning strategy."""
        if not self.curriculum_learning or self.difficulty_scores is None:
            return self.valid_indices
        
        if max_difficulty is None:
            # Progressive difficulty increase
            max_difficulty = min(1.0, 0.3 + 0.7 * (self.curriculum_epoch / 50))
        
        # Select samples within difficulty threshold
        selected_indices = []
        for i, idx in enumerate(self.valid_indices):
            if self.difficulty_scores[i] <= max_difficulty:
                selected_indices.append(idx)
        
        return selected_indices if selected_indices else self.valid_indices

    # NEW: Active Learning Methods
    def update_sample_uncertainties(self, uncertainties):
        """Update sample uncertainties for active learning."""
        self.sample_uncertainties = uncertainties
    
    def get_active_learning_samples(self, n_samples=None, strategy='uncertainty'):
        """Select samples based on active learning strategy."""
        if not self.active_learning or self.sample_uncertainties is None:
            return self.valid_indices
        
        if n_samples is None:
            n_samples = len(self.valid_indices)
        
        if strategy == 'uncertainty':
            # Select samples with highest uncertainty
            sorted_indices = np.argsort(self.sample_uncertainties)[::-1]
            return [self.valid_indices[i] for i in sorted_indices[:n_samples]]
        elif strategy == 'diversity':
            # Simple diversity-based selection (placeholder)
            return random.sample(self.valid_indices, min(n_samples, len(self.valid_indices)))
        else:
            # Hybrid strategy
            n_uncertain = n_samples // 2
            n_diverse = n_samples - n_uncertain
            
            uncertain_indices = self.get_active_learning_samples(n_uncertain, 'uncertainty')
            diverse_indices = self.get_active_learning_samples(n_diverse, 'diversity')
            
            return list(set(uncertain_indices + diverse_indices))

    # NEW: Domain Adaptation Methods
    def _assign_domain_labels(self):
        """Assign domain labels for domain adaptation."""
        try:
            domain_labels = []
            
            for idx in self.valid_indices:
                sample = self.data[idx]
                
                # Assign domain based on deepfake type or source
                deepfake_type = sample.get('deepfake_type', 'unknown')
                source = sample.get('source', 'unknown')
                
                # Simple domain assignment (can be made more sophisticated)
                if 'face' in deepfake_type:
                    domain = 0  # Face domain
                elif 'audio' in deepfake_type:
                    domain = 1  # Audio domain
                elif 'entire' in deepfake_type:
                    domain = 2  # Full synthesis domain
                else:
                    domain = 3  # Unknown domain
                
                domain_labels.append(domain)
            
            return np.array(domain_labels)
            
        except Exception as e:
            print(f"Warning: Could not assign domain labels: {e}")
            return np.zeros(len(self.valid_indices))

    # NEW: Self-Supervised Learning Methods
    def _get_ssl_augmentations(self):
        """Get self-supervised learning augmentations."""
        return {
            'rotation': transforms.RandomRotation(degrees=[0, 90, 180, 270]),
            'jigsaw': self._create_jigsaw_transform(),
            'colorization': self._create_colorization_transform(),
            'temporal_order': None  # Handled separately for video sequences
        }
    
    def _create_jigsaw_transform(self):
        """Create jigsaw puzzle transform."""
        def jigsaw_transform(img):
            # Simple jigsaw implementation
            patches = []
            h, w = img.shape[-2:]
            patch_h, patch_w = h // 3, w // 3
            
            for i in range(3):
                for j in range(3):
                    patch = img[..., i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                    patches.append(patch)
            
            # Shuffle patches
            random.shuffle(patches)
            
            # Reconstruct image
            reconstructed = torch.zeros_like(img)
            idx = 0
            for i in range(3):
                for j in range(3):
                    reconstructed[..., i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = patches[idx]
                    idx += 1
            
            return reconstructed
        
        return jigsaw_transform
    
    def _create_colorization_transform(self):
        """Create colorization transform (convert to grayscale)."""
        def colorization_transform(img):
            # Convert to grayscale
            if img.shape[0] == 3:
                gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
                return gray.unsqueeze(0).repeat(3, 1, 1)
            return img
        
        return colorization_transform

    # NEW: Adversarial Training Methods
    def _apply_adversarial_perturbation(self, data, method='fgsm'):
        """Apply adversarial perturbations to data."""
        if method == 'fgsm':
            return self._fgsm_attack(data)
        elif method == 'pgd':
            return self._pgd_attack(data)
        elif method == 'gaussian_noise':
            return self._gaussian_noise_attack(data)
        elif method == 'jpeg_compression':
            return self._jpeg_compression_attack(data)
        else:
            return data
    
    def _fgsm_attack(self, data):
        """Fast Gradient Sign Method attack."""
        try:
            # Generate random gradient-like perturbation
            perturbation = torch.randn_like(data) * self.adversarial_epsilon
            perturbation = torch.sign(perturbation) * self.adversarial_epsilon
            
            # Apply perturbation
            adversarial_data = data + perturbation
            
            # Clamp to valid range
            adversarial_data = torch.clamp(adversarial_data, 0, 1)
            
            return adversarial_data
        except Exception as e:
            if self.logging:
                print(f"Warning: FGSM attack failed: {e}")
            return data
    
    def _pgd_attack(self, data):
        """Projected Gradient Descent attack."""
        try:
            adversarial_data = data.clone()
            
            for _ in range(self.adversarial_iterations):
                # Generate random perturbation
                perturbation = torch.randn_like(adversarial_data) * self.adversarial_alpha
                perturbation = torch.sign(perturbation) * self.adversarial_alpha
                
                # Apply perturbation
                adversarial_data = adversarial_data + perturbation
                
                # Project back to epsilon ball
                delta = adversarial_data - data
                delta = torch.clamp(delta, -self.adversarial_epsilon, self.adversarial_epsilon)
                adversarial_data = data + delta
                
                # Clamp to valid range
                adversarial_data = torch.clamp(adversarial_data, 0, 1)
            
            return adversarial_data
        except Exception as e:
            if self.logging:
                print(f"Warning: PGD attack failed: {e}")
            return data
    
    def _gaussian_noise_attack(self, data):
        """Gaussian noise attack."""
        try:
            noise = torch.randn_like(data) * self.adversarial_epsilon
            adversarial_data = data + noise
            adversarial_data = torch.clamp(adversarial_data, 0, 1)
            return adversarial_data
        except Exception as e:
            if self.logging:
                print(f"Warning: Gaussian noise attack failed: {e}")
            return data
    
    def _jpeg_compression_attack(self, data):
        """JPEG compression attack."""
        try:
            # Simple compression simulation by adding quantization noise
            compressed_data = data + torch.randn_like(data) * 0.02
            compressed_data = torch.clamp(compressed_data, 0, 1)
            return compressed_data
        except Exception as e:
            if self.logging:
                print(f"Warning: JPEG compression attack failed: {e}")
            return data

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Get a sample by its index in the valid indices list."""
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

            # NEW: Apply adversarial perturbations during training
            if self.adversarial_training and self.phase == 'train' and random.random() < 0.3:
                adv_method = random.choice(self.adversarial_methods)
                video_frames = self._apply_adversarial_perturbation(video_frames, adv_method)
                if self.logging:
                    print(f"Applied adversarial perturbation: {adv_method}")

            # NEW: Apply self-supervised learning augmentations
            ssl_labels = {}
            if self.self_supervised_pretraining and self.phase == 'train':
                ssl_labels = self._apply_ssl_augmentations(video_frames)

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

            # Extract metadata features
            metadata_features = self._extract_metadata_features(video_path)
            
            # Calculate ELA (Error Level Analysis) for forgery detection
            ela_features = self._extract_ela_features(video_frames) if video_frames is not None else None
            
            # Extract enhanced features
            pulse_signal = None
            skin_color_variations = None
            if self.enhanced_preprocessing and video_frames is not None:
                pulse_signal = self._extract_pulse_signal(video_frames)
                skin_color_variations = self._extract_skin_color_variations(video_frames)
                
            # Extract head pose features
            head_pose_features = None
            if facial_landmarks is not None:
                head_pose_features = self._estimate_head_pose(facial_landmarks)
            
            # Extract eye blinking patterns
            eye_blink_features = self._extract_eye_blink_patterns(video_frames, facial_landmarks)
            
            # Extract frequency domain features
            frequency_features = self._extract_frequency_features(video_frames)
            
            # NEW: Extract ensemble features for multi-head aggregation
            ensemble_features = self._extract_ensemble_features(video_frames, audio_tensor)
            
            # NEW: Compute uncertainty estimates
            uncertainty_estimates = self._compute_uncertainty_estimates(video_frames, audio_tensor)
            
            # Label: 1 for fake, 0 for real
            label = torch.tensor(1 if sample.get('n_fakes', 0) > 0 else 0, dtype=torch.long)
            
            # Fine-grained deepfake type
            deepfake_type = sample.get('deepfake_type', 'unknown')
            deepfake_type_id = self._get_deepfake_type_id(deepfake_type)
            
            # NEW: Domain label for domain adaptation
            domain_label = 0
            if self.domain_adaptation:
                domain_idx = self.valid_indices.index(actual_idx)
                domain_label = self.domain_labels[domain_idx]

            # Debugging: Log successful data loading
            if self.logging:
                print(f"✅ Successfully loaded sample at index {actual_idx}")

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
                'file_path': video_path,
                'facial_landmarks': facial_landmarks,
                'mfcc_features': mfcc_features,
                'pulse_signal': pulse_signal,
                'skin_color_variations': skin_color_variations,
                'head_pose': head_pose_features,
                'eye_blink_features': eye_blink_features,
                'frequency_features': frequency_features,
                # NEW: Advanced features
                'ensemble_features': ensemble_features,
                'uncertainty_estimates': uncertainty_estimates,
                'domain_label': torch.tensor(domain_label, dtype=torch.long),
                'ssl_labels': ssl_labels,
                'difficulty_score': self.difficulty_scores[idx] if self.difficulty_scores is not None else 0.5,
                'sample_weight': self._compute_sample_weight(actual_idx)
            }

            return result

        except Exception as e:
            if self.logging:
                print(f"❌ Error in __getitem__ for index {actual_idx}: {e}")
                import traceback
                traceback.print_exc()
            return self._get_placeholder_sample()

    # NEW: Advanced Feature Extraction Methods
    def _extract_ensemble_features(self, video_frames, audio_tensor):
        """Extract features for ensemble/multi-head aggregation."""
        try:
            ensemble_features = {}
            
            # Visual ensemble features
            if video_frames is not None:
                # Statistical features
                ensemble_features['visual_mean'] = torch.mean(video_frames, dim=[1, 2, 3])
                ensemble_features['visual_std'] = torch.std(video_frames, dim=[1, 2, 3])
                ensemble_features['visual_max'] = torch.max(video_frames.flatten(1), dim=1)[0]
                ensemble_features['visual_min'] = torch.min(video_frames.flatten(1), dim=1)[0]
                
                # Temporal features
                if video_frames.shape[0] > 1:
                    frame_diffs = torch.abs(video_frames[1:] - video_frames[:-1])
                    ensemble_features['temporal_variation'] = torch.mean(frame_diffs, dim=[1, 2, 3])
                else:
                    ensemble_features['temporal_variation'] = torch.zeros(1)
            
            # Audio ensemble features
            if audio_tensor is not None:
                # Spectral features
                ensemble_features['audio_energy'] = torch.mean(audio_tensor**2)
                ensemble_features['audio_zero_crossing_rate'] = self._compute_zero_crossing_rate(audio_tensor)
                ensemble_features['audio_spectral_centroid'] = self._compute_spectral_centroid(audio_tensor)
            
            return ensemble_features
            
        except Exception as e:
            if self.logging:
                print(f"Warning: Could not extract ensemble features: {e}")
            return {}
    
    def _compute_uncertainty_estimates(self, video_frames, audio_tensor):
        """Compute uncertainty estimates for active learning."""
        try:
            uncertainty_estimates = {}
            
            # Visual uncertainty (based on variance)
            if video_frames is not None:
                frame_var = torch.var(video_frames, dim=[1, 2, 3])
                uncertainty_estimates['visual_uncertainty'] = torch.mean(frame_var)
            
            # Audio uncertainty
            if audio_tensor is not None:
                audio_var = torch.var(audio_tensor)
                uncertainty_estimates['audio_uncertainty'] = audio_var
            
            # Combined uncertainty
            uncertainties = list(uncertainty_estimates.values())
            if uncertainties:
                uncertainty_estimates['combined_uncertainty'] = torch.mean(torch.stack(uncertainties))
            else:
                uncertainty_estimates['combined_uncertainty'] = torch.tensor(0.5)
            
            return uncertainty_estimates
            
        except Exception as e:
            if self.logging:
                print(f"Warning: Could not compute uncertainty estimates: {e}")
            return {'combined_uncertainty': torch.tensor(0.5)}
    
    def _compute_zero_crossing_rate(self, audio_tensor):
        """Compute zero crossing rate for audio."""
        try:
            signs = torch.sign(audio_tensor)
            sign_changes = torch.abs(signs[1:] - signs[:-1])
            zcr = torch.sum(sign_changes) / (2 * len(audio_tensor))
            return zcr
        except Exception as e:
            return torch.tensor(0.0)
    
    def _compute_spectral_centroid(self, audio_tensor):
        """Compute spectral centroid for audio."""
        try:
            # Simple spectral centroid approximation
            fft = torch.fft.fft(audio_tensor)
            magnitude = torch.abs(fft)
            freqs = torch.arange(len(magnitude)).float()
            spectral_centroid = torch.sum(freqs * magnitude) / torch.sum(magnitude)
            return spectral_centroid / len(magnitude)  # Normalize
        except Exception as e:
            return torch.tensor(0.0)
    
    def _apply_ssl_augmentations(self, video_frames):
        """Apply self-supervised learning augmentations."""
        ssl_labels = {}
        
        try:
            if random.random() < 0.5:  # 50% chance to apply SSL augmentations
                ssl_task = random.choice(self.ssl_tasks)
                
                if ssl_task == 'rotation':
                    # Rotation prediction task
                    rotation_angle = random.choice([0, 90, 180, 270])
                    ssl_labels['rotation_label'] = torch.tensor(rotation_angle // 90, dtype=torch.long)
                    
                elif ssl_task == 'jigsaw':
                    # Jigsaw puzzle task
                    ssl_labels['jigsaw_label'] = torch.tensor(random.randint(0, 8), dtype=torch.long)
                    
                elif ssl_task == 'colorization':
                    # Colorization task
                    ssl_labels['colorization_label'] = torch.tensor(1, dtype=torch.long)
                    
                elif ssl_task == 'temporal_order':
                    # Temporal order prediction
                    if len(video_frames) > 1:
                        ssl_labels['temporal_order_label'] = torch.tensor(1, dtype=torch.long)
            
        except Exception as e:
            if self.logging:
                print(f"Warning: Could not apply SSL augmentations: {e}")
        
        return ssl_labels
    
    def _compute_sample_weight(self, sample_idx):
        """Compute sample weight for weighted sampling."""
        try:
            sample = self.data[sample_idx]
            
            # Base weight from class imbalance
            is_fake = sample.get('n_fakes', 0) > 0
            base_weight = self.class_weights[1] if is_fake else self.class_weights[0]
            
            # Adjust weight based on difficulty (if curriculum learning)
            if self.curriculum_learning and self.difficulty_scores is not None:
                idx_in_valid = self.valid_indices.index(sample_idx)
                difficulty = self.difficulty_scores[idx_in_valid]
                # Give higher weights to more difficult samples
                difficulty_weight = 1.0 + difficulty
                base_weight *= difficulty_weight
            
            return base_weight.item() if isinstance(base_weight, torch.Tensor) else base_weight
            
        except Exception as e:
            if self.logging:
                print(f"Warning: Could not compute sample weight: {e}")
            return 1.0

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
        video_frames = torch.zeros((self.max_frames, 3, 224, 224))
        audio_tensor = torch.zeros(self.audio_length)
        audio_spectrogram = torch.zeros((1, 128, 128))
        label = torch.tensor(0, dtype=torch.long)
        facial_landmarks = torch.zeros((self.max_frames, 136))
        
        # Additional placeholder features
        mfcc_features = torch.zeros((40, 100))
        pulse_signal = torch.zeros(self.max_frames)
        skin_color_variations = torch.zeros((self.max_frames, 3))
        head_pose_features = torch.zeros((self.max_frames, 3))
        eye_blink_features = torch.zeros(self.max_frames)
        frequency_features = torch.zeros((1, 32, 32))
        
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
            'file_path': 'placeholder',
            'facial_landmarks': facial_landmarks,
            'mfcc_features': mfcc_features,
            'pulse_signal': pulse_signal,
            'skin_color_variations': skin_color_variations,
            'head_pose': head_pose_features,
            'eye_blink_features': eye_blink_features,
            'frequency_features': frequency_features,
            'ensemble_features': {},
            'uncertainty_estimates': {'combined_uncertainty': torch.tensor(0.5)},
            'domain_label': torch.tensor(0, dtype=torch.long),
            'ssl_labels': {},
            'difficulty_score': 0.5,
            'sample_weight': 1.0
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
            if total_frames <= 0:
                if self.logging:
                    warnings.warn(f"⚠️ Video has no frames: {path}")
                return None, None, None, None
                
            # Create sampling indices for frames
            if self.phase == 'train':
                if total_frames <= self.max_frames:
                    frame_indices = list(range(total_frames))
                    if len(frame_indices) < self.max_frames:
                        frame_indices = frame_indices * math.ceil(self.max_frames / len(frame_indices))
                        frame_indices = frame_indices[:self.max_frames]
                else:
                    frame_indices = sorted(random.sample(range(total_frames), self.max_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
                
            video_frames = []
            face_crops = []
            all_landmarks = []
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
                consistency_score = 1.0
                landmarks = []
                
                if self.detect_faces:
                    try:
                        frame_rgb_8bit = (frame_rgb * 255).astype(np.uint8) if frame_rgb.dtype != np.uint8 else frame_rgb
                        pil_img = Image.fromarray(frame_rgb_8bit)
                        
                        # Detect faces
                        boxes, probs = self.face_detector.detect(pil_img)
                        
                        if boxes is not None and len(boxes) > 0:
                            box = boxes[0]
                            face_detected = True
                            
                            # Check temporal consistency
                            if self.temporal_features and prev_face_locs is not None:
                                movement = np.mean(np.abs(box - prev_face_locs))
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
                                
                            # Extract facial landmarks
                            if self.enhanced_preprocessing and hasattr(self, 'dlib_detector') and self.dlib_detector is not None:
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                dlib_faces = self.dlib_detector(gray)
                                
                                if dlib_faces and hasattr(self, 'landmark_predictor') and self.landmark_predictor is not None:
                                    shape = self.landmark_predictor(gray, dlib_faces[0])
                                    landmarks = []
                                    for i in range(68):
                                        x = shape.part(i).x
                                        y = shape.part(i).y
                                        landmarks.extend([x, y])
                                
                    except Exception as e:
                        if self.logging:
                            print(f"Face detection error on frame {frame_idx}: {e}")
                
                # Extract facial landmarks even if no face detected
                all_landmarks.append(landmarks if landmarks else [0] * 136)
                
                # Resize frame
                frame_rgb = cv2.resize(frame_rgb, (224, 224))
                
                # Apply transformations
                if self.transform:
                    if self.phase == 'train':
                        frame_rgb = self.transform(frame_rgb)
                    else:
                        frame_rgb = self.transform(frame_rgb)
                else:
                    frame_rgb = torch.tensor(frame_rgb).permute(2, 0, 1).float() / 255.0
                
                video_frames.append(frame_rgb)
            cap.release()

            if not video_frames:
                if self.logging:
                    warnings.warn(f"⚠️ No valid frames extracted from video: {path}")
                return None, None, None, None

            # Stack frames into tensor
            video_tensor = torch.stack(video_frames)
            
            # Process face crops
            face_embeddings = None
            if face_crops:
                face_crops_tensor = torch.stack([
                    torch.tensor(crop).permute(2, 0, 1).float() / 255.0
                    for crop in face_crops
                ])
                face_embeddings = torch.mean(face_crops_tensor.reshape(face_crops_tensor.size(0), -1), dim=1)
            else:
                face_embeddings = torch.zeros((1, 512))
                
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

            # Process audio length
            if len(audio) > self.audio_length:
                if self.phase == 'train':
                    start = random.randint(0, len(audio) - self.audio_length)
                    audio = audio[start:start + self.audio_length]
                else:
                    start = (len(audio) - self.audio_length) // 2
                    audio = audio[start:start + self.audio_length]
            else:
                audio = np.pad(audio, (0, self.audio_length - len(audio)), mode='constant')

            # Apply audio augmentation
            if self.audio_transform and self.phase == 'train':
                try:
                    audio = self.audio_transform(samples=audio, sample_rate=sample_rate)
                except Exception as audio_transform_error:
                    if self.logging:
                        warnings.warn(f"⚠️ Audio transform error for file {path}. Error: {audio_transform_error}")

            # Compute mel spectrogram
            audio_spec = None
            if self.compute_spectrograms:
                try:
                    mel_spec = librosa.feature.melspectrogram(
                        y=audio, 
                        sr=sample_rate,
                        n_mels=128,
                        hop_length=512,
                        n_fft=2048
                    )
                    
                    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                    mel_spec = cv2.resize(mel_spec, (128, 128))
                    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
                    
                    audio_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
                except Exception as e:
                    if self.logging:
                        warnings.warn(f"⚠️ Error computing spectrogram: {e}")
                    audio_spec = torch.zeros((1, 128, 128), dtype=torch.float32)
            else:
                audio_spec = torch.zeros((1, 128, 128), dtype=torch.float32)
            
            # Extract MFCC features
            mfcc_features = None
            try:
                mfccs = librosa.feature.mfcc(
                    y=audio, 
                    sr=sample_rate, 
                    n_mfcc=40,
                    hop_length=512,
                    n_fft=2048
                )
                mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
                mfcc_features = torch.tensor(mfccs, dtype=torch.float32)
            except Exception as e:
                if self.logging:
                    warnings.warn(f"⚠️ Error computing MFCC features: {e}")
                mfcc_features = torch.zeros((40, 100), dtype=torch.float32)

            return torch.tensor(audio, dtype=torch.float32), audio_spec, mfcc_features
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error loading audio file: {path}. Error: {e}")
            return None, None, None
            
    def _extract_metadata_features(self, video_path):
        """Extract metadata features like compression artifacts."""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Extract basic metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Get file size
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            
            # Calculate bitrate
            duration = frame_count / fps if fps > 0 else 0
            bitrate = file_size_mb * 8 / duration if duration > 0 else 0
            
            # Extract noise level
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
            
            # Compile features
            metadata_features = torch.tensor([
                fps / 30.0,
                min(1.0, file_size_mb / 10.0),
                min(1.0, bitrate / 5000.0),
                noise_level / 10.0,
                noise_std / 10.0,
                min(1.0, quantization_metric * 10),
                width / 1920.0,
                height / 1080.0,
                min(1.0, (width * height) / (1920 * 1080)),
                1.0 if file_size_mb < 0.5 else 0.0
            ], dtype=torch.float32)
            
            return metadata_features
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting metadata features: {e}")
            return torch.zeros(10, dtype=torch.float32)
     
    def _extract_ela_features(self, video_frames):
        """Extract Error Level Analysis features."""
        try:
            if video_frames is None or len(video_frames) == 0:
                return torch.zeros((224, 224), dtype=torch.float32)
                
            # Use first frame for ELA
            first_frame = video_frames[0].permute(1, 2, 0).cpu().numpy()
            first_frame = (first_frame * 255).astype(np.uint8)
            
            # Convert to PIL Image
            img = Image.fromarray(first_frame)
            
            # Save with specific quality
            quality = 90
            import uuid
            temp_filename = f"temp_ela_{os.getpid()}_{uuid.uuid4()}.jpg"
            img.save(temp_filename, 'JPEG', quality=quality)
            
            # Read back
            saved_img = np.array(Image.open(temp_filename))
            
            # Calculate difference
            ela = np.abs(first_frame.astype(np.float32) - saved_img.astype(np.float32))
            ela_gray = np.mean(ela, axis=2)
            ela_resized = cv2.resize(ela_gray, (224, 224))
            ela_normalized = ela_resized / ela_resized.max() if ela_resized.max() > 0 else ela_resized
            
            # Clean up
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception:
                    pass

            return torch.tensor(ela_normalized, dtype=torch.float32)
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting ELA features: {e}")
            return torch.zeros((224, 224), dtype=torch.float32)
    
    def _extract_av_sync_features(self, video_frames, audio_tensor):
        """Extract audio-visual synchronization features."""
        try:
            if video_frames.shape[0] < 2:
                return torch.zeros(5, dtype=torch.float32)
                
            # Calculate frame differences
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
                
                # Calculate lag
                max_lag = min(5, min_length-1)
                best_lag = 0
                best_corr = correlation
                
                for lag in range(1, max_lag+1):
                    corr_pos = torch.mean(frame_diffs[lag:] * audio_energy[:-lag]) if min_length > lag else torch.tensor(0.)
                    corr_neg = torch.mean(frame_diffs[:-lag] * audio_energy[lag:]) if min_length > lag else torch.tensor(0.)
                    
                    if corr_pos > best_corr:
                        best_corr = corr_pos
                        best_lag = lag
                    if corr_neg > best_corr:
                        best_corr = corr_neg
                        best_lag = -lag
                
                # Features
                sync_features = torch.tensor([
                    correlation,
                    float(best_lag),
                    best_corr,
                    torch.std(frame_diffs),
                    torch.std(audio_energy)
                ], dtype=torch.float32)
                
                return sync_features
            else:
                return torch.zeros(5, dtype=torch.float32)
                
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting A/V sync features: {e}")
            return torch.zeros(5, dtype=torch.float32)
    
    def _extract_pulse_signal(self, video_frames):
        """Extract pulse signal (rPPG)."""
        try:
            if video_frames is None or len(video_frames) < 2:
                return torch.zeros(self.max_frames, dtype=torch.float32)
            
            green_values = []
            
            for i in range(len(video_frames)):
                frame = video_frames[i].permute(1, 2, 0).cpu().numpy()
                
                # Simple skin detection
                r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
                skin_mask = (r > 0.4) & (g > 0.2) & (b > 0.2) & (r > g) & (r > b)
                
                if np.any(skin_mask):
                    green_mean = np.mean(g[skin_mask])
                else:
                    green_mean = np.mean(g)
                
                green_values.append(green_mean)
            
            signal = np.array(green_values)
            
            # Simple filtering
            if len(signal) > 5:
                fps = 30
                nyquist = fps / 2
                low = 0.7 / nyquist
                high = 4.0 / nyquist
                b, a = scipy.signal.butter(3, [low, high], btype='band')
                filtered_signal = scipy.signal.filtfilt(b, a, signal)
                filtered_signal = (filtered_signal - np.mean(filtered_signal)) / (np.std(filtered_signal) + 1e-8)
            else:
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
        """Extract skin color variations."""
        try:
            if video_frames is None or len(video_frames) < 2:
                return torch.zeros((self.max_frames, 3), dtype=torch.float32)
            
            skin_colors = []
            
            for i in range(len(video_frames)):
                frame = video_frames[i].permute(1, 2, 0).cpu().numpy()
                
                # Simple skin detection
                r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
                skin_mask = (r > 0.4) & (g > 0.2) & (b > 0.2) & (r > g) & (r > b)
                
                if np.any(skin_mask):
                    r_mean = np.mean(r[skin_mask])
                    g_mean = np.mean(g[skin_mask])
                    b_mean = np.mean(b[skin_mask])
                else:
                    r_mean = np.mean(r)
                    g_mean = np.mean(g)
                    b_mean = np.mean(b)
                
                skin_colors.append([r_mean, g_mean, b_mean])
            
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
        """Estimate head pose from facial landmarks."""
        try:
            if facial_landmarks is None or facial_landmarks.size(0) == 0:
                return torch.zeros((self.max_frames, 3), dtype=torch.float32)
            
            num_frames = facial_landmarks.size(0)
            head_poses = []
            
            for i in range(num_frames):
                landmarks = facial_landmarks[i]
                
                if torch.sum(landmarks) < 1e-6:
                    head_poses.append([0, 0, 0])
                    continue
                
                # Extract key landmarks
                left_eye_x = (landmarks[2*36] + landmarks[2*39]) / 2
                left_eye_y = (landmarks[2*36+1] + landmarks[2*39+1]) / 2
                right_eye_x = (landmarks[2*42] + landmarks[2*45]) / 2
                right_eye_y = (landmarks[2*42+1] + landmarks[2*45+1]) / 2
                nose_x = landmarks[2*30]
                nose_y = landmarks[2*30+1]
                mouth_x = (landmarks[2*48] + landmarks[2*54]) / 2
                mouth_y = (landmarks[2*48+1] + landmarks[2*54+1]) / 2
                
                # Calculate pose estimates
                eye_diff_x = left_eye_x - right_eye_x
                yaw = eye_diff_x.item() if abs(eye_diff_x.item()) < 100 else 0
                
                eyes_center_y = (left_eye_y + right_eye_y) / 2
                pitch = (eyes_center_y - nose_y).item() if abs(eyes_center_y - nose_y) < 100 else 0
                
                eye_diff_y = left_eye_y - right_eye_y
                roll = eye_diff_y.item() if abs(eye_diff_y.item()) < 100 else 0
                
                # Normalize
                yaw = yaw / 50.0
                pitch = pitch / 30.0
                roll = roll / 20.0
                
                head_poses.append([pitch, yaw, roll])
            
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
        """Extract eye blinking patterns."""
        try:
            if video_frames is None or len(video_frames) < 2 or facial_landmarks is None:
                return torch.zeros(self.max_frames, dtype=torch.float32)
            
            blink_patterns = []
            
            num_frames = min(len(video_frames), facial_landmarks.size(0))
            
            for i in range(num_frames):
                landmarks = facial_landmarks[i]
                
                if torch.sum(landmarks) < 1e-6:
                    blink_patterns.append(1.0)  # Default open eye
                    continue
                
                # Extract eye landmarks (assuming 68-point model)
                # Left eye: points 36-41, Right eye: points 42-47
                left_eye_landmarks = landmarks[2*36:2*42]  # x,y pairs for left eye
                right_eye_landmarks = landmarks[2*42:2*48]  # x,y pairs for right eye
                
                # Calculate eye aspect ratio (EAR)
                left_ear = self._calculate_eye_aspect_ratio(left_eye_landmarks)
                right_ear = self._calculate_eye_aspect_ratio(right_eye_landmarks)
                
                # Average EAR
                avg_ear = (left_ear + right_ear) / 2.0
                blink_patterns.append(avg_ear)
            
            blink_patterns = np.array(blink_patterns)
            
            # Ensure correct length
            if len(blink_patterns) < self.max_frames:
                blink_patterns = np.pad(blink_patterns, 
                                      (0, self.max_frames - len(blink_patterns)), 
                                      mode='constant', constant_values=1.0)
            else:
                blink_patterns = blink_patterns[:self.max_frames]
            
            return torch.tensor(blink_patterns, dtype=torch.float32)
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting eye blink patterns: {e}")
            return torch.zeros(self.max_frames, dtype=torch.float32)
    
    def _calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR) for blink detection."""
        try:
            if len(eye_landmarks) < 12:  # Need at least 6 points (12 coordinates)
                return 1.0
            
            # Reshape to get (x, y) pairs
            points = eye_landmarks.reshape(-1, 2)
            
            if len(points) < 6:
                return 1.0
            
            # Calculate distances
            # Vertical eye distances
            A = torch.dist(points[1], points[5])
            B = torch.dist(points[2], points[4])
            
            # Horizontal eye distance
            C = torch.dist(points[0], points[3])
            
            # EAR calculation
            ear = (A + B) / (2.0 * C + 1e-8)
            
            return ear.item()
            
        except Exception as e:
            return 1.0
    
    def _extract_frequency_features(self, video_frames):
        """Extract frequency domain features from video frames."""
        try:
            if video_frames is None or len(video_frames) < 2:
                return torch.zeros((1, 32, 32), dtype=torch.float32)
            
            # Use first frame for frequency analysis
            first_frame = video_frames[0]
            
            # Convert to grayscale
            if first_frame.shape[0] == 3:
                gray_frame = 0.299 * first_frame[0] + 0.587 * first_frame[1] + 0.114 * first_frame[2]
            else:
                gray_frame = first_frame[0]
            
            # Apply 2D FFT
            fft_frame = torch.fft.fft2(gray_frame)
            fft_magnitude = torch.abs(fft_frame)
            
            # Log transform for better visualization
            fft_magnitude = torch.log(fft_magnitude + 1e-8)
            
            # Resize to fixed size
            fft_resized = F.interpolate(fft_magnitude.unsqueeze(0).unsqueeze(0), 
                                       size=(32, 32), mode='bilinear', align_corners=False)
            
            # Normalize
            fft_normalized = (fft_resized - fft_resized.min()) / (fft_resized.max() - fft_resized.min() + 1e-8)
            
            return fft_normalized.squeeze(0)
            
        except Exception as e:
            if self.logging:
                warnings.warn(f"⚠️ Error extracting frequency features: {e}")
            return torch.zeros((1, 32, 32), dtype=torch.float32)


# NEW: Advanced DataLoader with Real-time Optimization
class OptimizedDataLoader:
    """
    Advanced DataLoader with real-time optimization, streaming capabilities,
    and adaptive resolution scaling.
    """
    
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, 
                 pin_memory=True, drop_last=False, streaming_mode=False,
                 adaptive_resolution=False, early_exit=False, hierarchical_processing=False,
                 quantization=False, model_pruning=False, knowledge_distillation=False):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
        # NEW: Real-time optimization features
        self.streaming_mode = streaming_mode
        self.adaptive_resolution = adaptive_resolution
        self.early_exit = early_exit
        self.hierarchical_processing = hierarchical_processing
        
        # NEW: Model optimization features
        self.quantization = quantization
        self.model_pruning = model_pruning
        self.knowledge_distillation = knowledge_distillation
        
        # Initialize frame buffer for streaming
        if self.streaming_mode:
            self.frame_buffer = {}
            self.buffer_size = 64
            self.sliding_window_size = 16
            
        # Initialize adaptive resolution parameters
        if self.adaptive_resolution:
            self.resolution_levels = [(224, 224), (192, 192), (160, 160), (128, 128)]
            self.current_resolution_idx = 0
            self.performance_history = []
            
        # Initialize hierarchical processing levels
        if self.hierarchical_processing:
            self.processing_levels = ['coarse', 'medium', 'fine']
            self.current_level = 'coarse'
            
        # Initialize quantization parameters
        if self.quantization:
            self.quantization_schemes = ['int8', 'fp16', 'dynamic']
            self.current_scheme = 'fp16'
            
        # Create weighted sampler for imbalanced data
        self.weighted_sampler = self._create_weighted_sampler()
        
        # Create the actual DataLoader
        self.dataloader = self._create_dataloader()
        
        print(f"OptimizedDataLoader initialized with advanced features:")
        print(f"  - Streaming mode: {self.streaming_mode}")
        print(f"  - Adaptive resolution: {self.adaptive_resolution}")
        print(f"  - Early exit: {self.early_exit}")
        print(f"  - Hierarchical processing: {self.hierarchical_processing}")
        print(f"  - Quantization: {self.quantization}")
        print(f"  - Model pruning: {self.model_pruning}")
        print(f"  - Knowledge distillation: {self.knowledge_distillation}")
    
    def _create_weighted_sampler(self):
        """Create weighted sampler for imbalanced datasets."""
        try:
            if hasattr(self.dataset, 'class_weights') and self.dataset.class_weights is not None:
                # Calculate sample weights
                sample_weights = []
                for idx in range(len(self.dataset)):
                    sample = self.dataset.data[self.dataset.valid_indices[idx]]
                    is_fake = sample.get('n_fakes', 0) > 0
                    weight = self.dataset.class_weights[1] if is_fake else self.dataset.class_weights[0]
                    sample_weights.append(weight.item())
                
                return WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
            else:
                return None
                
        except Exception as e:
            print(f"Warning: Could not create weighted sampler: {e}")
            return None
    
    def _create_dataloader(self):
        """Create the actual PyTorch DataLoader."""
        sampler = self.weighted_sampler if self.weighted_sampler and self.shuffle else None
        shuffle = self.shuffle if sampler is None else False
        
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function with advanced features."""
        try:
            # Standard collation
            collated = {}
            
            # Handle different data types
            for key in batch[0].keys():
                if key in ['ensemble_features', 'ssl_labels', 'uncertainty_estimates']:
                    # Handle dictionary features
                    collated[key] = batch[0][key]  # Keep as dict for first sample
                elif isinstance(batch[0][key], torch.Tensor):
                    # Stack tensors
                    collated[key] = torch.stack([item[key] for item in batch])
                elif isinstance(batch[0][key], (list, tuple)):
                    # Keep lists as lists
                    collated[key] = [item[key] for item in batch]
                else:
                    # Keep other types as lists
                    collated[key] = [item[key] for item in batch]
            
            # NEW: Apply real-time optimizations
            if self.streaming_mode:
                collated = self._apply_streaming_optimization(collated)
            
            if self.adaptive_resolution:
                collated = self._apply_adaptive_resolution(collated)
            
            if self.hierarchical_processing:
                collated = self._apply_hierarchical_processing(collated)
            
            if self.quantization:
                collated = self._apply_quantization(collated)
            
            return collated
            
        except Exception as e:
            print(f"Warning: Error in collate function: {e}")
            return batch[0]  # Return first item as fallback
    
    def _apply_streaming_optimization(self, batch):
        """Apply streaming optimization with sliding window."""
        try:
            # Implement sliding window for video frames
            if 'video_frames' in batch:
                video_frames = batch['video_frames']
                batch_size, num_frames = video_frames.shape[0], video_frames.shape[1]
                
                # Apply sliding window
                if num_frames > self.sliding_window_size:
                    # Select frames using sliding window
                    start_idx = random.randint(0, num_frames - self.sliding_window_size)
                    batch['video_frames'] = video_frames[:, start_idx:start_idx + self.sliding_window_size]
                
                # Update frame buffer
                for i in range(batch_size):
                    sample_id = f"sample_{i}"
                    if sample_id not in self.frame_buffer:
                        self.frame_buffer[sample_id] = []
                    
                    # Add current frames to buffer
                    self.frame_buffer[sample_id].extend(video_frames[i])
                    
                    # Maintain buffer size
                    if len(self.frame_buffer[sample_id]) > self.buffer_size:
                        self.frame_buffer[sample_id] = self.frame_buffer[sample_id][-self.buffer_size:]
            
            return batch
            
        except Exception as e:
            print(f"Warning: Error in streaming optimization: {e}")
            return batch
    
    def _apply_adaptive_resolution(self, batch):
        """Apply adaptive resolution scaling based on performance."""
        try:
            if 'video_frames' in batch:
                video_frames = batch['video_frames']
                current_resolution = self.resolution_levels[self.current_resolution_idx]
                
                # Resize frames to current resolution
                if video_frames.shape[-2:] != current_resolution:
                    batch['video_frames'] = F.interpolate(
                        video_frames.view(-1, *video_frames.shape[-3:]),
                        size=current_resolution,
                        mode='bilinear',
                        align_corners=False
                    ).view(video_frames.shape[0], video_frames.shape[1], video_frames.shape[2], *current_resolution)
                
                # Add resolution info to batch
                batch['current_resolution'] = torch.tensor(self.current_resolution_idx)
            
            return batch
            
        except Exception as e:
            print(f"Warning: Error in adaptive resolution: {e}")
            return batch
    
    def _apply_hierarchical_processing(self, batch):
        """Apply hierarchical processing levels."""
        try:
            # Add processing level info
            batch['processing_level'] = self.current_level
            
            # Modify features based on processing level
            if self.current_level == 'coarse':
                # Use only basic features
                if 'video_frames' in batch:
                    batch['video_frames'] = F.avg_pool2d(batch['video_frames'].view(-1, *batch['video_frames'].shape[-3:]), 2)
                    batch['video_frames'] = batch['video_frames'].view(batch['video_frames'].shape[0], -1, *batch['video_frames'].shape[-3:])
                    
            elif self.current_level == 'medium':
                # Use intermediate features
                pass  # Keep original resolution
                
            elif self.current_level == 'fine':
                # Use all features at full resolution
                pass  # Keep all features
            
            return batch
            
        except Exception as e:
            print(f"Warning: Error in hierarchical processing: {e}")
            return batch
    
    def _apply_quantization(self, batch):
        """Apply quantization to reduce precision."""
        try:
            if self.current_scheme == 'int8':
                # Convert to int8
                for key in ['video_frames', 'audio', 'audio_spectrogram']:
                    if key in batch and isinstance(batch[key], torch.Tensor):
                        # Scale to int8 range
                        tensor = batch[key]
                        if tensor.dtype == torch.float32:
                            tensor_scaled = (tensor * 127).clamp(-128, 127).to(torch.int8)
                            batch[f'{key}_quantized'] = tensor_scaled
                            batch[f'{key}_scale'] = torch.tensor(1.0 / 127.0)
                            
            elif self.current_scheme == 'fp16':
                # Convert to fp16
                for key in ['video_frames', 'audio', 'audio_spectrogram']:
                    if key in batch and isinstance(batch[key], torch.Tensor):
                        if batch[key].dtype == torch.float32:
                            batch[key] = batch[key].to(torch.float16)
                            
            elif self.current_scheme == 'dynamic':
                # Dynamic quantization (placeholder)
                batch['quantization_enabled'] = True
            
            return batch
            
        except Exception as e:
            print(f"Warning: Error in quantization: {e}")
            return batch
    
    def update_performance_metrics(self, inference_time, accuracy):
        """Update performance metrics for adaptive optimization."""
        try:
            self.performance_history.append({
                'inference_time': inference_time,
                'accuracy': accuracy,
                'resolution_idx': self.current_resolution_idx,
                'processing_level': self.current_level
            })
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            # Adaptive resolution adjustment
            if self.adaptive_resolution and len(self.performance_history) > 10:
                recent_times = [h['inference_time'] for h in self.performance_history[-10:]]
                recent_accs = [h['accuracy'] for h in self.performance_history[-10:]]
                
                avg_time = np.mean(recent_times)
                avg_acc = np.mean(recent_accs)
                
                # Adjust resolution based on performance
                if avg_time > 0.5 and self.current_resolution_idx < len(self.resolution_levels) - 1:
                    # Inference too slow, decrease resolution
                    self.current_resolution_idx += 1
                    print(f"Decreased resolution to {self.resolution_levels[self.current_resolution_idx]}")
                elif avg_time < 0.1 and avg_acc > 0.9 and self.current_resolution_idx > 0:
                    # Inference fast and accurate, increase resolution
                    self.current_resolution_idx -= 1
                    print(f"Increased resolution to {self.resolution_levels[self.current_resolution_idx]}")
            
            # Hierarchical processing adjustment
            if self.hierarchical_processing and len(self.performance_history) > 5:
                recent_accs = [h['accuracy'] for h in self.performance_history[-5:]]
                avg_acc = np.mean(recent_accs)
                
                if avg_acc < 0.7:
                    self.current_level = 'fine'
                elif avg_acc > 0.9:
                    self.current_level = 'coarse'
                else:
                    self.current_level = 'medium'
                    
        except Exception as e:
            print(f"Warning: Error updating performance metrics: {e}")
    
    def get_early_exit_threshold(self):
        """Get threshold for early exit mechanism."""
        if self.early_exit:
            return 0.9  # Exit early if confidence > 90%
        return 1.0
    
    def __iter__(self):
        """Iterator for the dataloader."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Length of the dataloader."""
        return len(self.dataloader)


# NEW: Advanced Training Utilities
class AdvancedTrainingUtils:
    """
    Utility class for advanced training techniques including:
    - Progressive training (curriculum learning)
    - Multi-task learning integration
    - Meta-learning for few-shot adaptation
    - Domain adaptation techniques
    - Active learning for data selection
    """
    
    def __init__(self):
        self.curriculum_scheduler = None
        self.meta_learner = None
        self.domain_adapter = None
        self.active_selector = None
        
    def setup_curriculum_learning(self, dataset, strategy='difficulty_based'):
        """Setup curriculum learning scheduler."""
        try:
            self.curriculum_scheduler = CurriculumScheduler(dataset, strategy)
            print(f"Curriculum learning setup with strategy: {strategy}")
        except Exception as e:
            print(f"Warning: Could not setup curriculum learning: {e}")
    
    def setup_meta_learning(self, model, meta_lr=0.001, adaptation_steps=5):
        """Setup meta-learning for few-shot adaptation."""
        try:
            self.meta_learner = MetaLearner(model, meta_lr, adaptation_steps)
            print(f"Meta-learning setup with lr: {meta_lr}, steps: {adaptation_steps}")
        except Exception as e:
            print(f"Warning: Could not setup meta-learning: {e}")
    
    def setup_domain_adaptation(self, source_domain, target_domain, adaptation_method='DANN'):
        """Setup domain adaptation."""
        try:
            self.domain_adapter = DomainAdapter(source_domain, target_domain, adaptation_method)
            print(f"Domain adaptation setup with method: {adaptation_method}")
        except Exception as e:
            print(f"Warning: Could not setup domain adaptation: {e}")
    
    def setup_active_learning(self, dataset, selection_strategy='uncertainty'):
        """Setup active learning for data selection."""
        try:
            self.active_selector = ActiveLearningSelector(dataset, selection_strategy)
            print(f"Active learning setup with strategy: {selection_strategy}")
        except Exception as e:
            print(f"Warning: Could not setup active learning: {e}")


class CurriculumScheduler:
    """Curriculum learning scheduler."""
    
    def __init__(self, dataset, strategy='difficulty_based'):
        self.dataset = dataset
        self.strategy = strategy
        self.current_epoch = 0
        
    def get_curriculum_data(self, epoch):
        """Get curriculum data for current epoch."""
        self.current_epoch = epoch
        
        if hasattr(self.dataset, 'update_curriculum_epoch'):
            self.dataset.update_curriculum_epoch(epoch)
        
        if hasattr(self.dataset, 'get_curriculum_samples'):
            return self.dataset.get_curriculum_samples()
        
        return list(range(len(self.dataset)))


class MetaLearner:
    """Meta-learner for few-shot adaptation."""
    
    def __init__(self, model, meta_lr=0.001, adaptation_steps=5):
        self.model = model
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        
    def adapt_to_task(self, support_data, query_data):
        """Adapt model to new task using support data."""
        # Clone model for adaptation
        adapted_model = copy.deepcopy(self.model)
        adapted_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=0.01)
        
        # Adaptation phase
        for _ in range(self.adaptation_steps):
            # Forward pass on support data
            support_loss = self._compute_loss(adapted_model, support_data)
            
            # Backward pass
            adapted_optimizer.zero_grad()
            support_loss.backward()
            adapted_optimizer.step()
        
        # Evaluate on query data
        query_loss = self._compute_loss(adapted_model, query_data)
        
        return query_loss
    
    def _compute_loss(self, model, data):
        """Compute loss for given data."""
        # Placeholder loss computation
        return torch.tensor(0.0, requires_grad=True)


class DomainAdapter:
    """Domain adaptation utility."""
    
    def __init__(self, source_domain, target_domain, adaptation_method='DANN'):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.adaptation_method = adaptation_method
        self.domain_classifier = None
        
        if adaptation_method == 'DANN':
            self.domain_classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 2)  # 2 domains
            )
    
    def compute_domain_loss(self, features, domain_labels):
        """Compute domain adaptation loss."""
        if self.domain_classifier is None:
            return torch.tensor(0.0)
        
        # Gradient reversal layer (simplified)
        reversed_features = features * -1.0
        domain_pred = self.domain_classifier(reversed_features)
        
        loss_fn = nn.CrossEntropyLoss()
        domain_loss = loss_fn(domain_pred, domain_labels)
        
        return domain_loss


class ActiveLearningSelector:
    """Active learning data selector."""
    
    def __init__(self, dataset, selection_strategy='uncertainty'):
        self.dataset = dataset
        self.selection_strategy = selection_strategy
        self.uncertainty_estimates = None
        
    def select_samples(self, model, n_samples=100):
        """Select most informative samples."""
        if self.selection_strategy == 'uncertainty':
            return self._select_by_uncertainty(model, n_samples)
        elif self.selection_strategy == 'diversity':
            return self._select_by_diversity(model, n_samples)
        else:
            return self._select_hybrid(model, n_samples)
    
    def _select_by_uncertainty(self, model, n_samples):
        """Select samples with highest uncertainty."""
        # Placeholder implementation
        return list(range(min(n_samples, len(self.dataset))))
    
    def _select_by_diversity(self, model, n_samples):
        """Select diverse samples."""
        # Placeholder implementation
        return list(range(min(n_samples, len(self.dataset))))
    
    def _select_hybrid(self, model, n_samples):
        """Select using hybrid strategy."""
        # Placeholder implementation
        return list(range(min(n_samples, len(self.dataset))))


# NEW: Model Optimization Utilities
class ModelOptimizer:
    """
    Model optimization utilities including:
    - Post-training quantization (INT8/FP16)
    - Model pruning (structured/unstructured)
    - Knowledge distillation
    - Dynamic quantization for mobile
    - TensorRT optimization
    """
    
    def __init__(self):
        self.quantization_schemes = ['int8', 'fp16', 'dynamic']
        self.pruning_methods = ['magnitude', 'structured', 'unstructured']
        self.distillation_methods = ['knowledge_distillation', 'feature_distillation']
        
    def apply_quantization(self, model, scheme='fp16'):
        """Apply quantization to model."""
        try:
            if scheme == 'int8':
                return self._apply_int8_quantization(model)
            elif scheme == 'fp16':
                return self._apply_fp16_quantization(model)
            elif scheme == 'dynamic':
                return self._apply_dynamic_quantization(model)
            else:
                return model
        except Exception as e:
            print(f"Warning: Could not apply quantization: {e}")
            return model
    
    def _apply_int8_quantization(self, model):
        """Apply INT8 quantization."""
        try:
            import torch.quantization as quant
            
            # Prepare model for quantization
            model.eval()
            model.qconfig = quant.get_default_qconfig('fbgemm')
            model_prepared = quant.prepare(model)
            
            # Calibrate (would need calibration data in practice)
            # For now, just return prepared model
            model_quantized = quant.convert(model_prepared)
            
            return model_quantized
        except Exception as e:
            print(f"Warning: INT8 quantization failed: {e}")
            return model
    
    def _apply_fp16_quantization(self, model):
        """Apply FP16 quantization."""
        try:
            return model.half()
        except Exception as e:
            print(f"Warning: FP16 quantization failed: {e}")
            return model
    
    def _apply_dynamic_quantization(self, model):
        """Apply dynamic quantization."""
        try:
            import torch.quantization as quant
            
            model_quantized = quant.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
            return model_quantized
        except Exception as e:
            print(f"Warning: Dynamic quantization failed: {e}")
            return model
    
    def apply_pruning(self, model, method='magnitude', sparsity=0.3):
        """Apply model pruning."""
        try:
            if method == 'magnitude':
                return self._apply_magnitude_pruning(model, sparsity)
            elif method == 'structured':
                return self._apply_structured_pruning(model, sparsity)
            elif method == 'unstructured':
                return self._apply_unstructured_pruning(model, sparsity)
            else:
                return model
        except Exception as e:
            print(f"Warning: Could not apply pruning: {e}")
            return model
    
    def _apply_magnitude_pruning(self, model, sparsity):
        """Apply magnitude-based pruning."""
        try:
            import torch.nn.utils.prune as prune
            
            parameters_to_prune = []
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    parameters_to_prune.append((module, 'weight'))
            
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=sparsity
            )
            
            return model
        except Exception as e:
            print(f"Warning: Magnitude pruning failed: {e}")
            return model
    
    def _apply_structured_pruning(self, model, sparsity):
        """Apply structured pruning."""
        try:
            import torch.nn.utils.prune as prune
            
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=sparsity, n=2, dim=0)
                elif isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
            
            return model
        except Exception as e:
            print(f"Warning: Structured pruning failed: {e}")
            return model
    
    def _apply_unstructured_pruning(self, model, sparsity):
        """Apply unstructured pruning."""
        try:
            import torch.nn.utils.prune as prune
            
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
            
            return model
        except Exception as e:
            print(f"Warning: Unstructured pruning failed: {e}")
            return model
    
    def apply_knowledge_distillation(self, teacher_model, student_model, temperature=3.0):
        """Apply knowledge distillation."""
        try:
            return KnowledgeDistillationLoss(teacher_model, student_model, temperature)
        except Exception as e:
            print(f"Warning: Could not setup knowledge distillation: {e}")
            return None


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss function."""
    
    def __init__(self, teacher_model, student_model, temperature=3.0):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, inputs, targets, alpha=0.5):
        """Compute knowledge distillation loss."""
        try:
            # Teacher predictions
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)
                teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
            
            # Student predictions
            student_outputs = self.student_model(inputs)
            student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
            
            # Distillation loss
            distillation_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
            
            # Classification loss
            classification_loss = F.cross_entropy(student_outputs, targets)
            
            # Combined loss
            total_loss = alpha * distillation_loss + (1 - alpha) * classification_loss
            
            return total_loss
        except Exception as e:
            print(f"Warning: Knowledge distillation loss computation failed: {e}")
            return F.cross_entropy(self.student_model(inputs), targets)


# NEW: Export and utility functions
def create_enhanced_dataloader(json_path, data_dir, batch_size=32, phase='train', 
                             max_frames=32, audio_length=16000, 
                             adversarial_training=False, self_supervised_pretraining=False,
                             curriculum_learning=False, active_learning=False, 
                             domain_adaptation=False, streaming_mode=False,
                             adaptive_resolution=False, quantization=False):
    """
    Create an enhanced dataloader with all advanced features.
    
    Args:
        json_path: Path to JSON annotation file
        data_dir: Directory containing video/audio files
        batch_size: Batch size for training
        phase: 'train', 'val', or 'test'
        max_frames: Maximum number of frames to extract
        audio_length: Length of audio samples
        adversarial_training: Enable adversarial training
        self_supervised_pretraining: Enable self-supervised pretraining
        curriculum_learning: Enable curriculum learning
        active_learning: Enable active learning
        domain_adaptation: Enable domain adaptation
        streaming_mode: Enable streaming/real-time mode
        adaptive_resolution: Enable adaptive resolution scaling
        quantization: Enable quantization
    
    Returns:
        OptimizedDataLoader: Enhanced dataloader with all features
    """
    
    # Create transforms
    if phase == 'train':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        audio_transform = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.3),
            Shift(min_fraction=-0.1, max_fraction=0.1, p=0.3),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        audio_transform = None
    
    # Create dataset with all advanced features
    dataset = MultiModalDeepfakeDataset(
        json_path=json_path,
        data_dir=data_dir,
        max_frames=max_frames,
        audio_length=audio_length,
        transform=transform,
        audio_transform=audio_transform,
        logging=True,
        phase=phase,
        detect_faces=True,
        compute_spectrograms=True,
        temporal_features=True,
        enhanced_preprocessing=True,
        adversarial_training=adversarial_training,
        self_supervised_pretraining=self_supervised_pretraining,
        curriculum_learning=curriculum_learning,
        active_learning=active_learning,
        domain_adaptation=domain_adaptation
    )
    
    # Create optimized dataloader
    dataloader = OptimizedDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(phase == 'train'),
        num_workers=4,
        pin_memory=True,
        drop_last=(phase == 'train'),
        streaming_mode=streaming_mode,
        adaptive_resolution=adaptive_resolution,
        early_exit=True,
        hierarchical_processing=True,
        quantization=quantization,
        model_pruning=False,
        knowledge_distillation=False
    )
    
    return dataloader


def get_training_utilities():
    """Get training utilities for advanced training techniques."""
    return AdvancedTrainingUtils()


def get_model_optimizer():
    """Get model optimizer for model optimization techniques."""
    return ModelOptimizer()


# Export all classes and functions
__all__ = [
    'MultiModalDeepfakeDataset',
    'OptimizedDataLoader', 
    'AdvancedTrainingUtils',
    'ModelOptimizer',
    'KnowledgeDistillationLoss',
    'create_enhanced_dataloader',
    'get_training_utilities',
    'get_model_optimizer'
]

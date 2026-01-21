"""
Enhanced Augmentation Pipeline for Deepfake Detection

COMBINED VERSION: Merges improved_augmentation.py + production_robustness.py

This module provides:
1. Original advanced augmentation techniques (MixUp, CutMix, TemporalConsistency)
2. Production robustness features (compression, resolution, lighting, domain adaptation)
3. Demographic-aware sampling for fairness

Addresses:
- Temporal consistency across video frames
- Social media compression artifacts
- Resolution degradation (low/mid/high quality)
- Lighting variations and real-world conditions
- Domain adaptation for production deployment
- Demographic bias prevention
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)
from audiomentations import (
    Compose, AddGaussianNoise, PitchShift, TimeStretch, 
    Shift, Gain, BandPassFilter, LowPassFilter
)
import torchaudio
import librosa
from packaging import version
if version.parse(A.__version__) < version.parse("1.3.0"):
    raise ImportError("Albumentations >= 1.3.0 required for ReplayCompose and .replay(). Please upgrade.")
# Set deterministic random seed for reproducibility
random.seed(42)
np.random.seed(42)


# ============================================================================
# PRODUCTION ROBUSTNESS CLASSES (NEW)
# ============================================================================

class SocialMediaCompressionSimulator:
    """
    Simulates aggressive compression patterns from social media platforms.
    
    Platforms like TikTok, Instagram, Facebook apply 2-3 rounds of compression
    with varying quality levels. This helps the model generalize to real-world data.
    """
    
    def __init__(self):
        # Common compression quality ranges for different platforms
        self.platform_profiles = {
            'instagram': {'quality_range': (65, 85), 'rounds': 2},
            'tiktok': {'quality_range': (60, 80), 'rounds': 3},
            'facebook': {'quality_range': (70, 90), 'rounds': 2},
            'youtube': {'quality_range': (75, 95), 'rounds': 1},
            'whatsapp': {'quality_range': (50, 70), 'rounds': 2},  # Very aggressive
            'twitter': {'quality_range': (65, 85), 'rounds': 2},
        }
    
    def __call__(self, image, platform=None):
        """
        Apply multi-round compression similar to social media platforms.
        
        Args:
            image: numpy array [H, W, C]
            platform: str or None (random if None)
        
        Returns:
            Compressed image
        """
        if platform is None:
            platform = random.choice(list(self.platform_profiles.keys()))
        
        profile = self.platform_profiles[platform]
        
        # Apply multiple rounds of compression
        compressed = image.copy()
        for _ in range(profile['rounds']):
            quality = random.randint(*profile['quality_range'])
            
            # JPEG compression
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', compressed, encode_param)
            compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        return compressed


class ResolutionDegradation:
    """
    Simulates low/mid/high resolution variations.
    
    Many detectors fail on low-resolution videos because they're trained
    on high-quality datasets. This augmentation helps bridge the gap.
    """
    
    def __init__(self):
        # Common resolution downsample factors
        self.resolution_profiles = {
            'high': 1.0,      # 224x224 → 224x224
            'mid': 0.5,       # 224x224 → 112x112 → 224x224
            'low': 0.3,       # 224x224 → 67x67 → 224x224  
            'very_low': 0.2,  # 224x224 → 45x45 → 224x224 (phone recording of screen)
        }
    
    def __call__(self, image, quality=None):
        """
        Downscale and upscale to simulate resolution loss.
        
        Args:
            image: numpy array [H, W, C]
            quality: str or None (random if None)
        
        Returns:
            Resolution-degraded image
        """
        if quality is None:
            quality = random.choice(list(self.resolution_profiles.keys()))
        
        scale = self.resolution_profiles[quality]
        
        if scale >= 1.0:
            return image
        
        h, w = image.shape[:2]
        
        # Downscale
        new_h, new_w = int(h * scale), int(w * scale)
        downscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Upscale back (introduces blur and artifacts)
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return upscaled


class AdaptiveLightingAugmentation:
    """
    Simulates diverse lighting conditions that cause detector failures.
    
    Includes:
    - Low light (underexposed)
    - Overexposed (bright conditions)
    - Uneven lighting (shadows)
    - Color temperature shifts (indoor/outdoor)
    """
    
    def __init__(self):
        pass
    
    def __call__(self, image):
        """
        Apply realistic lighting variations.
        
        Args:
            image: numpy array [H, W, C]
        
        Returns:
            Lighting-augmented image
        """
        augmentation = random.choice([
            'low_light',
            'overexposed', 
            'shadow',
            'color_temp',
            'none'
        ])
        
        if augmentation == 'low_light':
            # Reduce brightness and increase noise
            gamma = random.uniform(0.3, 0.7)
            image = np.clip(255 * (image / 255) ** gamma, 0, 255).astype(np.uint8)
            # Add noise in low light
            noise = np.random.normal(0, 5, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        elif augmentation == 'overexposed':
            # Increase brightness, lose detail in highlights
            gamma = random.uniform(1.3, 1.8)
            image = np.clip(255 * (image / 255) ** gamma, 0, 255).astype(np.uint8)
        
        elif augmentation == 'shadow':
            # Apply non-uniform brightness
            h, w = image.shape[:2]
            
            # Create gradient mask
            gradient = np.linspace(0.5, 1.0, w)
            gradient = np.tile(gradient, (h, 1))
            gradient = np.expand_dims(gradient, axis=2)
            
            image = np.clip(image * gradient, 0, 255).astype(np.uint8)
        
        elif augmentation == 'color_temp':
            # Shift color temperature (warm/cool)
            temp = random.choice(['warm', 'cool'])
            
            if temp == 'warm':
                # Orange/yellow tint (indoor/tungsten)
                image[:, :, 2] = np.clip(image[:, :, 2] * 1.1, 0, 255)  # More red
                image[:, :, 0] = np.clip(image[:, :, 0] * 0.9, 0, 255)  # Less blue
            else:
                # Blue tint (outdoor/daylight)
                image[:, :, 0] = np.clip(image[:, :, 0] * 1.1, 0, 255)  # More blue
                image[:, :, 2] = np.clip(image[:, :, 2] * 0.9, 0, 255)  # Less red
        
        return image


# ============================================================================
# PICKLABLE WRAPPER CLASSES FOR MULTIPROCESSING COMPATIBILITY
# ============================================================================

class CompressionAugmenter:
    """Picklable wrapper for social media compression augmentation."""
    def __init__(self, probability=0.7):
        self.probability = probability
        self.compressor = SocialMediaCompressionSimulator()
    
    def __call__(self, image, **kwargs):
        if random.random() < self.probability:
            return self.compressor(image)
        return image


class ResolutionAugmenter:
    """Picklable wrapper for resolution degradation augmentation."""
    def __init__(self, probability=0.5):
        self.probability = probability
        self.degrader = ResolutionDegradation()
    
    def __call__(self, image, **kwargs):
        if random.random() < self.probability:
            return self.degrader(image)
        return image


class LightingAugmenter:
    """Picklable wrapper for adaptive lighting augmentation."""
    def __init__(self, probability=0.6):
        self.probability = probability
        self.lighter = AdaptiveLightingAugmentation()
    
    def __call__(self, image, **kwargs):
        if random.random() < self.probability:
            return self.lighter(image)
        return image


class FixedCompressionAugmenter:
    """Picklable wrapper for fixed platform compression (validation)."""
    def __init__(self, platform='instagram'):
        self.compressor = SocialMediaCompressionSimulator()
        self.platform = platform
    
    def __call__(self, image, **kwargs):
        return self.compressor(image, platform=self.platform)


class FixedResolutionAugmenter:
    """Picklable wrapper for fixed quality resolution degradation (validation)."""
    def __init__(self, quality='low'):
        self.degrader = ResolutionDegradation()
        self.quality = quality
    
    def __call__(self, image, **kwargs):
        return self.degrader(image, quality=self.quality)


class DemographicAwareSampling:
    """
    Ensures balanced sampling across demographics to reduce bias.
    
    Tracks skin tone, age, gender distribution and ensures training
    sees balanced examples to prevent performance disparities.
    """
    
    def __init__(self, metadata_path=None):
        """
        Args:
            metadata_path: Path to demographic metadata (if available)
        """
        self.demographic_groups = {
            'skin_tone': ['light', 'medium', 'dark'],
            'age': ['young', 'middle', 'senior'],
            'gender': ['male', 'female', 'other']
        }
        
        # Track sample counts per group
        self.group_counts = {}
        for category in self.demographic_groups:
            for group in self.demographic_groups[category]:
                self.group_counts[f"{category}_{group}"] = 0
    
    def update_counts(self, sample_metadata):
        """Update demographic counts for a sampled item."""
        for category in self.demographic_groups:
            if category in sample_metadata:
                group = sample_metadata[category]
                key = f"{category}_{group}"
                if key in self.group_counts:
                    self.group_counts[key] += 1
    
    def get_underrepresented_groups(self):
        """
        Returns list of demographic groups that need more samples.
        """
        if not self.group_counts:
            return []
        
        avg_count = np.mean(list(self.group_counts.values()))
        underrepresented = []
        
        for group, count in self.group_counts.items():
            if count < avg_count * 0.8:  # Less than 80% of average
                underrepresented.append(group)
        
        return underrepresented
    
    def should_sample(self, sample_metadata):
        """
        Decides if a sample should be included based on demographic balance.
        
        Returns:
            (should_sample: bool, weight: float)
        """
        # Always sample, but assign higher weight to underrepresented groups
        weight = 1.0
        
        underrepresented = self.get_underrepresented_groups()
        
        for category in self.demographic_groups:
            if category in sample_metadata:
                group = sample_metadata[category]
                key = f"{category}_{group}"
                
                if key in underrepresented:
                    weight *= 1.5  # 50% higher weight
        
        return True, weight


# ============================================================================
# COMBINED AUGMENTATION FUNCTIONS (ORIGINAL + PRODUCTION ROBUSTNESS)
# ============================================================================

def get_advanced_video_transforms(train=True, use_production_robust=True):
    """
    Get enhanced video augmentation pipeline specifically designed for deepfake detection.
    
    COMBINED VERSION: Includes both original augmentations AND production robustness features.
    
    Args:
        train: Whether to use training or validation transforms
        use_production_robust: If True (default), adds social media compression, resolution degradation, and lighting variations
    
    Returns:
        Albumentation transforms pipeline
    """
    if train:
        transforms_list = []
        
        # ===== PRODUCTION ROBUSTNESS FEATURES (NEW) =====
        if use_production_robust:
            # CRITICAL: Social media compression simulation (70% of training data)
            # Use picklable wrapper instead of lambda for multiprocessing compatibility
            transforms_list.append(
                A.Lambda(
                    name="SocialMediaCompression",
                    image=CompressionAugmenter(probability=0.7),
                    p=1.0
                )
            )
            
            # CRITICAL: Resolution degradation (50% of training data)
            # Use picklable wrapper instead of lambda for multiprocessing compatibility
            transforms_list.append(
                A.Lambda(
                    name="ResolutionDegradation", 
                    image=ResolutionAugmenter(probability=0.5),
                    p=1.0
                )
            )
            
            # CRITICAL: Lighting variations (60% of training data)
            # Use picklable wrapper instead of lambda for multiprocessing compatibility
            transforms_list.append(
                A.Lambda(
                    name="AdaptiveLighting",
                    image=LightingAugmenter(probability=0.6),
                    p=1.0
                )
            )
        
        # ===== ORIGINAL AUGMENTATIONS (PRESERVED) =====
        
        # Spatial transforms
        transforms_list.append(
            A.OneOf([
                A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                A.Resize(height=224, width=224),
            ], p=1.0)
        )
        
        # Color/Intensity transforms - crucial for deepfake detection
        transforms_list.append(
            A.OneOf([
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
            ], p=0.8)
        )
        
        # Noise transforms to make model robust to different quality levels
        if use_production_robust:
            # Enhanced noise (stacked with compression artifacts)
            transforms_list.append(
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    A.MotionBlur(blur_limit=(3, 7), p=0.3),
                    A.GaussNoise(var_limit=(5.0, 20.0), p=0.4),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.15), p=0.4),
                ], p=0.7)
            )
        else:
            # Original noise transforms
            transforms_list.append(
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=0.4),
                    A.GaussNoise(var_limit=(5.0, 20.0), p=0.4),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.15), p=0.4),
                    A.ImageCompression(quality_lower=70, quality_upper=99, p=0.4),
                ], p=0.7)
            )
        
        # Facial detail preservation and enhancement transforms
        transforms_list.append(
            A.OneOf([
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3), p=0.5),
                A.UnsharpMask(sigma_limit=(0.5, 1.5), alpha=(0.1, 0.5), p=0.5),
                A.RandomToneCurve(scale=0.1, p=0.3),
            ], p=0.5)
        )
        
        # Slight geometric distortions
        transforms_list.append(
            A.OneOf([
                A.ElasticTransform(alpha=0.5, sigma=25, alpha_affine=5, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.3),
            ], p=0.3)
        )
        
        # Small rotations
        transforms_list.append(
            A.Affine(rotate=[-5, 5], scale=[0.95, 1.05], p=0.5)
        )
        
        # Normalization and conversion to tensor
        transforms_list.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        return A.Compose(transforms_list)
    
    else:
        # Validation transforms
        if use_production_robust:
            # Test on multiple quality levels for robustness evaluation
            quality_level = random.choice(['high', 'mid', 'low'])
            
            if quality_level == 'high':
                # Clean validation
                return A.Compose([
                    A.Resize(height=224, width=224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            
            elif quality_level == 'mid':
                # Medium quality (typical social media)
                return A.Compose([
                    A.Lambda(image=FixedCompressionAugmenter(platform='instagram'), p=1.0),
                    A.Resize(height=224, width=224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            
            else:  # low
                # Low quality (aggressive compression + resolution loss)
                return A.Compose([
                    A.Lambda(image=FixedCompressionAugmenter(platform='whatsapp'), p=1.0),
                    A.Lambda(image=FixedResolutionAugmenter(quality='low'), p=1.0),
                    A.Resize(height=224, width=224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
        else:
            # Original validation (only essential preprocessing)
            return A.Compose([
                A.Resize(height=224, width=224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

def get_advanced_audio_transforms(train=True):
    """
    Get enhanced audio augmentation pipeline for deepfake detection.
    
    Args:
        train: Whether to use training or validation transforms
    
    Returns:
        Audiomentations transform pipeline
    """
    if train:
        return Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
            Shift(min_shift=-0.1, max_shift=0.1, p=0.5),
            Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
            BandPassFilter(min_center_freq=200.0, max_center_freq=4000.0, 
                          min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.5, p=0.3),
            LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=7500, p=0.3),
        ])
    else:
        # No transformations for validation/testing
        return None

class TemporalConsistencyAugmenter:
    """
    Applies consistent augmentations across video frames to maintain temporal coherence.
    """
    def __init__(self, base_transform):
        """
        Args:
            base_transform: Albumentation transform to apply consistently
        """
        self.base_transform = base_transform
    
    def __call__(self, frames):
        """
        Apply the same transform to all frames in the video
        
        Args:
            frames: List of frames or tensor of shape [T, H, W, C]
            
        Returns:
            Transformed frames with same augmentation applied to all
        """
        # Handle channel order: if frames are [T, C, H, W], convert to [T, H, W, C]
        if isinstance(frames, torch.Tensor):
            is_tensor = True
            device = frames.device
            shape = frames.shape
            # If shape is [T, C, H, W], transpose to [T, H, W, C]
            if frames.dim() == 4 and shape[1] in [1, 3]:
                frames = frames.permute(0, 2, 3, 1).cpu().numpy()
            else:
                frames = frames.cpu().numpy()
        else:
            is_tensor = False
        
        # Use ReplayCompose to capture a single randomization and replay it across frames
        # If the provided base_transform is not a ReplayCompose, wrap its transforms.
        try:
            if isinstance(self.base_transform, A.ReplayCompose):
                replay_transform = self.base_transform
            else:
                # Wrap the underlying transforms into a ReplayCompose for deterministic replay
                replay_transform = A.ReplayCompose(self.base_transform.transforms)

            # Ensure ReplayCompose does not include ToTensorV2 (we want numpy arrays here)
            if not isinstance(self.base_transform, A.ReplayCompose):
                # If base_transform has a .transforms list, filter out ToTensorV2 so outputs are numpy arrays
                if hasattr(self.base_transform, 'transforms'):
                    filtered = [t for t in getattr(self.base_transform, 'transforms', []) if not isinstance(t, ToTensorV2)]
                    try:
                        replay_transform = A.ReplayCompose(filtered)
                    except Exception:
                        # Fallback to wrapping original transforms
                        replay_transform = A.ReplayCompose(getattr(self.base_transform, 'transforms', []))
                else:
                    replay_transform = A.ReplayCompose(getattr(self.base_transform, 'transforms', []))
            else:
                replay_transform = self.base_transform

            # Apply to first frame and capture the replay metadata
            first_out = replay_transform(image=frames[0])
            replay = first_out.get('replay', None)

            def _to_numpy(img):
                # Convert torch.Tensor outputs to numpy arrays and handle albumentations outputs
                if isinstance(img, torch.Tensor):
                    try:
                        return img.cpu().numpy()
                    except Exception:
                        return np.array(img)
                return img

            first_image = first_out['image'] if isinstance(first_out, dict) and 'image' in first_out else first_out
            first_image = _to_numpy(first_image)
            result = [first_image]

            # Replay the same augmentation on remaining frames
            for frame in frames[1:]:
                try:
                    replayed = replay_transform.replay(replay, image=frame)
                    img = replayed['image'] if isinstance(replayed, dict) and 'image' in replayed else replayed
                    img = _to_numpy(img)
                    result.append(img)
                except Exception:
                    # Fallback: apply the base transform deterministically if replay fails
                    try:
                        out = self.base_transform(image=frame) if hasattr(self.base_transform, '__call__') else self.base_transform(image=frame)
                        img = out['image'] if isinstance(out, dict) and 'image' in out else out
                        img = _to_numpy(img)
                        result.append(img)
                    except Exception:
                        # Last-resort: append the raw frame
                        result.append(frame)
        except Exception:
            # If anything goes wrong, fall back to applying the base transform per-frame
            result = []
            for frame in frames:
                try:
                    out = self.base_transform(image=frame) if hasattr(self.base_transform, '__call__') else self.base_transform(image=frame)
                    result.append(out['image'] if isinstance(out, dict) and 'image' in out else out)
                except Exception:
                    # Last-resort: append the raw frame
                    result.append(frame)
        
        if is_tensor:
            # Convert list of numpy arrays to a single numpy array robustly
            try:
                result = np.stack([r if isinstance(r, np.ndarray) else np.array(r) for r in result], axis=0)
            except Exception:
                # Last-resort: create object array then try to coerce
                result = np.array(result)

            # If result is uint8, normalize to float32 0-1
            if getattr(result, 'dtype', None) == np.uint8:
                result = result.astype(np.float32) / 255.0

            # Ensure float32
            try:
                result = result.astype(np.float32)
            except Exception:
                pass

            result = torch.tensor(result, device=device, dtype=torch.float32)
            # Restore channel order if needed
            if result.dim() == 4 and shape[1] in [1, 3]:
                result = result.permute(0, 3, 1, 2)
            try:
                result = result.view(shape)
            except Exception:
                # If view fails, leave as-is
                pass
        
        return result

def mix_up_augmentation(inputs1, inputs2, targets1, targets2, alpha=0.2):
    """
    Implements MixUp augmentation for both video and audio inputs.
    
    Args:
        inputs1, inputs2: Dictionary of tensors containing 'video_frames' and 'audio'
        targets1, targets2: Target tensors
        alpha: Alpha parameter for beta distribution
        
    Returns:
        Tuple of mixed inputs and targets
    """
    # Sample lambda from beta distribution (deterministic seed already set)
    lam = np.random.beta(alpha, alpha)
    
    # Create empty mixed inputs dictionary
    mixed_inputs = {}
    
    # Mix video frames
    if 'video_frames' in inputs1 and 'video_frames' in inputs2:
        mixed_inputs['video_frames'] = lam * inputs1['video_frames'] + (1 - lam) * inputs2['video_frames']
    
    # Mix audio
    if 'audio' in inputs1 and 'audio' in inputs2:
        mixed_inputs['audio'] = lam * inputs1['audio'] + (1 - lam) * inputs2['audio']
    
    # Mix targets
    mixed_targets = lam * targets1 + (1 - lam) * targets2
    
    return mixed_inputs, mixed_targets

def cut_mix_augmentation(inputs1, inputs2, targets1, targets2, alpha=0.2):
    """
    Implements CutMix augmentation for both video and audio inputs.
    Args:
        inputs1, inputs2: Dictionary of tensors containing 'video_frames' and 'audio'
        targets1, targets2: Target tensors
        alpha: Alpha parameter for beta distribution
    Returns:
        Tuple of mixed inputs and targets
    """
    lam = np.random.beta(alpha, alpha)
    mixed_inputs = {}
    # Video CutMix
    if 'video_frames' in inputs1 and 'video_frames' in inputs2:
        frames1 = inputs1['video_frames']
        frames2 = inputs2['video_frames']
        # Assume shape [B, C, H, W] or [C, H, W]
        bbx1, bby1, bbx2, bby2 = rand_bbox(frames1.shape, lam)
        mixed = frames1.clone()
        mixed[..., bby1:bby2, bbx1:bbx2] = frames2[..., bby1:bby2, bbx1:bbx2]
        mixed_inputs['video_frames'] = mixed
    # Audio CutMix (simple: replace random segment)
    if 'audio' in inputs1 and 'audio' in inputs2:
        audio1 = inputs1['audio']
        audio2 = inputs2['audio']
        length = audio1.shape[-1]
        cut_len = int(length * lam)
        if cut_len <= 0:
            mixed_inputs['audio'] = audio1
        elif cut_len >= length:
            mixed_inputs['audio'] = audio2.clone()
        else:
            # ensure randint upper bound >= 1
            start = np.random.randint(0, length - cut_len + 1)
            mixed_audio = audio1.clone()
            mixed_audio[..., start:start+cut_len] = audio2[..., start:start+cut_len]
            mixed_inputs['audio'] = mixed_audio
    mixed_targets = lam * targets1 + (1 - lam) * targets2
    return mixed_inputs, mixed_targets

def rand_bbox(size, lam):
    """
    Generate random bounding box for CutMix.
    Args:
        size: shape of the input tensor
        lam: lambda value
    Returns:
        bbx1, bby1, bbx2, bby2
    """
    W = size[-1]
    H = size[-2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

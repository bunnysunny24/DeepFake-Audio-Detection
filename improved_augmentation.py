"""
Enhanced Augmentation Pipeline for Deepfake Detection
This module provides improved data augmentation techniques specifically designed for deepfake detection.
"""

import cv2
import numpy as np
import torch
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

def get_advanced_video_transforms(train=True):
    """
    Get enhanced video augmentation pipeline specifically designed for deepfake detection.
    
    Args:
        train: Whether to use training or validation transforms
    
    Returns:
        Albumentation transforms pipeline
    """
    if train:
        return A.Compose([
            # Spatial transforms
            A.OneOf([
                A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                A.Resize(height=224, width=224),
            ], p=1.0),
            
            # Color/Intensity transforms - crucial for deepfake detection
            A.OneOf([
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
            ], p=0.8),
            
            # Noise transforms to make model robust to different quality levels
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.4),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.4),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.15), p=0.4),
                A.ImageCompression(quality_lower=70, quality_upper=99, p=0.4),
            ], p=0.7),
            
            # Facial detail preservation and enhancement transforms
            A.OneOf([
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3), p=0.5),
                A.UnsharpMask(sigma_limit=(0.5, 1.5), alpha=(0.1, 0.5), p=0.5),
                A.RandomToneCurve(scale=0.1, p=0.3),
            ], p=0.5),
            
            # Slight geometric distortions
            A.OneOf([
                A.ElasticTransform(alpha=0.5, sigma=25, alpha_affine=5, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.3),
            ], p=0.3),
            
            # Small rotations
            A.Affine(rotate=[-5, 5], scale=[0.95, 1.05], p=0.5),
            
            # Normalization and conversion to tensor
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        # Validation transforms - only essential preprocessing
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
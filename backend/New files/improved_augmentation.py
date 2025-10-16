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
from audiomentations import (
    Compose, AddGaussianNoise, PitchShift, TimeStretch, 
    Shift, Gain, BandPassFilter, LowPassFilter
)
import torchaudio
import librosa

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
        if isinstance(frames, torch.Tensor):
            # Convert tensor to numpy for albumentation transforms
            is_tensor = True
            device = frames.device
            shape = frames.shape
            frames = frames.cpu().numpy()
        else:
            is_tensor = False
        
        # Get a single transform params instance for consistent application
        params = self.base_transform.get_params()
        
        # Apply consistently to all frames
        result = []
        for frame in frames:
            aug_frame = self.base_transform.apply(image=frame, **params)
            result.append(aug_frame)
        
        if is_tensor:
            # Convert back to tensor
            result = torch.tensor(np.array(result), device=device)
            result = result.view(shape)
        
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
    # Sample lambda from beta distribution
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

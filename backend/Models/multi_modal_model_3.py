
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, AutoModel
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, swin_v2_b
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import torch.nn.functional as F
import timm
import cv2


class PhysiologicalSignalDetector(nn.Module):
    """Module for detecting physiological signals such as heart rate from facial videos."""
    def __init__(self, input_dim=3, hidden_dim=64):
        super(PhysiologicalSignalDetector, self).__init__()
        
        # Convolutional layers for spatial feature extraction
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Enhanced temporal analysis with attention for more accurate heart rate extraction
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim*2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Temporal attention for focusing on relevant heartbeat signal segments
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim*2, 1),
            nn.Sigmoid()
        )
        
        # Frequency domain analysis - NEW for enhanced rPPG
        self.frequency_encoder = nn.Sequential(
            nn.Linear(30, hidden_dim),  # Frequency bands up to 3Hz (typical heart rate range)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Heart rate variability analysis with enhanced metrics (SDNN, RMSSD, etc.)
        self.hrv_analyzer = nn.Sequential(
            nn.Linear(hidden_dim*2 + hidden_dim, hidden_dim),  # Added frequency features
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # Outputs: [estimated_hr, hrv_score, consistency_score, sdnn, rmssd]
        )
        
        # Enhanced skin color variation detection - NEW more detail
        self.skin_analyzer = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # Outputs: [color_variation_score, naturality_score, blood_perfusion, melanin_consistency]
        )
        
        # Enhanced respiratory pattern analysis - NEW more detail
        self.resp_analyzer = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # Outputs: [breathing_rate, breathing_consistency, breathing_depth, breath_holding]
        )
        
        # Enhanced combined physiological features
        self.feature_combiner = nn.Sequential(
            nn.Linear(13, 128),  # 5 + 4 + 4 = 13 enhanced features
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
    def extract_rppg(self, face_regions):
        """Extract enhanced remote photoplethysmography (rPPG) signal from facial regions."""
        # Split RGB channels and focus on green channel (most sensitive to blood flow)
        batch_size, seq_len, c, h, w = face_regions.shape
        
        # Reshape for spatial encoder
        face_flat = face_regions.view(batch_size * seq_len, c, h, w)
        spatial_features = self.spatial_encoder(face_flat)
        
        # Reshape back for temporal processing
        temporal_features = spatial_features.view(batch_size, seq_len, -1)
        
        # Apply temporal attention to focus on relevant parts of the signal
        attention_weights = self.temporal_attention(temporal_features)
        weighted_features = temporal_features * attention_weights
        
        # Extract temporal patterns (pulse signal)
        rppg_features, _ = self.temporal_encoder(weighted_features)
        
        # Get final representation (from last timestep)
        final_features = rppg_features[:, -1, :]
        
        # NEW: Extract frequency domain features for more accurate heart rate estimation
        # This simulates a heart rate spectrum analysis (in real implementation, use FFT)
        frequency_features = torch.zeros((batch_size, 30), device=face_regions.device)
        for b in range(batch_size):
            # Extract green channel signal (most sensitive to blood flow)
            if c >= 3:  # Ensure we have an RGB image
                green_channel = face_regions[b, :, 1, :, :].mean(dim=(1, 2))
                
                # Remove trend
                detrended = green_channel - torch.mean(green_channel)
                
                # Simple frequency analysis (simulate - in real impl use torch.fft)
                # For demonstration - in production use proper FFT
                for f in range(30):  # 0-30 frequency bands (0-3Hz)
                    freq = (f + 1) / 10.0  # 0.1Hz to 3.0Hz
                    sin_wave = torch.sin(torch.tensor([i * freq * 2 * 3.14159 / seq_len for i in range(seq_len)]))
                    # Correlation with this frequency
                    correlation = torch.abs(torch.sum(detrended * sin_wave.to(detrended.device)))
                    frequency_features[b, f] = correlation
                    
            # Normalize
            if torch.max(frequency_features[b]) > 0:
                frequency_features[b] = frequency_features[b] / torch.max(frequency_features[b])
                
        # Process frequency features
        freq_encoding = self.frequency_encoder(frequency_features)
        
        # Combine temporal and frequency features
        combined_features = torch.cat([final_features, freq_encoding], dim=1)
        
        return combined_features
        
    def forward(self, face_regions):
        """
        Analyze physiological signals from facial video sequences.
        
        Args:
            face_regions: Tensor of facial regions [batch_size, sequence_length, channels, height, width]
            
        Returns:
            Dictionary of physiological features and a combined feature vector
        """
        # Extract enhanced rPPG features
        physio_features = self.extract_rppg(face_regions)
        
        # Analyze heart rate and HRV with enhanced metrics
        hrv_outputs = self.hrv_analyzer(physio_features)
        heart_rate = hrv_outputs[:, 0:1]
        hrv_score = hrv_outputs[:, 1:2]
        hr_consistency = hrv_outputs[:, 2:3]
        sdnn = hrv_outputs[:, 3:4]  # Standard deviation of NN intervals
        rmssd = hrv_outputs[:, 4:5]  # Root mean square of successive differences
        
        # Analyze skin color variations with enhanced metrics
        skin_outputs = self.skin_analyzer(physio_features[:, :128])  # Use only temporal features
        color_variation = skin_outputs[:, 0:1]
        skin_naturality = skin_outputs[:, 1:2]
        blood_perfusion = skin_outputs[:, 2:3]  # NEW: Blood perfusion indicator
        melanin_consistency = skin_outputs[:, 3:4]  # NEW: Skin tone consistency
        
        # Analyze respiratory patterns with enhanced metrics
        resp_outputs = self.resp_analyzer(physio_features[:, :128])  # Use only temporal features
        breathing_rate = resp_outputs[:, 0:1]
        breathing_consistency = resp_outputs[:, 1:2]
        breathing_depth = resp_outputs[:, 2:3]  # NEW: Depth of breathing
        breath_holding = resp_outputs[:, 3:4]  # NEW: Unnatural breath holding detection
        
        # Combine all physiological indicators with enhanced metrics
        combined_physio = torch.cat([
            heart_rate, hrv_score, hr_consistency, sdnn, rmssd,
            color_variation, skin_naturality, blood_perfusion, melanin_consistency,
            breathing_rate, breathing_consistency, breathing_depth, breath_holding
        ], dim=1)
        
        # Get final feature representation
        physio_embedding = self.feature_combiner(combined_physio)
        
        # Create detailed output dictionary for interpretability
        outputs = {
            'heart_rate': heart_rate,
            'hrv_score': hrv_score,
            'hr_consistency': hr_consistency,
            'sdnn': sdnn,  # NEW: Standard deviation of NN intervals
            'rmssd': rmssd,  # NEW: Root mean square of successive differences
            'color_variation': color_variation,
            'skin_naturality': skin_naturality,
            'blood_perfusion': blood_perfusion,  # NEW: Blood perfusion indicator
            'melanin_consistency': melanin_consistency,  # NEW: Skin tone consistency
            'breathing_rate': breathing_rate,
            'breathing_consistency': breathing_consistency,
            'breathing_depth': breathing_depth,  # NEW: Depth of breathing
            'breath_holding': breath_holding,  # NEW: Unnatural breath holding detection
            'physio_embedding': physio_embedding
        }
        
        return outputs, physio_embedding


class OcularBehavioralAnalyzer(nn.Module):
    """Module for analyzing ocular and facial behavioral cues."""
    def __init__(self, input_dim=3, hidden_dim=64):
        super(OcularBehavioralAnalyzer, self).__init__()
        
        # Enhanced convolutional layers for eye region analysis
        self.eye_encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, stride=1, padding=1),  # NEW: deeper network
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Enhanced temporal analysis for eye movements with attention
        self.eye_temporal = nn.GRU(
            input_size=hidden_dim*4,  # Increased feature dimension
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3  # Increased dropout for better generalization
        )
        
        # NEW: Attention mechanism for temporal eye movement analysis
        self.movement_attention = nn.Sequential(
            nn.Linear(hidden_dim*2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # Enhanced pupil dilation analysis
        self.pupil_analyzer = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Added dropout
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 4)  # [dilation_score, consistency_score, light_response, emotional_response]
        )
        
        # Enhanced eye movement pattern analysis (saccades, fixations, etc.)
        self.movement_analyzer = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Added dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)  # [saccade_score, fixation_score, blink_naturalness, smooth_pursuit, vergence]
        )
        
        # NEW: Advanced micro-expression detection network
        self.micro_expression_encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # NEW: Temporal analysis for micro-expressions
        self.micro_expression_temporal = nn.GRU(
            input_size=hidden_dim*2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Enhanced micro-expression analysis
        self.micro_expression = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Added dropout
            nn.Linear(hidden_dim, 4)  # [micro_expr_score, naturalness_score, emotion_consistency, facial_muscle_activation]
        )
        
        # NEW: Facial muscle movement analysis
        self.muscle_analyzer = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [mouth_area_naturality, eye_area_naturality, forehead_naturality]
        )
        
        # Enhanced combined ocular behavioral features
        self.feature_combiner = nn.Sequential(
            nn.Linear(16, 256),  # 4 + 5 + 4 + 3 = 16 features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
    def extract_movement_patterns(self, eye_temporal_features):
        """Extract fine-grained eye movement patterns with attention mechanism."""
        # Apply attention to focus on relevant eye movements
        attention_weights = self.movement_attention(eye_temporal_features)
        attended_features = torch.sum(eye_temporal_features * attention_weights, dim=1)
        return attended_features
        
    def forward(self, eye_regions, face_sequence=None):
        """
        Analyze ocular behavior from eye region sequences with enhanced detection.
        
        Args:
            eye_regions: Tensor of eye regions [batch_size, sequence_length, channels, height, width]
            face_sequence: Optional tensor of full face for micro-expression analysis [batch_size, sequence_length, channels, height, width]
            
        Returns:
            Dictionary of ocular features and a combined feature vector
        """
        batch_size, seq_len, c, h, w = eye_regions.shape
        
        # Process each frame through enhanced spatial encoder
        eye_flat = eye_regions.view(batch_size * seq_len, c, h, w)
        eye_features = self.eye_encoder(eye_flat)
        
        # Reshape for temporal processing
        eye_temporal = eye_features.view(batch_size, seq_len, -1)
        
        # Extract temporal patterns of eye movements
        eye_movement_features, _ = self.eye_temporal(eye_temporal)
        
        # NEW: Apply attention-based movement pattern extraction
        attended_features = self.extract_movement_patterns(eye_movement_features)
        
        # Enhanced pupil dilation analysis
        pupil_outputs = self.pupil_analyzer(attended_features)
        dilation_score = pupil_outputs[:, 0:1]
        dilation_consistency = pupil_outputs[:, 1:2]
        light_response = pupil_outputs[:, 2:3]  # NEW: Pupil response to light changes
        emotional_response = pupil_outputs[:, 3:4]  # NEW: Pupil response to emotional stimuli
        
        # Enhanced eye movement patterns analysis
        movement_outputs = self.movement_analyzer(attended_features)
        saccade_score = movement_outputs[:, 0:1]
        fixation_score = movement_outputs[:, 1:2]
        blink_naturalness = movement_outputs[:, 2:3]
        smooth_pursuit = movement_outputs[:, 3:4]  # NEW: Smooth pursuit eye movements
        vergence = movement_outputs[:, 4:5]  # NEW: Vergence eye movements (depth perception)
        
        # Process micro-expressions if face sequence is provided
        if face_sequence is not None and face_sequence.shape[0] == batch_size:
            # Process each frame of the face
            face_flat = face_sequence.view(batch_size * seq_len, c, face_sequence.shape[3], face_sequence.shape[4])
            face_features = self.micro_expression_encoder(face_flat)
            
            # Reshape for temporal processing
            face_temporal = face_features.view(batch_size, seq_len, -1)
            
            # Analyze temporal micro-expression patterns
            face_temporal_features, _ = self.micro_expression_temporal(face_temporal)
            
            # Apply same attention mechanism
            face_attended = torch.sum(face_temporal_features * attention_weights, dim=1)
            
            # Analyze micro-expressions
            micro_outputs = self.micro_expression(face_attended)
            
            # NEW: Analyze facial muscle movements
            muscle_outputs = self.muscle_analyzer(face_attended)
        else:
            # Default values if face not provided
            micro_outputs = torch.zeros((batch_size, 4), device=eye_regions.device)
            muscle_outputs = torch.zeros((batch_size, 3), device=eye_regions.device)
        
        # Unpack micro-expression outputs
        micro_expression_score = micro_outputs[:, 0:1]
        expression_naturalness = micro_outputs[:, 1:2]
        emotion_consistency = micro_outputs[:, 2:3]  # NEW: Emotional consistency across time
        facial_muscle_activation = micro_outputs[:, 3:4]  # NEW: Natural muscle activation patterns
        
        # Unpack muscle movement outputs
        mouth_naturalness = muscle_outputs[:, 0:1]  # NEW: Mouth area movement naturalness
        eye_area_naturalness = muscle_outputs[:, 1:2]  # NEW: Eye area movement naturalness
        forehead_naturalness = muscle_outputs[:, 2:3]  # NEW: Forehead movement naturalness
        
        # Combine all ocular and facial behavioral indicators
        combined_ocular = torch.cat([
            # Pupil features
            dilation_score, dilation_consistency, light_response, emotional_response,
            # Eye movement features
            saccade_score, fixation_score, blink_naturalness, smooth_pursuit, vergence,
            # Micro-expression features
            micro_expression_score, expression_naturalness, emotion_consistency, facial_muscle_activation,
            # Facial muscle features
            mouth_naturalness, eye_area_naturalness, forehead_naturalness
        ], dim=1)
        
        # Get final enhanced feature representation
        ocular_embedding = self.feature_combiner(combined_ocular)
        
        # Create detailed output dictionary for interpretability
        outputs = {
            # Pupil features
            'dilation_score': dilation_score,
            'dilation_consistency': dilation_consistency,
            'light_response': light_response,  # NEW
            'emotional_response': emotional_response,  # NEW
            
            # Eye movement features
            'saccade_score': saccade_score,
            'fixation_score': fixation_score,
            'blink_naturalness': blink_naturalness,
            'smooth_pursuit': smooth_pursuit,  # NEW
            'vergence': vergence,  # NEW
            
            # Micro-expression features
            'micro_expression_score': micro_expression_score,
            'expression_naturalness': expression_naturalness,
            'emotion_consistency': emotion_consistency,  # NEW
            'facial_muscle_activation': facial_muscle_activation,  # NEW
            
            # Facial muscle features
            'mouth_naturalness': mouth_naturalness,  # NEW
            'eye_area_naturalness': eye_area_naturalness,  # NEW
            'forehead_naturalness': forehead_naturalness,  # NEW
            
            # Combined features
            'ocular_embedding': ocular_embedding
        }
        
        return outputs, ocular_embedding


class LipAudioSyncAnalyzer(nn.Module):
    """
    Module for analyzing synchronization between lip movements and audio.
    Detects inconsistencies in lip-audio timing, phoneme-viseme mismatches,
    and unnatural mouth movements.
    """
    def __init__(self, visual_dim=512, audio_dim=512, hidden_dim=256):
        super(LipAudioSyncAnalyzer, self).__init__()
        
        # CNN for extracting lip region features
        self.lip_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Visual-temporal feature extraction for lip movement
        self.visual_temporal = nn.GRU(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Audio feature adaptation network
        self.audio_adapter = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-modal attention for lip-audio alignment
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Temporal alignment detection
        self.alignment_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3)  # [sync_score, temporal_offset, speech_consistency]
        )
        
        # Phoneme-viseme matching network
        self.phoneme_viseme_matcher = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)  # [phoneme_match, articulation_naturalness]
        )
        
        # Mouth movement naturalness analyzer
        self.mouth_movement_analyzer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)  # [movement_naturalness, speaking_style_consistency]
        )
        
        # Combined feature representation
        self.feature_combiner = nn.Sequential(
            nn.Linear(7, 64),  # 3 + 2 + 2 = 7 features
            nn.ReLU(),
            nn.Linear(64, 7)
        )
        
    def extract_lip_features(self, video_frames):
        """Extract features from lip regions in video frames."""
        batch_size, seq_len, c, h, w = video_frames.shape
        
        # Process each frame through lip encoder
        lip_flat = video_frames.view(batch_size * seq_len, c, h, w)
        lip_features = self.lip_encoder(lip_flat)
        
        # Reshape for temporal processing
        lip_temporal = lip_features.view(batch_size, seq_len, -1)
        
        # Extract temporal patterns of lip movements
        lip_movement_features, _ = self.visual_temporal(lip_temporal)
        
        return lip_movement_features
    
    def forward(self, video_frames, audio_features):
        """
        Analyze lip-audio synchronization.
        
        Args:
            video_frames: Tensor of video frames [batch_size, seq_len, channels, height, width]
            audio_features: Tensor of audio features [batch_size, seq_len, audio_dim] or [batch_size, audio_dim]
        
        Returns:
            Dictionary of lip-audio sync features and combined feature vector
        """
        batch_size = video_frames.shape[0]
        
        # Extract lip movement features
        lip_features = self.extract_lip_features(video_frames)
        
        # Prepare audio features
        if len(audio_features.shape) == 2:  # [batch, features]
            # Expand to match sequence length
            audio_seq = audio_features.unsqueeze(1).expand(-1, lip_features.shape[1], -1)
        else:
            audio_seq = audio_features
            
        # Process audio features
        audio_adapted = self.audio_adapter(audio_seq)
        
        # Cross-modal attention for alignment
        attn_output, attn_weights = self.cross_attention(
            audio_adapted, lip_features, lip_features
        )
        
        # Compute alignment features
        alignment_features = torch.cat([attn_output, audio_adapted], dim=-1)
        
        # Use mean pooling for sequence-level analysis
        alignment_pooled = torch.mean(alignment_features, dim=1)
        
        # Detect temporal alignment issues
        alignment_scores = self.alignment_detector(alignment_pooled)
        sync_score = alignment_scores[:, 0:1]
        temporal_offset = alignment_scores[:, 1:2]
        speech_consistency = alignment_scores[:, 2:3]
        
        # Analyze phoneme-viseme matching
        phoneme_scores = self.phoneme_viseme_matcher(alignment_pooled)
        phoneme_match = phoneme_scores[:, 0:1]
        articulation_naturalness = phoneme_scores[:, 1:2]
        
        # Analyze mouth movement naturalness
        movement_scores = self.mouth_movement_analyzer(alignment_pooled)
        movement_naturalness = movement_scores[:, 0:1]
        speaking_style_consistency = movement_scores[:, 1:2]
        
        # Combine all lip-audio sync features
        combined_features = torch.cat([
            sync_score, temporal_offset, speech_consistency,
            phoneme_match, articulation_naturalness,
            movement_naturalness, speaking_style_consistency
        ], dim=1)
        
        # Get final feature representation
        lip_sync_embedding = self.feature_combiner(combined_features)
        
        # Create detailed output dictionary
        outputs = {
            'sync_score': sync_score,
            'temporal_offset': temporal_offset,
            'speech_consistency': speech_consistency,
            'phoneme_match': phoneme_match,
            'articulation_naturalness': articulation_naturalness,
            'movement_naturalness': movement_naturalness,
            'speaking_style_consistency': speaking_style_consistency,
            'attention_weights': attn_weights
        }
        
        return outputs, lip_sync_embedding


class HeadPoseEstimator(nn.Module):
    """Module for head pose estimation to detect unnatural movements."""
    def __init__(self, input_dim=68*2):  # 68 landmarks x 2 coordinates
        super(HeadPoseEstimator, self).__init__()
        
        self.pose_estimator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Pitch, Yaw, Roll
        )
        
        # Temporal consistency network for pose trajectories
        self.temporal_net = nn.GRU(
            input_size=3,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Naturality classifier
        self.naturality_classifier = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)  # [consistency_score, jitter_score, motion_smoothness, rotation_naturalness]
        )
    
    def forward(self, landmark_sequence):
        """
        Estimate head pose from facial landmarks sequence.
        
        Args:
            landmark_sequence: Tensor of shape [batch_size, seq_len, num_landmarks*2]
            
        Returns:
            Dictionary with head pose features and consistency scores
        """
        batch_size, seq_len, num_features = landmark_sequence.shape
        
        # Process each frame independently
        poses = []
        for t in range(seq_len):
            landmarks = landmark_sequence[:, t, :]
            pose = self.pose_estimator(landmarks)
            poses.append(pose)
        
        # Stack poses to form trajectory
        pose_trajectory = torch.stack(poses, dim=1)  # [batch_size, seq_len, 3]
        
        # Process trajectory through temporal network
        trajectory_features, _ = self.temporal_net(pose_trajectory)
        
        # Get final features using last timestep
        final_features = trajectory_features[:, -1, :]
        
        # Classify naturality
        naturality_scores = self.naturality_classifier(final_features)
        
        # Create score dict
        consistency_score = naturality_scores[:, 0:1]
        jitter_score = naturality_scores[:, 1:2]
        smoothness = naturality_scores[:, 2:3]
        rotation_naturalness = naturality_scores[:, 3:4]
        
        return {
            'pose_trajectory': pose_trajectory,
            'consistency_score': consistency_score,
            'jitter_score': jitter_score,
            'motion_smoothness': smoothness,
            'rotation_naturalness': rotation_naturalness
        }


class GANFingerprintDetector(nn.Module):
    """
    Module to detect GAN fingerprints in generated images.
    Different GANs leave distinct patterns that can be detected.
    """
    def __init__(self, in_channels=3):
        super(GANFingerprintDetector, self).__init__()
        
        # Specialized high-pass filter to extract noise patterns
        self.init_filter = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=0, bias=False)
        # Initialize with high-pass filter kernels
        self._init_kernels()
        
        # Feature extraction pathway
        self.features = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # GAN classifier - identifies common GAN types
        self.gan_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 7)  # [real, stylegan, stylegan2, proggan, biggan, stargan, glow]
        )
        
        # Artifact features extractor
        self.artifact_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5)  # [frequency_artifacts, checkerboard_artifacts, boundary_artifacts, blending_artifacts, resizing_artifacts]
        )
    
    def _init_kernels(self):
        """Initialize with high-pass filter kernels to extract noise traces"""
        # SRM kernels (Steganalysis Rich Model) - good for extracting noise patterns
        kernel1 = torch.tensor([
            [-1, 2, -1],
            [2, -4, 2],
            [-1, 2, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        kernel2 = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        kernel3 = torch.tensor([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Make 3-channel kernels
        kernels = []
        for kernel in [kernel1, kernel2, kernel3]:
            kernel = kernel.repeat(3, 1, 1, 1)
            kernel = kernel.repeat(5, 1, 1, 1, 1)  # Create multiple variations
            kernel = kernel.view(-1, 3, 3, 3)
            kernels.append(kernel)
        
        # Create variation of kernels with 1 random padding
        kernels = torch.cat(kernels, dim=0)[:16]  # Take first 16 
        
        # Set kernels as non-trainable parameters
        self.init_filter.weight = nn.Parameter(kernels, requires_grad=False)
    
    def forward(self, x):
        """
        Detect GAN fingerprints in images.
        
        Args:
            x: Input tensor of shape [batch_size, 3, H, W]
            
        Returns:
            Dictionary with GAN detection scores and artifact features
        """
        # Apply initial filter to extract noise patterns
        x = self.init_filter(x)
        
        # Extract features
        features = self.features(x)
        
        # Classify GAN type
        gan_scores = self.gan_classifier(features)
        
        # Detect artifacts
        artifact_scores = self.artifact_detector(features)
        
        # Separate scores
        real_score = gan_scores[:, 0:1]
        gan_type_scores = gan_scores[:, 1:]
        
        # Separate artifact types
        frequency_artifacts = artifact_scores[:, 0:1]
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_transformer_layers)
        
        # Temporal attention for frame-level analysis
        self.temporal_attention = TemporalAttention(dim=transformer_dim)
        
        # Frame-level classification head
        self.frame_classifier = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(transformer_dim // 2, num_classes)
        )
        
        # Video-level classification head (after pooling)
        self.video_classifier = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(transformer_dim // 2, num_classes)
        )
        
        # Deepfake type classifier (optional)
        if detect_deepfake_type:
            self.deepfake_type_classifier = nn.Sequential(
                nn.Linear(transformer_dim, transformer_dim // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(transformer_dim // 2, num_deepfake_types)
            )
        else:
            self.deepfake_type_classifier = None
        
        # Edge optimization module for real-time inference
        self.edge_optimizer = None
        
    def extract_visual_features(self, video_frames):
        """Extract visual features from video frames."""
        batch_size, seq_len, c, h, w = video_frames.shape
        
        # Reshape for backbone processing
        video_flat = video_frames.view(batch_size * seq_len, c, h, w)
        
        # Process through visual backbone
        visual_features = self.visual_backbone(video_flat)
        
        # Reshape back to sequence
        visual_features = visual_features.view(batch_size, seq_len, -1)
        
        return visual_features
    
    def extract_audio_features(self, audio, audio_spectrogram=None):
        """Extract audio features from waveform and optional spectrogram."""
        batch_size = audio.shape[0]
        
        # Process audio waveform
        if isinstance(self.audio_backbone, (Wav2Vec2Model, type(AutoModel))):
            # Format for Wav2Vec2/HuBERT
            audio_input = audio.unsqueeze(1)  # [batch, 1, time]
            
            try:
                # Process through transformer-based audio backbone
                with torch.no_grad():
                    audio_outputs = self.audio_backbone(audio_input).last_hidden_state
                
                # Pool across time dimension
                audio_features = torch.mean(audio_outputs, dim=1)
            except Exception as e:
                print(f"Error processing audio: {e}")
                # Fallback to zeros
                audio_features = torch.zeros((batch_size, self.audio_feature_dim), device=audio.device)
        else:
            # Format for CNN (1D convolutions expect [batch, channels, time])
            audio_input = audio.unsqueeze(1)  # [batch, 1, time]
            
            # Process through CNN-based audio backbone
            audio_features = self.audio_backbone(audio_input)
        
        # Process spectrogram if available
        spec_features = None
        if self.use_spectrogram and audio_spectrogram is not None:
            spec_features = self.spectrogram_network(audio_spectrogram)
            
            # Concatenate with waveform features
            if spec_features is not None:
                audio_features = torch.cat([audio_features, spec_features], dim=1)
        
        return audio_features
    
    def extract_face_mesh_features(self, video_frames):
        """Extract face mesh features for detailed facial forensic analysis."""
        if not self.enable_face_mesh:
            return None
        
        try:
            # Process through face mesh module
            face_mesh_features = self.face_mesh_processor(video_frames)
            
            # Apply stats pooling
            face_mesh_pooled = self.face_mesh_pooling(face_mesh_features)
            
            return face_mesh_pooled
        except Exception as e:
            print(f"Error extracting face mesh features: {e}")
            return None
    
    def extract_physiological_features(self, video_frames):
        """Extract physiological signals (heart rate, breathing, etc.) from facial videos."""
        try:
            physio_outputs, physio_features = self.physio_detector(video_frames)
            return physio_outputs, physio_features
        except Exception as e:
            print(f"Error extracting physiological features: {e}")
            batch_size = video_frames.shape[0]
            return None, torch.zeros((batch_size, self.physio_dim), device=video_frames.device)
    
    def extract_ocular_features(self, video_frames):
        """Extract ocular behavioral cues from eye regions in facial videos."""
        try:
            # For simplicity, use the whole face as input
            # In a production system, you would extract eye regions first
            ocular_outputs, ocular_features = self.ocular_analyzer(video_frames, video_frames)
            return ocular_outputs, ocular_features
        except Exception as e:
            print(f"Error extracting ocular features: {e}")
            batch_size = video_frames.shape[0]
            return None, torch.zeros((batch_size, self.ocular_dim), device=video_frames.device)
    
    def analyze_lip_audio_sync(self, video_frames, audio_features):
        """Analyze synchronization between lip movements and audio."""
        try:
            # For simplicity, use whole frames
            # In a production system, you would extract lip regions first
            batch_size, seq_len = video_frames.shape[:2]
            
            # Create audio feature sequence matching video frames length
            if len(audio_features.shape) == 2:  # [batch, features]
                # Expand to sequence
                audio_seq = audio_features.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                audio_seq = audio_features
                
            sync_outputs, sync_features = self.lip_sync_analyzer(video_frames, audio_seq)
            return sync_outputs, sync_features
        except Exception as e:
            print(f"Error analyzing lip-audio sync: {e}")
            batch_size = video_frames.shape[0]
            return None, torch.zeros((batch_size, self.lip_sync_dim), device=video_frames.device)
    
    def forward(self, inputs):
        """
        Forward pass through the multi-modal deepfake detection model.
        
        Args:
            inputs: Dictionary containing:
                - video_frames: [batch_size, seq_len, 3, height, width]
                - audio: [batch_size, audio_length]
                - audio_spectrogram: Optional [batch_size, 1, freq_bins, time_bins]
                - face_embeddings: Optional facial embeddings
                - additional metadata
        
        Returns:
            Tuple of (logits, output_dict)
        """
        # Extract inputs
        video_frames = inputs.get('video_frames')
        audio = inputs.get('audio')
        audio_spectrogram = inputs.get('audio_spectrogram')
        face_embeddings = inputs.get('face_embeddings')
        
        batch_size, seq_len = video_frames.shape[:2]
        device = video_frames.device
        
        # Extract visual features
        visual_features = self.extract_visual_features(video_frames)
        
        # Extract audio features
        audio_features = self.extract_audio_features(audio, audio_spectrogram)
        
        # Extract face mesh features if enabled
        face_mesh_features = self.extract_face_mesh_features(video_frames)
        
        # NEW: Extract physiological features (heart rate, breathing, etc.)
        physio_outputs, physio_features = self.extract_physiological_features(video_frames)
        
        # NEW: Extract ocular behavioral features (eye movements, micro-expressions)
        ocular_outputs, ocular_features = self.extract_ocular_features(video_frames)
        
        # NEW: Analyze lip-audio synchronization
        lip_sync_outputs, lip_sync_features = self.analyze_lip_audio_sync(video_frames, audio_features)
        
        # Detect audio-visual synchronization issues
        av_sync_score = self.av_sync_detector(
            visual_features.reshape(batch_size, -1),
            audio_features
        )
        
        # Prepare for advanced fusion
        # Get the last frame's visual features as video-level representation
        video_level_visual = visual_features[:, -1, :]
        
        # Fuse modalities with advanced fusion module
        fused_features, alignment_scores = self.fusion_module(
            video_level_visual,
            audio_features,
            physio_features,
            ocular_features
        )
        
        # Apply temporal modeling with transformer
        transformed_features = self.transformer_encoder(visual_features)
        
        # Apply temporal attention
        temporal_features = self.temporal_attention(transformed_features)
        
        # Get video-level representation by average pooling across frames
        video_features = torch.mean(temporal_features, dim=1)
        
        # Frame-level classification (for visualization and temporal analysis)
        frame_logits = self.frame_classifier(temporal_features)
        
        # Video-level classification
        video_logits = self.video_classifier(video_features)
        
        # Deepfake type classification if enabled
        deepfake_type_logits = None
        if self.detect_deepfake_type and self.deepfake_type_classifier is not None:
            deepfake_type_logits = self.deepfake_type_classifier(video_features)
        
        # Create detailed output dictionary for interpretability
        outputs = {
            'frame_logits': frame_logits,
            'video_logits': video_logits,
            'deepfake_type': deepfake_type_logits,
            'av_sync_score': av_sync_score,
            'alignment_scores': alignment_scores,
            'visual_features': visual_features,
            'audio_features': audio_features,
            'physio_features': physio_features,
            'ocular_features': ocular_features,
            'lip_sync_features': lip_sync_features,
            'physiological_outputs': physio_outputs,
            'ocular_outputs': ocular_outputs,
            'lip_sync_outputs': lip_sync_outputs
        }
        
        # Generate explanation if enabled
        if self.enable_explainability:
            explanation = self.generate_explanation(inputs, outputs)
            outputs['explanation'] = explanation
        
        return video_logits, outputs
        
    def generate_explanation(self, inputs, outputs):
        """
        Generate human-interpretable explanation for the model's decision.
        
        Args:
            inputs: Dictionary of input tensors
            outputs: Dictionary of output tensors and scores
        
        Returns:
            Dictionary containing explanation data
        """
        try:
            # Get prediction and confidence
            video_logits = outputs['video_logits']
            probabilities = torch.softmax(video_logits, dim=1)
            predicted_class = torch.argmax(video_logits, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            
            # Initialize lists for issues found and highlighted regions
            issues_found = []
            highlighted_regions = []
            detection_scores = {}
            
            # Check for deepfake indicators
            
            # 1. Check physiological signal inconsistencies
            if 'physiological_outputs' in outputs and outputs['physiological_outputs'] is not None:
                physio_outputs = outputs['physiological_outputs']
                
                # Check for heart rate inconsistencies
                if 'hr_consistency' in physio_outputs:
                    hr_consistency = physio_outputs['hr_consistency'].mean().item()
                    detection_scores['heart_rate_consistency'] = hr_consistency
                    if hr_consistency < 0.7:
                        issues_found.append(f"Inconsistent heart rate patterns detected (score: {hr_consistency:.2f})")
                
                # Check for unnatural skin color variations
                if 'skin_naturality' in physio_outputs:
                    skin_naturality = physio_outputs['skin_naturality'].mean().item()
                    detection_scores['skin_naturality'] = skin_naturality
                    if skin_naturality < 0.65:
                        issues_found.append(f"Unnatural skin color variations detected (score: {skin_naturality:.2f})")
                
                # Check for unnatural breathing patterns
                if 'breathing_consistency' in physio_outputs and 'breath_holding' in physio_outputs:
                    breathing_consistency = physio_outputs['breathing_consistency'].mean().item()
                    breath_holding = physio_outputs['breath_holding'].mean().item()
                    detection_scores['breathing_consistency'] = breathing_consistency
                    detection_scores['breath_holding'] = breath_holding
                    if breathing_consistency < 0.6 or breath_holding > 0.7:
                        issues_found.append(f"Irregular breathing patterns detected (consistency: {breathing_consistency:.2f})")
            
            # 2. Check ocular behavioral inconsistencies
            if 'ocular_outputs' in outputs and outputs['ocular_outputs'] is not None:
                ocular_outputs = outputs['ocular_outputs']
                
                # Check for unnatural eye movements
                if 'saccade_score' in ocular_outputs and 'fixation_score' in ocular_outputs:
                    saccade_score = ocular_outputs['saccade_score'].mean().item()
                    fixation_score = ocular_outputs['fixation_score'].mean().item()
                    detection_scores['eye_movement_naturality'] = (saccade_score + fixation_score) / 2
                    if saccade_score < 0.6 or fixation_score < 0.6:
                        issues_found.append(f"Unnatural eye movement patterns detected (score: {saccade_score:.2f})")
                
                # Check for pupil dilation inconsistencies
                if 'dilation_consistency' in ocular_outputs:
                    dilation_consistency = ocular_outputs['dilation_consistency'].mean().item()
                    detection_scores['pupil_dilation_consistency'] = dilation_consistency
                    if dilation_consistency < 0.65:
                        issues_found.append(f"Inconsistent pupil dilation detected (score: {dilation_consistency:.2f})")
                
                # Check for micro-expression issues
                if 'micro_expression_score' in ocular_outputs:
                    micro_expr_score = ocular_outputs['micro_expression_score'].mean().item()
                    expression_naturalness = ocular_outputs.get('expression_naturalness', torch.tensor([0.5])).mean().item()
                    detection_scores['micro_expression_naturalness'] = expression_naturalness
                    if micro_expr_score > 0.7 or expression_naturalness < 0.6:
                        issues_found.append(f"Suspicious micro-expressions detected (score: {expression_naturalness:.2f})")
            
            # 3. Check lip-audio synchronization issues
            if 'lip_sync_outputs' in outputs and outputs['lip_sync_outputs'] is not None:
                lip_sync_outputs = outputs['lip_sync_outputs']
                
                if 'sync_score' in lip_sync_outputs:
                    sync_score = lip_sync_outputs['sync_score'].mean().item()
                    detection_scores['lip_sync_score'] = sync_score
                    if sync_score < 0.6:
                        issues_found.append(f"Lip synchronization issues detected (score: {sync_score:.2f})")
                
                if 'phoneme_match' in lip_sync_outputs:
                    phoneme_match = lip_sync_outputs['phoneme_match'].mean().item()
                    detection_scores['phoneme_viseme_match'] = phoneme_match
                    if phoneme_match < 0.55:
                        issues_found.append(f"Phoneme-viseme mismatch detected (score: {phoneme_match:.2f})")
            
            # 4. Check temporal inconsistencies from alignment scores
            if 'alignment_scores' in outputs:
                alignment_scores = outputs['alignment_scores']
                
                if 'temporal_alignment' in alignment_scores:
                    temp_alignment = alignment_scores['temporal_alignment'].mean().item()
                    detection_scores['temporal_alignment'] = temp_alignment
                    if temp_alignment < 0.6:
                        issues_found.append(f"Temporal inconsistencies detected between modalities (score: {temp_alignment:.2f})")
                
                if 'audio_video_drift' in alignment_scores:
                    av_drift = alignment_scores['audio_video_drift'].mean().item()
                    detection_scores['audio_video_drift'] = av_drift
                    if av_drift > 0.6:
                        issues_found.append(f"Audio-video drift detected (score: {av_drift:.2f})")
            
            # Calculate frame-level suspiciousness for visualization
            if 'frame_logits' in outputs:
                frame_logits = outputs['frame_logits']
                frame_probs = torch.softmax(frame_logits, dim=2)
                
                # Get probabilities for the fake class (index 1)
                fake_probs = frame_probs[:, :, 1]
                
                # Get frames with high fake probability
                for batch_idx in range(batch_idx):
                    for frame_idx in range(min(fake_probs.shape[1], 20)):  # Limit to 20 frames
                        fake_prob = fake_probs[batch_idx, frame_idx].item()
                        if fake_prob > 0.7:
                            highlighted_regions.append((batch_idx, frame_idx, fake_prob))
            
            # Create final explanation
            explanation = {
                'prediction': 'Fake' if predicted_class[0].item() == 1 else 'Real',
                'confidence': confidence[0].item(),
                'issues_found': issues_found,
                'highlighted_regions': highlighted_regions,
                'detection_scores': detection_scores
            }
            
            return explanation
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return {
                'prediction': 'Unknown',
                'confidence': 0.5,
                'issues_found': ['Error generating detailed explanation'],
                'highlighted_regions': [],
                'detection_scores': {}
            }
    
    def enable_edge_optimization(self, quantize=True, use_onnx=False, adaptive_resolution=True):
        """Enable edge optimization for real-time inference."""
        self.edge_optimizer = EdgeOptimizedDeepfakeDetector(
            self, 
            quantize=quantize,
            use_onnx=use_onnx,
            adaptive_resolution=adaptive_resolution
        )
        return self.edge_optimizer
    
    def get_attention_maps(self, inputs):
        """Get attention maps for visualization and explainability."""
        try:
            video_frames = inputs.get('video_frames')
            batch_size, seq_len = video_frames.shape[:2]
            
            # Extract visual features
            visual_features = self.extract_visual_features(video_frames)
            
            # Apply transformer to get attention weights
            # Note: This requires accessing internal attention weights from transformer
            # For simplicity, simulate attention maps based on frame logits
            
            with torch.no_grad():
                # Get frame-level predictions
                transformed_features = self.transformer_encoder(visual_features)
                frame_logits = self.frame_classifier(transformed_features)
                frame_probs = torch.softmax(frame_logits, dim=2)
                
                # Create attention maps based on frame probabilities
                attention_maps = torch.zeros((batch_size, seq_len, 224, 224), device=video_frames.device)
                
                for b in range(batch_size):
                    for t in range(seq_len):
                        # Use fake class probability as attention weight
                        fake_prob = frame_probs[b, t, 1].item()
                        
                        # Create a simple Gaussian attention map centered on the frame
                        h, w = 224, 224
                        y, x = torch.meshgrid(
                            torch.linspace(-1, 1, h),
                            torch.linspace(-1, 1, w)
                        )
                        
                        # Create 2D Gaussian
                        gaussian = torch.exp(-(x**2 + y**2) / 0.25)
                        
                        # Scale by fake probability
                        attention_maps[b, t] = gaussian.to(video_frames.device) * fake_prob
            
            return attention_maps
        except Exception as e:
            print(f"Error generating attention maps: {e}")
            return None

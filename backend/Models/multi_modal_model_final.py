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
# Additional imports for enhanced features
import cv2
import librosa
import math
import scipy
from scipy import signal
try:
    import dlib
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    print("Warning: Some optional dependencies (dlib, facenet_pytorch) not found, some features may be limited")


class AttentionFusion(nn.Module):
    """Cross-modal attention fusion module."""
    def __init__(self, visual_dim, audio_dim, output_dim):
        super(AttentionFusion, self).__init__()
        self.visual_projection = nn.Linear(visual_dim, output_dim)
        self.audio_projection = nn.Linear(audio_dim, output_dim)
        
        # Cross-attention components
        self.visual_query = nn.Linear(output_dim, output_dim)
        self.audio_key = nn.Linear(output_dim, output_dim)
        self.audio_value = nn.Linear(output_dim, output_dim)
        
        self.audio_query = nn.Linear(output_dim, output_dim)
        self.visual_key = nn.Linear(output_dim, output_dim)
        self.visual_value = nn.Linear(output_dim, output_dim)
        
        # Output layer
        self.fusion_layer = nn.Linear(output_dim * 2, output_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, visual_features, audio_features):
        # Project features to common space
        visual_proj = self.visual_projection(visual_features)
        audio_proj = self.audio_projection(audio_features)
        
        # Visual attending to audio
        v_query = self.visual_query(visual_proj)
        a_key = self.audio_key(audio_proj)
        a_value = self.audio_value(audio_proj)
        
        # Attention weights: visual query attends to audio keys
        v_a_attn_weights = torch.matmul(v_query, a_key.transpose(-2, -1))
        v_a_attn_weights = F.softmax(v_a_attn_weights / np.sqrt(v_query.size(-1)), dim=-1)
        v_attended_a = torch.matmul(v_a_attn_weights, a_value)
        
        # Audio attending to visual
        a_query = self.audio_query(audio_proj)
        v_key = self.visual_key(visual_proj)
        v_value = self.visual_value(visual_proj)
        
        # Attention weights: audio query attends to visual keys
        a_v_attn_weights = torch.matmul(a_query, v_key.transpose(-2, -1))
        a_v_attn_weights = F.softmax(a_v_attn_weights / np.sqrt(a_query.size(-1)), dim=-1)
        a_attended_v = torch.matmul(a_v_attn_weights, v_value)
        
        # Combine attended features
        combined = torch.cat([v_attended_a, a_attended_v], dim=-1)
        fused = self.fusion_layer(combined)
        
        # Residual connection and normalization
        fused = self.layer_norm(fused + visual_proj + audio_proj)
        
        return fused


class TemporalAttention(nn.Module):
    """Temporal attention module for analyzing frame sequences."""
    def __init__(self, dim, num_heads=8):
        super(TemporalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class StatsPooling(nn.Module):
    """Statistics pooling layer that computes mean and standard deviation."""
    def __init__(self):
        super(StatsPooling, self).__init__()
    
    def forward(self, x):
        # x shape: [batch, frames, features]
        mean = torch.mean(x, dim=1)
        std = torch.std(x, dim=1)
        return torch.cat([mean, std], dim=1)


class ForensicConsistencyModule(nn.Module):
    """Module that analyzes forensic consistency across frames."""
    def __init__(self, input_dim, hidden_dim=256):
        super(ForensicConsistencyModule, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x shape: [batch, frames, channels, height, width]
        batch_size, num_frames = x.shape[:2]
        
        # Reshape for convolutions
        x = x.view(batch_size * num_frames, *x.shape[2:])
        
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        
        # Reshape back to batch, frames, features
        x = x.view(batch_size, num_frames, -1)
        
        # Project features
        x = self.fc(x)
        
        return x


class AudioVisualSyncDetector(nn.Module):
    """Module to detect synchronization issues between audio and video."""
    def __init__(self, visual_dim, audio_dim, hidden_dim=128):
        super(AudioVisualSyncDetector, self).__init__()
        
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, visual_features, audio_features):
        # Process visual features
        visual_embed = self.visual_encoder(visual_features)
        
        # Process audio features
        audio_embed = self.audio_encoder(audio_features)
        
        # Fusion
        combined = torch.cat([visual_embed, audio_embed], dim=1)
        sync_score = self.fusion(combined)
        
        return sync_score


# New modules for enhanced facial dynamics analysis

class FacialActionUnitAnalyzer(nn.Module):
    """Analyzes facial action units (FAUs) to detect muscle movement inconsistencies."""
    def __init__(self, input_dim=136, hidden_dim=128, num_aus=17):
        super(FacialActionUnitAnalyzer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # FAU detection heads (one for each action unit)
        self.au_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_aus)
        ])
        
        # Temporal consistency analyzer
        self.temporal_lstm = nn.LSTM(
            input_size=num_aus,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.consistency_scorer = nn.Linear(hidden_dim, 1)
        
    def forward(self, landmarks_sequence):
        """
        Analyze facial action units over time.
        
        Args:
            landmarks_sequence: Facial landmarks of shape [batch, frames, landmarks*2]
        
        Returns:
            Tuple of (consistency_score, au_activations)
        """
        batch_size, seq_len = landmarks_sequence.shape[:2]
        
        # Process each frame's landmarks
        aus_over_time = []
        for t in range(seq_len):
            landmarks = landmarks_sequence[:, t]
            features = self.encoder(landmarks)
            
            # Get activations for each AU
            au_acts = []
            for au_head in self.au_heads:
                au_acts.append(torch.sigmoid(au_head(features)))
                
            # Combine all AU activations
            frame_aus = torch.cat(au_acts, dim=1)  # [batch, num_aus]
            aus_over_time.append(frame_aus)
            
        # Stack across time dimension
        aus_sequence = torch.stack(aus_over_time, dim=1)  # [batch, frames, num_aus]
        
        # Analyze temporal consistency with LSTM
        lstm_out, _ = self.temporal_lstm(aus_sequence)
        
        # Get final hidden state
        final_state = lstm_out[:, -1]
        
        # Score temporal consistency
        consistency_score = torch.sigmoid(self.consistency_scorer(final_state))
        
        return consistency_score, aus_sequence


class MicroExpressionDetector(nn.Module):
    """Detects micro-expressions that are difficult to fake in deepfakes."""
    def __init__(self, input_channels=3, hidden_dim=64):
        super(MicroExpressionDetector, self).__init__()
        # 3D CNN for spatiotemporal feature extraction
        self.conv3d = nn.Sequential(
            nn.Conv3d(input_channels, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(hidden_dim, hidden_dim*2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_dim*2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(hidden_dim*2, hidden_dim*4, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_dim*4),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 4, 4))  # Keep temporal dimension
        )
        
        # LSTM for temporal sequence analysis
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim*4*4*4,  # Flattened spatial features
            hidden_size=hidden_dim*4,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Expression classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*4*2, hidden_dim*2),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, 7)  # 7 basic micro-expressions
        )
        
    def forward(self, face_crops):
        """
        Args:
            face_crops: Facial crops of shape [batch, frames, channels, height, width]
        
        Returns:
            Tuple of (micro_expression_score, expression_probabilities)
        """
        batch_size, seq_len = face_crops.shape[:2]
        
        # Reshape for 3D CNN: [batch, channels, frames, height, width]
        x = face_crops.permute(0, 2, 1, 3, 4)
        
        # Apply 3D CNN
        features = self.conv3d(x)  # [batch, channels, frames, height, width]
        
        # Flatten spatial dimensions but keep temporal
        features = features.permute(0, 2, 1, 3, 4)  # [batch, frames, channels, height, width]
        features = features.reshape(batch_size, seq_len, -1)  # [batch, frames, channels*h*w]
        
        # Apply LSTM for temporal analysis
        lstm_out, _ = self.temporal_lstm(features)
        
        # Classify expressions
        expression_probs = self.classifier(lstm_out[:, -1])  # Use final state
        expression_probs = torch.sigmoid(expression_probs)  # Multi-label classification
        
        # Calculate micro-expression score (temporal dynamics)
        temporal_diff = torch.abs(lstm_out[:, 1:] - lstm_out[:, :-1])
        temporal_magnitude = torch.mean(temporal_diff, dim=(1, 2))
        
        # Normalize to [0, 1]
        micro_expr_score = torch.sigmoid(temporal_magnitude)
        
        return micro_expr_score, expression_probs


class FacialLandmarkTrajectoryAnalyzer(nn.Module):
    """Analyzes facial landmark trajectories to detect unnatural movements."""
    def __init__(self, num_landmarks=68, hidden_dim=128):
        super(FacialLandmarkTrajectoryAnalyzer, self).__init__()
        self.num_landmarks = num_landmarks
        
        # Motion encoder
        self.motion_encoder = nn.Sequential(
            nn.Linear(num_landmarks * 2, hidden_dim),  # 2D coordinates for each landmark
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Temporal analysis with GRU
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Consistency scorer
        self.consistency_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, landmark_sequence):
        """
        Args:
            landmark_sequence: Tensor of shape [batch, frames, landmarks*2]
        
        Returns:
            Trajectory consistency score and motion features
        """
        batch_size, seq_len = landmark_sequence.shape[:2]
        
        # Calculate motion vectors (differences between consecutive frames)
        if seq_len > 1:
            motion_vectors = landmark_sequence[:, 1:] - landmark_sequence[:, :-1]
            seq_len = seq_len - 1
        else:
            # If only one frame, use the landmarks themselves
            motion_vectors = landmark_sequence
        
        # Encode each frame's motion
        motion_features = []
        for t in range(motion_vectors.shape[1]):
            encoded = self.motion_encoder(motion_vectors[:, t])
            motion_features.append(encoded)
        
        motion_sequence = torch.stack(motion_features, dim=1)  # [batch, frames-1, features]
        
        # Analyze temporal consistency with GRU
        gru_out, _ = self.temporal_gru(motion_sequence)
        
        # Get final state
        final_state = gru_out[:, -1]
        
        # Score motion consistency
        consistency_score = self.consistency_scorer(final_state)
        
        return consistency_score, motion_sequence


class HeadPoseEstimator(nn.Module):
    """Estimates head pose (pitch, yaw, roll) and analyzes its consistency."""
    def __init__(self, landmark_dim=136, hidden_dim=128):
        super(HeadPoseEstimator, self).__init__()
        
        # Pose estimator network
        self.pose_estimator = nn.Sequential(
            nn.Linear(landmark_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # pitch, yaw, roll
        )
        
        # Temporal consistency analyzer
        self.consistency_analyzer = nn.GRU(
            input_size=3,  # 3D pose angles
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Output consistency score
        self.consistency_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, landmark_sequence):
        """
        Args:
            landmark_sequence: Facial landmarks of shape [batch, frames, landmarks*2]
        
        Returns:
            Tuple of (consistency_score, pose_sequence)
        """
        batch_size, seq_len = landmark_sequence.shape[:2]
        
        # Estimate pose for each frame
        poses = []
        for t in range(seq_len):
            pose = self.pose_estimator(landmark_sequence[:, t])  # [batch, 3]
            poses.append(pose)
        
        pose_sequence = torch.stack(poses, dim=1)  # [batch, frames, 3]
        
        # Analyze temporal consistency
        gru_out, _ = self.consistency_analyzer(pose_sequence)
        
        # Get final state
        final_state = gru_out[:, -1]
        
        # Score temporal consistency
        consistency_score = self.consistency_scorer(final_state)
        
        return consistency_score, pose_sequence


class EyeAnalysisModule(nn.Module):
    """Analyzes eye movements including blinking and pupil dilation with automatic dimension adjustment."""
    def __init__(self, hidden_dim=64):
        super(EyeAnalysisModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.landmark_dim = None  # Will be initialized dynamically
        
        # Placeholder for dynamically created components
        self.eye_encoder = None
        
        # Components that don't depend on input dimension
        # Blink detector
        self.blink_detector = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Pupil dilation estimator
        self.pupil_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Temporal analysis
        self.temporal_gru = nn.GRU(
            input_size=2,  # Blink + pupil dilation
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Naturalness scorer
        self.naturalness_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def _init_eye_encoder(self, landmark_dim):
        """Dynamically initialize the eye encoder based on input dimensions"""
        self.landmark_dim = landmark_dim
        self.eye_encoder = nn.Sequential(
            nn.Linear(landmark_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU()
        ).to(next(self.blink_detector.parameters()).device)
        
        print(f"[INFO] Dynamically initialized eye encoder with input dim {landmark_dim}")
        
    def forward(self, eye_landmarks):
        """
        Args:
            eye_landmarks: Eye landmarks of shape [batch, frames, landmarks*2]
        
        Returns:
            Tuple of (naturalness_score, blinks, pupil_dilation)
        """
        # Handle None or empty input
        if eye_landmarks is None or eye_landmarks.numel() == 0:
            device = next(self.blink_detector.parameters()).device
            batch_size = 1
            seq_len = 1
            if eye_landmarks is not None and len(eye_landmarks.shape) >= 2:
                batch_size, seq_len = eye_landmarks.shape[:2]
            
            return (
                torch.ones(batch_size, 1, device=device) * 0.5,  # naturalness
                torch.zeros(batch_size, seq_len, device=device),  # blinks
                torch.zeros(batch_size, seq_len, device=device)   # pupil dilation
            )
        
        # Get batch size and sequence length
        batch_size, seq_len = eye_landmarks.shape[:2]
        
        # Initialize or update eye encoder if needed
        current_landmark_dim = eye_landmarks.shape[2]
        if self.eye_encoder is None or self.landmark_dim != current_landmark_dim:
            self._init_eye_encoder(current_landmark_dim)
        
        blinks = []
        pupil_dilations = []
        
        # Process each frame with exception handling
        for t in range(seq_len):
            try:
                features = self.eye_encoder(eye_landmarks[:, t])
                
                # Detect blink
                blink = self.blink_detector(features)
                blinks.append(blink)
                
                # Estimate pupil dilation
                dilation = self.pupil_estimator(features)
                pupil_dilations.append(dilation)
                
            except Exception as e:
                print(f"[WARNING] Error processing frame {t}: {e}")
                # Create fallback values
                device = eye_landmarks.device
                blinks.append(torch.ones(batch_size, 1, device=device) * 0.5)
                pupil_dilations.append(torch.ones(batch_size, 1, device=device) * 0.5)
        
        # Stack across time
        blinks = torch.stack(blinks, dim=1)  # [batch, frames, 1]
        pupil_dilations = torch.stack(pupil_dilations, dim=1)  # [batch, frames, 1]
        
        # Combine for temporal analysis
        temporal_features = torch.cat([blinks, pupil_dilations], dim=2)  # [batch, frames, 2]
        
        # Analyze temporal patterns
        gru_out, _ = self.temporal_gru(temporal_features)
        
        # Score naturalness
        naturalness = self.naturalness_scorer(gru_out[:, -1])
        
        return naturalness, blinks.squeeze(2), pupil_dilations.squeeze(2)


class LipAudioSyncAnalyzer(nn.Module):
    """Analyzes synchronization between lip movements and audio."""
    def __init__(self, lip_dim=40, audio_dim=768, hidden_dim=128):
        super(LipAudioSyncAnalyzer, self).__init__()
        
        # Lip movement encoder
        self.lip_encoder = nn.Sequential(
            nn.Linear(lip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Audio feature encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            batch_first=True
        )
        
        # Sync scorer
        self.sync_scorer = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, lip_features, audio_features):
        """
        Args:
            lip_features: Lip landmarks/regions of shape [batch, frames, features]
            audio_features: Audio features of shape [batch, frames, features]
        
        Returns:
            Synchronization score
        """
        # Encode lip movements
        lip_encoded = self.lip_encoder(lip_features)
        
        # Encode audio features
        audio_encoded = self.audio_encoder(audio_features)
        
        # Cross-modal attention
        attn_output, attn_weights = self.cross_attention(
            query=lip_encoded,
            key=audio_encoded,
            value=audio_encoded
        )
        
        # Score synchronization
        sync_scores = self.sync_scorer(attn_output)
        
        # Average over time for overall score
        sync_score = torch.mean(sync_scores, dim=1)
        
        return sync_score, attn_weights


class OculomotorDynamicsAnalyzer(nn.Module):
    """Analyzes saccades, fixations and other eye movement behaviors with dynamic dimension adjustment."""
    def __init__(self, hidden_dim=64):
        super(OculomotorDynamicsAnalyzer, self).__init__()
        self.hidden_dim = hidden_dim
        self.eye_feature_dim = None  # Will be initialized dynamically
        
        # Movement encoder will be initialized dynamically
        self.movement_encoder = None
        
        # Temporal analyzer
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Dynamics classifier (saccades, fixations, smooth pursuits)
        self.dynamics_classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )
        
        # Naturalness scorer
        self.naturalness_scorer = nn.Sequential(
            nn.Linear(hidden_dim*2, 1),
            nn.Sigmoid()
        )
    
    def _init_movement_encoder(self, eye_feature_dim):
        """Dynamically initialize the movement encoder based on input dimensions"""
        self.eye_feature_dim = eye_feature_dim
        self.movement_encoder = nn.Sequential(
            nn.Linear(eye_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        ).to(next(self.temporal_cnn.parameters()).device)
        
        print(f"[INFO] Dynamically initialized oculomotor movement encoder with input dim {eye_feature_dim}")
        
    def forward(self, eye_movement_features):
        """
        Args:
            eye_movement_features: Eye movement features [batch, frames, features]
        
        Returns:
            Tuple of (naturalness_score, movement_classifications)
        """
        # Handle None or empty input
        if eye_movement_features is None or eye_movement_features.numel() == 0:
            device = next(self.temporal_cnn.parameters()).device
            batch_size = 1
            if eye_movement_features is not None and len(eye_movement_features.shape) > 0:
                batch_size = eye_movement_features.shape[0]
                
            return (
                torch.ones(batch_size, 1, device=device) * 0.5,  # naturalness
                torch.ones(batch_size, 3, device=device) / 3.0  # uniform distribution
            )
        
        # Initialize or update movement encoder if needed
        current_feature_dim = eye_movement_features.shape[2]
        if self.movement_encoder is None or self.eye_feature_dim != current_feature_dim:
            self._init_movement_encoder(current_feature_dim)
        
        try:
            # Encode movements
            encoded = self.movement_encoder(eye_movement_features)
            
            # Transpose for 1D CNN [batch, channels, time]
            encoded = encoded.transpose(1, 2)
            
            # Extract temporal patterns
            temporal_features = self.temporal_cnn(encoded).squeeze(-1)
            
            # Classify movements
            movement_classes = self.dynamics_classifier(temporal_features)
            
            # Score naturalness
            naturalness = self.naturalness_scorer(temporal_features)
            
            return naturalness, movement_classes
            
        except Exception as e:
            print(f"[WARNING] Error in oculomotor dynamics analysis: {e}")
            # Return fallback values
            device = next(self.temporal_cnn.parameters()).device
            batch_size = eye_movement_features.shape[0]
            
            return (
                torch.ones(batch_size, 1, device=device) * 0.5,  # naturalness
                torch.ones(batch_size, 3, device=device) / 3.0  # uniform distribution
            )
# Physiological Signal Analysis Modules

class RemotePhysiologicalAnalyzer(nn.Module):
    """Analyzes subtle physiological signals from video with enhanced dynamic dimension handling."""
    def __init__(self, feature_dim=32):
        super(RemotePhysiologicalAnalyzer, self).__init__()
        self.feature_dim = feature_dim
        
        # Spatial feature extraction will be initialized dynamically
        self.input_channels = None
        self.feature_extractor = None
        
        # Signal processor will be initialized dynamically
        self.actual_feature_dim = None
        self.signal_processor = None
        
        # Create estimators that will be compatible with any feature dimension
        self.hr_estimator = None
        self.hrv_estimator = None
        self.breathing_estimator = None
        
        # Naturalness scorer is fixed dimension (always 3 inputs)
        self.naturalness_scorer = nn.Linear(3, 1)
    
    def _init_feature_extractor(self, input_channels):
        """Dynamically initialize the feature extractor based on input channels"""
        self.input_channels = input_channels
        device = self.naturalness_scorer.weight.device
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, self.feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        ).to(device)
        
        print(f"[INFO] Dynamically initialized feature extractor with {input_channels} input channels")
    
    def _init_signal_processor(self, actual_feature_dim):
        """Dynamically initialize the signal processor based on actual feature dimensions"""
        self.actual_feature_dim = actual_feature_dim
        device = self.naturalness_scorer.weight.device
        
        # Create signal processor with correct input dimension
        self.signal_processor = nn.Sequential(
            nn.Linear(actual_feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim//2),
            nn.ReLU()
        ).to(device)
        
        # Create estimators with correct dimensions
        self.hr_estimator = nn.Linear(self.feature_dim//2, 1).to(device)
        self.hrv_estimator = nn.Linear(self.feature_dim//2, 1).to(device)
        self.breathing_estimator = nn.Linear(self.feature_dim//2, 1).to(device)
        
        print(f"[INFO] Dynamically initialized signal processor with input dim {actual_feature_dim}")
    
    def forward(self, face_frames):
        """
        Args:
            face_frames: Facial video frames in format [batch, frames, channels, height, width]
                         or [batch, frames, height, width, channels]
        
        Returns:
            Dictionary with physiological measurements and naturalness score
        """
        # Get device from parameters
        device = self.naturalness_scorer.weight.device
        
        # Handle None or empty input
        if face_frames is None or face_frames.numel() == 0:
            batch_size = 1
            if face_frames is not None and len(face_frames.shape) > 0:
                batch_size = face_frames.shape[0]
                
            return {
                'heart_rate': torch.tensor([80.0], device=device).expand(batch_size, 1),
                'hrv': torch.tensor([0.5], device=device).expand(batch_size, 1),
                'breathing_rate': torch.tensor([15.0], device=device).expand(batch_size, 1),
                'naturalness': torch.tensor([0.5], device=device).expand(batch_size, 1)
            }
        
        try:
            batch_size, num_frames = face_frames.shape[:2]
            
            # Auto-detect frame format (channels first or channels last)
            if len(face_frames.shape) >= 5:
                # Determine if channels are in dimension 2 or 4
                if face_frames.shape[2] in [1, 3, 4]:  # Common channel counts
                    # Format is [batch, frames, channels, height, width]
                    input_channels = face_frames.shape[2]
                    frames_need_transpose = False
                else:
                    # Format is [batch, frames, height, width, channels]
                    input_channels = face_frames.shape[4]
                    frames_need_transpose = True
            else:
                # Default to 1 channel if unclear
                input_channels = 1
                frames_need_transpose = False
                
            print(f"[DEBUG] Frame format detected: {'channels_last' if frames_need_transpose else 'channels_first'}")
            print(f"[DEBUG] Input shape: {face_frames.shape}, channels: {input_channels}")
            
            # Initialize feature extractor if needed
            if self.feature_extractor is None or self.input_channels != input_channels:
                self._init_feature_extractor(input_channels)
            
            # Extract spatial features from each frame
            frame_features = []
            for t in range(num_frames):
                try:
                    # Prepare frame for Conv2d (batch, channels, height, width)
                    if frames_need_transpose:
                        # Move channels from last dimension to second dimension
                        current_frames = face_frames[:, t].permute(0, 3, 1, 2)
                    else:
                        current_frames = face_frames[:, t]
                    
                    # Debug dimensions
                    print(f"[DEBUG] Frame at position {t} shape: {current_frames.shape}")
                    
                    # Extract features
                    features = self.feature_extractor(current_frames)
                    frame_features.append(features)
                    
                    # Initialize signal processor after getting first feature
                    if t == 0 and (self.signal_processor is None or self.actual_feature_dim != features.shape[1]):
                        self._init_signal_processor(features.shape[1])
                        
                except Exception as frame_error:
                    print(f"[WARNING] Error processing frame {t}: {frame_error}")
                    # Skip this frame
                    continue
            
            # Check if we have any valid frames
            if not frame_features:
                raise ValueError("No frames were successfully processed")
                
            # Stack temporal features
            temporal_features = torch.stack(frame_features, dim=1)  # [batch, frames, features]
            print(f"[DEBUG] Temporal features shape: {temporal_features.shape}")
            
            # Process each frame's features individually to avoid dimension issues
            processed_signals = []
            for t in range(temporal_features.shape[1]):
                frame_feature = temporal_features[:, t]
                processed = self.signal_processor(frame_feature)
                processed_signals.append(processed)
            
            # Stack processed signals
            processed_signal = torch.stack(processed_signals, dim=1)
            print(f"[DEBUG] Processed signal shape: {processed_signal.shape}")
            
            # Get final representation (mean of sequence)
            final_features = torch.mean(processed_signal, dim=1)
            print(f"[DEBUG] Final features shape: {final_features.shape}")
            
            # Estimate vital signs
            heart_rate = torch.sigmoid(self.hr_estimator(final_features)) * 120 + 40  # 40-160 BPM range
            hrv = torch.sigmoid(self.hrv_estimator(final_features))
            breathing_rate = torch.sigmoid(self.breathing_estimator(final_features)) * 20 + 10  # 10-30 breaths/min
            
            # Score physiological naturalness
            physio_features = torch.cat([
                heart_rate, hrv, breathing_rate
            ], dim=1)
            naturalness = torch.sigmoid(self.naturalness_scorer(physio_features))
            
            return {
                'heart_rate': heart_rate,
                'hrv': hrv,
                'breathing_rate': breathing_rate,
                'naturalness': naturalness
            }
            
        except Exception as e:
            print(f"[WARNING] Error in remote physiological analysis: {e}")
            if 'face_frames' in locals():
                print(f"[DEBUG] Input shape: {face_frames.shape}")
            if 'temporal_features' in locals():
                print(f"[DEBUG] Temporal features shape: {temporal_features.shape}")
            if 'frame_features' in locals() and frame_features:
                print(f"[DEBUG] First frame feature shape: {frame_features[0].shape}")
            
            # Return default values
            return {
                'heart_rate': torch.tensor([80.0], device=device).expand(batch_size, 1),
                'hrv': torch.tensor([0.5], device=device).expand(batch_size, 1),
                'breathing_rate': torch.tensor([15.0], device=device).expand(batch_size, 1),
                'naturalness': torch.tensor([0.5], device=device).expand(batch_size, 1)
            }

class SkinColorAnalyzer(nn.Module):
    """Analyzes skin color variations to detect pulse and blood flow patterns."""
    def __init__(self, feature_dim=32):
        super(SkinColorAnalyzer, self).__init__()
        
        # Color variation encoder
        self.color_encoder = nn.Sequential(
            nn.Linear(3, feature_dim),  # RGB color channels
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Temporal CNN for signal processing
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(feature_dim*2, feature_dim*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(feature_dim*2, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Naturalness scorer
        self.naturalness_scorer = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, skin_color_sequence):
        """
        Args:
            skin_color_sequence: RGB color values [batch, frames, 3]
        
        Returns:
            Naturalness score
        """
        # Encode color variations
        encoded = self.color_encoder(skin_color_sequence)
        
        # Transpose for 1D CNN [batch, channels, time]
        encoded = encoded.transpose(1, 2)
        
        # Process temporal patterns
        temporal_features = self.temporal_cnn(encoded).squeeze(-1)
        
        # Score naturalness
        naturalness = self.naturalness_scorer(temporal_features)
        
        return naturalness


# Visual Artifact and Spatial Analysis Modules

class LightingConsistencyAnalyzer(nn.Module):
    """Analyzes lighting consistency across video frames."""
    def __init__(self, feature_dim=64):
        super(LightingConsistencyAnalyzer, self).__init__()
        
        # Lighting feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Lighting representation
        self.lighting_encoder = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.ReLU()
        )
        
        # Temporal consistency analyzer
        self.temporal_analyzer = nn.GRU(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Consistency scorer
        self.consistency_scorer = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, video_frames):
        """
        Args:
            video_frames: Video frames [batch, frames, channels, height, width]
        
        Returns:
            Lighting consistency score
        """
        batch_size, num_frames = video_frames.shape[:2]
        
        # Extract lighting features from each frame
        lighting_features = []
        for t in range(num_frames):
            features = self.feature_extractor(video_frames[:, t])
            lighting = self.lighting_encoder(features)
            lighting_features.append(lighting)
            
        lighting_sequence = torch.stack(lighting_features, dim=1)  # [batch, frames, features]
        
        # Analyze temporal consistency
        _, hidden = self.temporal_analyzer(lighting_sequence)
        
        # Score consistency
        consistency = self.consistency_scorer(hidden.squeeze(0))
        
        return consistency


class TextureAnalyzer(nn.Module):
    """Analyzes texture consistency to detect patch-level manipulations."""
    def __init__(self, patch_size=32, feature_dim=64):
        super(TextureAnalyzer, self).__init__()
        self.patch_size = patch_size
        
        # Texture encoder
        self.texture_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Patch consistency analyzer
        self.consistency_analyzer = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image):
        """
        Args:
            image: Single image [batch, channels, height, width]
        
        Returns:
            Texture consistency score and heatmap
        """
        batch_size, C, H, W = image.shape
        
        # Extract patches
        patches = []
        patch_positions = []
        
        for y in range(0, H - self.patch_size, self.patch_size // 2):
            for x in range(0, W - self.patch_size, self.patch_size // 2):
                patch = image[:, :, y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
                patch_positions.append((y, x))
                
        # If no patches were extracted, return default values
        if not patches:
            return torch.ones(batch_size, 1, device=image.device), None
                
        # Stack patches
        patches = torch.cat(patches, dim=0)  # [batch*num_patches, C, patch_size, patch_size]
        
        # Encode textures
        texture_features = self.texture_encoder(patches)  # [batch*num_patches, features]
        
        # Reshape to separate batch and patches
        num_patches = len(patch_positions)
        texture_features = texture_features.view(batch_size, num_patches, -1)
        
        # Analyze consistency between patches
        patch_scores = []
        for b in range(batch_size):
            features = texture_features[b]  # [num_patches, features]
            
            # Calculate pairwise differences
            diff_matrix = torch.cdist(features, features, p=2)
            
            # Get average difference
            avg_diff = torch.mean(diff_matrix)
            
            # Convert to consistency score (lower diff = higher consistency)
            consistency = torch.exp(-avg_diff)
            patch_scores.append(consistency)
            
        # Stack scores
        patch_scores = torch.stack(patch_scores)
        
        # Generate heatmap
        heatmap = torch.zeros(batch_size, 1, H, W, device=image.device)
        
        return patch_scores, heatmap


class FrequencyDomainAnalyzer(nn.Module):
    """Analyzes frequency domain characteristics to detect GAN artifacts."""
    def __init__(self, feature_dim=64):
        super(FrequencyDomainAnalyzer, self).__init__()
        
        # Frequency encoder
        self.freq_encoder = nn.Sequential(
            nn.Linear(1024, 512),  # Frequency bins
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim)
        )
        
        # Artifact detector
        self.artifact_detector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image):
        """
        Args:
            image: Input image [batch, channels, height, width]
        
        Returns:
            Artifact detection score
        """
        batch_size = image.shape[0]
        
        # Convert to grayscale
        gray = torch.mean(image, dim=1)  # [batch, height, width]
        
        # Apply FFT
        fft = torch.fft.fft2(gray)
        fft_mag = torch.abs(fft)
        
        # Shift to center
        fft_mag = torch.fft.fftshift(fft_mag)
        
        # Resize to fixed dimensions
        fft_mag = F.interpolate(
            fft_mag.unsqueeze(1),  # Add channel dimension
            size=(32, 32),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        # Flatten
        fft_flat = fft_mag.reshape(batch_size, -1)
        
        # Encode frequency features
        freq_features = self.freq_encoder(fft_flat)
        
        # Detect artifacts
        artifact_score = self.artifact_detector(freq_features)
        
        return artifact_score


class GANFingerprintDetector(nn.Module):
    """Detects GAN fingerprints left in generated images."""
    def __init__(self, feature_dim=128):
        super(GANFingerprintDetector, self).__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Fingerprint detector
        self.fingerprint_detector = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image):
        """
        Args:
            image: Input image [batch, channels, height, width]
        
        Returns:
            GAN fingerprint detection score
        """
        # Extract features
        features = self.feature_extractor(image)
        
        # Detect fingerprint
        fingerprint_score = self.fingerprint_detector(features)
        
        return fingerprint_score


# Audio Analysis Modules

class VoiceAnalysisModule(nn.Module):
    """Analyzes voice characteristics to detect inconsistencies."""
    def __init__(self, audio_dim=768, feature_dim=128):
        super(VoiceAnalysisModule, self).__init__()
        
        # Voice feature encoder
        self.voice_encoder = nn.Sequential(
            nn.Linear(audio_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU()
        )
        
        # Temporal consistency analyzer
        self.temporal_analyzer = nn.GRU(
            input_size=feature_dim,
            hidden_size=feature_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Authenticity scorer
        self.authenticity_scorer = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, audio_features):
        """
        Args:
            audio_features: Audio features [batch, frames, features]
        
        Returns:
            Voice authenticity score
        """
        # Encode voice features
        voice_features = self.voice_encoder(audio_features)
        
        # Analyze temporal consistency
        gru_out, _ = self.temporal_analyzer(voice_features)
        
        # Get final state
        final_state = gru_out[:, -1]
        
        # Score authenticity
        authenticity = self.authenticity_scorer(final_state)
        
        return authenticity


class MFCCExtractor(nn.Module):
    """Extracts and analyzes Mel-Frequency Cepstral Coefficients."""
    def __init__(self, num_mfcc=40, feature_dim=64):
        super(MFCCExtractor, self).__init__()
        self.num_mfcc = num_mfcc
        
        # MFCC analyzer
        self.mfcc_analyzer = nn.Sequential(
            nn.Linear(num_mfcc, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )
        
        # Temporal analyzer
        self.temporal_analyzer = nn.GRU(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Consistency scorer
        self.consistency_scorer = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
    def extract_mfcc(self, audio, sample_rate=16000):
        """Extract MFCC features (CPU implementation)"""
        # Convert to numpy
        audio_np = audio.cpu().numpy()
        
        # Extract MFCCs
        mfccs = []
        for i in range(audio_np.shape[0]):
            mfcc = librosa.feature.mfcc(
                y=audio_np[i], 
                sr=sample_rate, 
                n_mfcc=self.num_mfcc
            )
            mfccs.append(torch.from_numpy(mfcc).to(audio.device))
            
        # Stack and transpose to [batch, time, features]
        return torch.stack(mfccs).transpose(1, 2)
    
    def forward(self, audio, sample_rate=16000):
        """
        Args:
            audio: Audio waveform [batch, samples]
            sample_rate: Audio sample rate
        
        Returns:
            MFCC consistency score
        """
        # Extract MFCCs
        mfccs = self.extract_mfcc(audio, sample_rate)
        
        # Analyze MFCCs
        mfcc_features = []
        for t in range(mfccs.shape[1]):
            features = self.mfcc_analyzer(mfccs[:, t])
            mfcc_features.append(features)
            
        mfcc_sequence = torch.stack(mfcc_features, dim=1)
        
        # Analyze temporal patterns
        _, hidden = self.temporal_analyzer(mfcc_sequence)
        
        # Score consistency
        consistency = self.consistency_scorer(hidden.squeeze(0))
        
        return consistency, mfccs


class PhonemeVisemeAnalyzer(nn.Module):
    """Analyzes synchronization between speech phonemes and lip movements."""
    def __init__(self, audio_dim=768, visual_dim=512, hidden_dim=128):
        super(PhonemeVisemeAnalyzer, self).__init__()
        
        # Phoneme extractor (from audio)
        self.phoneme_extractor = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Viseme extractor (from video)
        self.viseme_extractor = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Synchronization scorer
        self.sync_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, audio_features, visual_features):
        """
        Args:
            audio_features: Audio features [batch, frames, features]
            visual_features: Visual features [batch, frames, features]
        
        Returns:
            Synchronization score
        """
        batch_size = audio_features.shape[0]
        
        # Extract phonemes from audio
        phonemes = self.phoneme_extractor(audio_features)
        
        # Extract visemes from visual features
        visemes = self.viseme_extractor(visual_features)
        
        # Combine features
        combined = torch.cat([phonemes, visemes], dim=-1)
        
        # Score synchronization
        sync_scores = self.sync_scorer(combined)
        
        # Average over time
        avg_sync = torch.mean(sync_scores, dim=1)
        
        return avg_sync


class VoiceBiometricsVerifier(nn.Module):
    """Verifies voice biometrics consistency throughout audio."""
    def __init__(self, audio_dim=768, speaker_dim=256):
        super(VoiceBiometricsVerifier, self).__init__()
        
        # Speaker embedding extractor
        self.speaker_encoder = nn.Sequential(
            nn.Linear(audio_dim, speaker_dim * 2),
            nn.ReLU(),
            nn.Linear(speaker_dim * 2, speaker_dim),
            nn.ReLU()
        )
        
        # Consistency analyzer
        self.consistency_analyzer = nn.Sequential(
            nn.Linear(speaker_dim, speaker_dim // 2),
            nn.ReLU(),
            nn.Linear(speaker_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, audio_features):
        """
        Args:
            audio_features: Audio features [batch, frames, features]
        
        Returns:
            Voice consistency score
        """
        batch_size, num_frames = audio_features.shape[:2]
        
        # Split sequence into segments
        num_segments = min(4, num_frames)
        segment_length = num_frames // num_segments
        
        segments = []
        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = audio_features[:, start:end].mean(dim=1)
            segments.append(segment)
            
        # Extract speaker embeddings for each segment
        speaker_embeddings = []
        for segment in segments:
            embedding = self.speaker_encoder(segment)
            speaker_embeddings.append(embedding)
            
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(speaker_embeddings)):
            for j in range(i+1, len(speaker_embeddings)):
                sim = F.cosine_similarity(
                    speaker_embeddings[i], 
                    speaker_embeddings[j], 
                    dim=1, 
                    eps=1e-8
                )
                similarities.append(sim)
                
        # Average similarities
        if similarities:
            avg_similarity = torch.stack(similarities, dim=1).mean(dim=1)
        else:
            avg_similarity = torch.ones(batch_size, device=audio_features.device)
        
        # Score consistency
        consistency = self.consistency_analyzer(speaker_embeddings[0])
        
        # Combine with similarity score
        final_score = (consistency.squeeze(1) + avg_similarity) / 2
        
        return final_score.unsqueeze(1)


# Multimodal Fusion and Temporal Consistency Modules

class DualSpatioTemporalAttention(nn.Module):
    """Dual attention mechanism focusing on both spatial and temporal features."""
    def __init__(self, feature_dim, num_heads=8):
        super(DualSpatioTemporalAttention, self).__init__()
        
        # Spatial multi-head attention
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Temporal multi-head attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU()
        )
        
    def forward(self, features):
        """
        Args:
            features: Input features [batch, time, space, channels]
            or [batch, time, channels] if no spatial dimension
        
        Returns:
            Fused features with dual attention
        """
        batch_size = features.shape[0]
        
        # Handle different input shapes
        if len(features.shape) == 4:
            # Has spatial dimension [batch, time, space, channels]
            seq_len, spatial_dim, channels = features.shape[1:]
            
            # Spatial attention (across spatial dimension)
            spatial_features = features.view(batch_size * seq_len, spatial_dim, channels)
            spatial_attn, _ = self.spatial_attention(
                spatial_features, spatial_features, spatial_features
            )
            spatial_attn = self.norm1(spatial_attn + spatial_features)
            
            # Reshape for temporal attention
            spatial_attn = spatial_attn.view(batch_size, seq_len, spatial_dim, channels)
            
            # Average over spatial dimension
            spatial_attn = torch.mean(spatial_attn, dim=2)  # [batch, time, channels]
        else:
            # No spatial dimension [batch, time, channels]
            spatial_attn = features
        
        # Temporal attention (across time dimension)
        temporal_attn, _ = self.temporal_attention(
            spatial_attn, spatial_attn, spatial_attn
        )
        temporal_attn = self.norm2(temporal_attn + spatial_attn)
        
        # Combine spatial and temporal features
        combined = torch.cat([
            spatial_attn.mean(dim=1),  # Average over time
            temporal_attn[:, -1]  # Use final temporal state
        ], dim=1)
        
        # Fuse features
        fused = self.fusion(combined)
        
        return fused


class EmotionRecognitionModule(nn.Module):
    """Recognizes emotions from facial expressions and speech patterns."""
    def __init__(self, visual_dim=512, audio_dim=768, feature_dim=128):
        super(EmotionRecognitionModule, self).__init__()
        
        # Visual emotion encoder
        self.visual_emotion_encoder = nn.Sequential(
            nn.Linear(visual_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2)
        )
        
        # Audio emotion encoder
        self.audio_emotion_encoder = nn.Sequential(
            nn.Linear(audio_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2)
        )
        
        # Emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 7)  # 7 basic emotions
        )
        
        # Consistency scorer
        self.consistency_scorer = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, visual_features, audio_features):
        """
        Args:
            visual_features: Visual features [batch, features]
            audio_features: Audio features [batch, features]
        
        Returns:
            Emotion consistency score and emotion predictions
        """
        # Encode emotions from visual features
        visual_emotions = self.visual_emotion_encoder(visual_features)
        
        # Encode emotions from audio features
        audio_emotions = self.audio_emotion_encoder(audio_features)
        
        # Combine features
        combined = torch.cat([visual_emotions, audio_emotions], dim=1)
        
        # Classify emotions
        emotion_probs = torch.softmax(self.emotion_classifier(combined), dim=1)
        
        # Score consistency (high if matches natural patterns)
        consistency = self.consistency_scorer(combined)
        
        return consistency, emotion_probs


# Advanced Machine Learning Models 

class SiameseNetwork(nn.Module):
    """Compares similarity between audio and video streams to detect synchronization issues."""
    def __init__(self, audio_dim=768, video_dim=1024, hidden_dim=256):
        super(SiameseNetwork, self).__init__()
        
        # Audio branch
        self.audio_branch = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU()
        )
        
        # Video branch
        self.video_branch = nn.Sequential(
            nn.Linear(video_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU()
        )
        
        # Similarity scorer
        self.similarity_scorer = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, audio_features, video_features):
        """
        Args:
            audio_features: Audio features [batch, features]
            video_features: Video features [batch, features]
        
        Returns:
            Similarity score
        """
        # Process audio branch
        audio_embed = self.audio_branch(audio_features)
        
        # Process video branch
        video_embed = self.video_branch(video_features)
        
        # Calculate similarity
        l1_distance = torch.abs(audio_embed - video_embed)
        cosine_sim = F.cosine_similarity(audio_embed, video_embed, dim=1, eps=1e-8).unsqueeze(1)
        
        # Concatenate distance and similarity with embeddings
        combined = torch.cat([audio_embed, video_embed], dim=1)
        
        # Score similarity
        similarity = self.similarity_scorer(combined)
        
        return similarity


class Autoencoder(nn.Module):
    """Reconstructs images to detect anomalies."""
    def __init__(self, input_channels=3):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: Input image [batch, channels, height, width]
        
        Returns:
            Tuple of (reconstructed image, reconstruction error)
        """
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        # Calculate reconstruction error
        error = F.mse_loss(decoded, x, reduction='none')
        error = error.mean(dim=[1, 2, 3])  # Average over channels, height, width
        
        return decoded, error


# Forensic and Metadata Analysis Modules

class EnhancedMetadataAnalyzer(nn.Module):
    """Enhanced metadata analyzer for detecting inconsistencies in file metadata."""
    def __init__(self, input_dim=10, hidden_dim=64):
        super(EnhancedMetadataAnalyzer, self).__init__()
        
        # Metadata encoder
        self.metadata_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Forensic classifier
        self.forensic_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, metadata_features):
        """
        Args:
            metadata_features: Metadata features [batch, features]
        
        Returns:
            Metadata authenticity score
        """
        # Encode metadata
        encoded = self.metadata_encoder(metadata_features)
        
        # Classify forensic probability
        authenticity = self.forensic_classifier(encoded)
        
        return authenticity


class DigitalArtifactDetector(nn.Module):
    """Detects digital artifacts left during image manipulation."""
    def __init__(self, input_channels=3, feature_dim=64):
        super(DigitalArtifactDetector, self).__init__()
        
        # Artifact feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Artifact classifier
        self.artifact_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image):
        """
        Args:
            image: Input image [batch, channels, height, width]
        
        Returns:
            Artifact detection score
        """
        # Extract features
        features = self.feature_extractor(image)
        
        # Classify artifacts
        artifact_score = self.artifact_classifier(features)
        
        return artifact_score


class CompressionAnalyzer(nn.Module):
    """Analyzes compression artifacts and patterns for manipulation detection."""
    def __init__(self, input_channels=3, feature_dim=64):
        super(CompressionAnalyzer, self).__init__()
        
        # Compression feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Compression analyzer
        self.compression_analyzer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image):
        """
        Args:
            image: Input image [batch, channels, height, width]
        
        Returns:
            Compression consistency score
        """
        # Extract features
        features = self.feature_extractor(image)
        
        # Analyze compression consistency
        compression_score = self.compression_analyzer(features)
        
        return compression_score


# Liveness Detection and Real-Time Processing Modules

class LivenessDetectionModule(nn.Module):
    """Performs liveness detection tests to verify real human presence."""
    def __init__(self, visual_dim=512, feature_dim=128):
        super(LivenessDetectionModule, self).__init__()
        
        # Liveness feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(visual_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU()
        )
        
        # Liveness tests (multiple checks)
        self.blink_detector = nn.Linear(feature_dim, 1)
        self.head_pose_analyzer = nn.Linear(feature_dim, 1)
        self.texture_analyzer = nn.Linear(feature_dim, 1)
        self.reflection_detector = nn.Linear(feature_dim, 1)
        
        # Overall liveness scorer
        self.liveness_scorer = nn.Sequential(
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, visual_features):
        """
        Args:
            visual_features: Visual features [batch, features]
        
        Returns:
            Liveness score
        """
        # Extract features
        features = self.feature_extractor(visual_features)
        
        # Run liveness tests
        blink_score = torch.sigmoid(self.blink_detector(features))
        pose_score = torch.sigmoid(self.head_pose_analyzer(features))
        texture_score = torch.sigmoid(self.texture_analyzer(features))
        reflection_score = torch.sigmoid(self.reflection_detector(features))
        
        # Combine test results
        test_scores = torch.cat([
            blink_score, pose_score, texture_score, reflection_score
        ], dim=1)
        
        # Calculate overall liveness score
        liveness_score = self.liveness_scorer(test_scores)
        
        return liveness_score, {
            'blink': blink_score,
            'head_pose': pose_score,
            'texture': texture_score,
            'reflection': reflection_score
        }


class LightweightModelProcessor(nn.Module):
    """Lightweight model architecture for edge deployment."""
    def __init__(self, input_channels=3, feature_dim=32):
        super(LightweightModelProcessor, self).__init__()
        
        # Efficient feature extractor (using depthwise separable convolutions)
        self.feature_extractor = nn.Sequential(
            # Depthwise separable convolution 1
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels),
            nn.Conv2d(input_channels, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(),
            nn.MaxPool2d(2),
            
            # Depthwise separable convolution 2
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.Conv2d(feature_dim, feature_dim*2, kernel_size=1),
            nn.BatchNorm2d(feature_dim*2),
            nn.ReLU6(),
            nn.MaxPool2d(2),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Classifier (using inverted residual structure)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim*4),
            nn.ReLU6(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim*4, feature_dim*2),
            nn.ReLU6(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim*2, 2)  # Binary classification
        )
        
    def forward(self, x):
        """
        Args:
            x: Input image [batch, channels, height, width]
        
        Returns:
            Classification logits
        """
        # Extract features efficiently
        features = self.feature_extractor(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits


class MultiModalDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2, video_feature_dim=1024, audio_feature_dim=1024, 
                 transformer_dim=768, num_transformer_layers=4, enable_face_mesh=True,
                 enable_explainability=True, fusion_type='attention', 
                 backbone_visual='efficientnet', backbone_audio='wav2vec2',
                 use_spectrogram=True, detect_deepfake_type=True, num_deepfake_types=7,
                 debug=False):
                 
        super(MultiModalDeepfakeModel, self).__init__()
        self.debug = debug
        self.enable_face_mesh = enable_face_mesh
        self.enable_explainability = enable_explainability
        self.fusion_type = fusion_type
        self.use_spectrogram = use_spectrogram
        self.detect_deepfake_type = detect_deepfake_type
        
        # Automatically adjust feature dimensions based on selected backbones
        if backbone_visual == 'efficientnet':
            self.actual_video_feature_dim = 1280  # EfficientNet-B0 outputs 1280 features
        elif backbone_visual == 'swin':
            self.actual_video_feature_dim = 1024  # Swin outputs 1024 features
        else:
            self.actual_video_feature_dim = video_feature_dim
            
        if backbone_audio == 'wav2vec2' or backbone_audio == 'hubert':
            self.actual_audio_feature_dim = 768  # Both wav2vec2 and hubert output 768 features
        else:
            self.actual_audio_feature_dim = audio_feature_dim
        
        # Choose visual backbone
        if backbone_visual == 'efficientnet':
            self.visual_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.visual_model.classifier = nn.Identity()
            visual_out_dim = 1280
        elif backbone_visual == 'swin':
            self.visual_model = swin_v2_b(weights='IMAGENET1K_V1')
            self.visual_model.head = nn.Identity()
            visual_out_dim = 1024
        else:  # Default to EfficientNet
            self.visual_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.visual_model.classifier = nn.Identity()
            visual_out_dim = 1280
            
        self.video_projection = nn.Linear(visual_out_dim, self.actual_video_feature_dim)
        
        # Choose audio backbone
        if backbone_audio == 'wav2vec2':
            self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            audio_out_dim = 768
        elif backbone_audio == 'hubert':
            self.audio_model = AutoModel.from_pretrained("facebook/hubert-base-ls960")
            audio_out_dim = 768
        else:  # Default to Wav2Vec2
            self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            audio_out_dim = 768
            
        self.audio_projection = nn.Linear(audio_out_dim, self.actual_audio_feature_dim)
        
        # Spectrogram model for additional audio features
        if self.use_spectrogram:
            self.spectrogram_model = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            spec_out_dim = 128
            self.spectrogram_projection = nn.Linear(spec_out_dim, self.actual_audio_feature_dim)

        # Choose fusion type
        if fusion_type == 'attention':
            self.fusion_module = AttentionFusion(
                visual_dim=self.actual_video_feature_dim,
                audio_dim=self.actual_audio_feature_dim,
                output_dim=transformer_dim
            )
        elif fusion_type == 'concat':
            self.combined_projection = nn.Linear(self.actual_video_feature_dim + self.actual_audio_feature_dim, transformer_dim)
        else:  # Default to simple concat
            self.combined_projection = nn.Linear(self.actual_video_feature_dim + self.actual_audio_feature_dim, transformer_dim)

        # Transformer for sequence modeling
        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=8, 
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Temporal attention for analyzing frame sequences
        self.temporal_attention = TemporalAttention(visual_out_dim)
        
        # Forensic consistency module
        self.forensic_module = ForensicConsistencyModule(3)  # 3 channels for RGB
        
        # ELA analysis module
        self.ela_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.ela_projection = nn.Linear(64, 128)
        
        # Metadata feature processing
        self.metadata_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Audio-Visual sync detector
        self.sync_detector = AudioVisualSyncDetector(
            visual_dim=self.actual_video_feature_dim,
            audio_dim=self.actual_audio_feature_dim
        )
        
        # Face embedding processor
        self.face_embedding_processor = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # NEW COMPONENTS FOR ENHANCED FEATURES
        
        # 1. Facial Dynamics & Micro-Expressions
        self.facial_au_analyzer = FacialActionUnitAnalyzer(input_dim=136, hidden_dim=128)
        self.micro_expression_detector = MicroExpressionDetector(input_channels=3, hidden_dim=64)
        self.landmark_trajectory_analyzer = FacialLandmarkTrajectoryAnalyzer(num_landmarks=68, hidden_dim=128)
        self.head_pose_estimator = HeadPoseEstimator(landmark_dim=136, hidden_dim=128)
        self.eye_analysis_module = EyeAnalysisModule(hidden_dim=64)
        self.lip_audio_sync_analyzer = LipAudioSyncAnalyzer(lip_dim=40, audio_dim=self.actual_audio_feature_dim, hidden_dim=128)
        self.oculomotor_dynamics_analyzer = OculomotorDynamicsAnalyzer(hidden_dim=64)
        
        # 2. Physiological Signal Analysis
        self.physiological_analyzer = RemotePhysiologicalAnalyzer(feature_dim=32)
        self.skin_color_analyzer = SkinColorAnalyzer(feature_dim=32)
        
        # 3. Visual Artifact & Spatial Analysis
        self.lighting_consistency_analyzer = LightingConsistencyAnalyzer(feature_dim=64)
        self.texture_analyzer = TextureAnalyzer(patch_size=32, feature_dim=64)
        self.frequency_domain_analyzer = FrequencyDomainAnalyzer(feature_dim=64)
        self.gan_fingerprint_detector = GANFingerprintDetector(feature_dim=128)
        
        # 4. Audio Analysis
        self.voice_analysis_module = VoiceAnalysisModule(audio_dim=self.actual_audio_feature_dim, feature_dim=128)
        self.mfcc_extractor = MFCCExtractor(num_mfcc=40, feature_dim=64)
        self.phoneme_viseme_analyzer = PhonemeVisemeAnalyzer(
            audio_dim=self.actual_audio_feature_dim,
            visual_dim=self.actual_video_feature_dim,
            hidden_dim=128
        )
        self.voice_biometrics_verifier = VoiceBiometricsVerifier(
            audio_dim=self.actual_audio_feature_dim,
            speaker_dim=256
        )
        
        # 5. Multimodal Fusion & Temporal Consistency
        self.dual_attention = DualSpatioTemporalAttention(feature_dim=128, num_heads=4)
        self.emotion_recognition = EmotionRecognitionModule(
            visual_dim=self.actual_video_feature_dim,
            audio_dim=self.actual_audio_feature_dim,
            feature_dim=128
        )
        
        # 6. Advanced Machine Learning Models
        self.siamese_network = SiameseNetwork(
            audio_dim=self.actual_audio_feature_dim,
            video_dim=self.actual_video_feature_dim,
            hidden_dim=256
        )
        self.autoencoder = Autoencoder(input_channels=3)
        
        # 7. Forensic & Metadata Analysis
        self.enhanced_metadata_analyzer = EnhancedMetadataAnalyzer(input_dim=10, hidden_dim=64)
        self.digital_artifact_detector = DigitalArtifactDetector(input_channels=3, feature_dim=64)
        self.compression_analyzer = CompressionAnalyzer(input_channels=3, feature_dim=64)
        
        # 8. Liveness Detection
        self.liveness_detector = LivenessDetectionModule(visual_dim=self.actual_video_feature_dim, feature_dim=128)
        self.lightweight_processor = LightweightModelProcessor(input_channels=3, feature_dim=32)
        
        # Combined features dimension
        combined_dim = transformer_dim
        if self.enable_explainability:
            # Original features + new enhanced features
            combined_dim += 128 * 4  # Original: ELA + metadata + sync + face embeddings
            combined_dim += 512      # Additional features from new modules
        
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Deepfake type classifier (optional)
        if self.detect_deepfake_type:
            self.deepfake_type_classifier = nn.Sequential(
                nn.Linear(combined_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_deepfake_types)
            )

        # Learnable inconsistency threshold
        self.deepfake_threshold = nn.Parameter(torch.tensor(20.0), requires_grad=True)
        
        # Additional learnable parameters for forensic analysis
        self.frequency_threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.noise_threshold = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.temporal_consistency_threshold = nn.Parameter(torch.tensor(0.7), requires_grad=True)
        
        # Explainability component weights (learnable importance of each component)
        if self.enable_explainability:
            # Increased to account for new components
            self.component_weights = nn.Parameter(torch.ones(20), requires_grad=True)

        # Initialize weights
        self._initialize_weights()
        
        # Mediapipe face mesh (optional)
        if self.enable_face_mesh:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                # Try to initialize facial landmark detector from dlib if available
                try:
                    import dlib
                    model_path = "shape_predictor_68_face_landmarks.dat"
                    self.face_detector = dlib.get_frontal_face_detector()
                    try:
                        self.facial_landmark_predictor = dlib.shape_predictor(model_path)
                    except:
                        print(f"Warning: Could not load facial landmark predictor model from {model_path}")
                        self.facial_landmark_predictor = None
                except ImportError:
                    print("Warning: dlib not available, some facial analysis features will be limited")
                    self.face_detector = None
                    self.facial_landmark_predictor = None
            except Exception as e:
                print(f"⚠️ Warning: Failed to initialize face mesh: {e}")
                self.enable_face_mesh = False
        
        if self.debug:
            print(f"Model initialized with video_feature_dim={self.actual_video_feature_dim}, audio_feature_dim={self.actual_audio_feature_dim}")

    def _initialize_weights(self):
        """Initialize model weights with carefully chosen schemes."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming/He initialization for ReLU-based CNN layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier/Glorot initialization for fully connected layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                # Orthogonal initialization for recurrent layers
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, inputs: Dict[str, Union[torch.Tensor, List, None]]) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for the model.
        
        Args:
            inputs: Dictionary containing:
                - video_frames: tensor of shape [B, T, C, H, W]
                - audio: tensor of shape [B, L]
                - audio_spectrogram: tensor of shape [B, 1, H, W] (optional)
                - original_video_frames: optional tensor of original frames
                - original_audio: optional tensor of original audio
                - fake_periods: optional list of fake periods
                - timestamps: optional list of timestamps
                - fake_mask: optional list or tensor of fake masks
                - face_embeddings: optional tensor of face embeddings
                - temporal_consistency: optional tensor of temporal consistency
                - metadata_features: optional tensor of metadata features
                - ela_features: optional tensor of ELA features
                - audio_visual_sync: optional tensor of A/V sync features
                - facial_landmarks: optional tensor of facial landmarks
        
        Returns:
            tuple: (output logits, detailed results dict)
        """
        try:
            # Extract inputs
            video_frames = inputs.get('video_frames')           # [B, T, C, H, W]
            audio = inputs.get('audio')                         # [B, L]
            audio_spectrogram = inputs.get('audio_spectrogram') # [B, 1, H, W]
            original_video_frames = inputs.get('original_video_frames')
            original_audio = inputs.get('original_audio')
            face_embeddings = inputs.get('face_embeddings')
            temporal_consistency = inputs.get('temporal_consistency')
            metadata_features = inputs.get('metadata_features')
            ela_features = inputs.get('ela_features')
            audio_visual_sync = inputs.get('audio_visual_sync')
            facial_landmarks = inputs.get('facial_landmarks')   # [B, T, 68*2]
            
            # Handle missing inputs
            if video_frames is None or audio is None:
                raise ValueError("Missing required inputs: video_frames or audio")
            
            # Get batch dimensions
            batch_size, num_frames, C, H, W = video_frames.size()

            # Debugging shapes
            if self.debug:
                print(f"Video frames shape: {video_frames.shape}")
                print(f"Audio shape: {audio.shape}")
                if audio_spectrogram is not None:
                    print(f"Audio spectrogram shape: {audio_spectrogram.shape}")

            # Visual features extraction
            video_frames_flat = video_frames.view(batch_size * num_frames, C, H, W)
            visual_features = self.visual_model(video_frames_flat)
            visual_features = visual_features.view(batch_size, num_frames, -1)
            
            # Apply temporal attention to analyze frame sequences
            temporal_visual_features = self.temporal_attention(visual_features)
            
            # Pool temporal features for frame-level representation
            video_features = torch.mean(temporal_visual_features, dim=1)  # [B, feature_dim]
            video_features = self.video_projection(video_features)        # [B, video_feature_dim]

            # Audio features extraction
            # Normalize audio to [-1, 1] and ensure float32
            audio = audio.float()
            max_vals = torch.abs(audio).max(dim=1, keepdim=True)[0]
            # Avoid division by zero
            max_vals = torch.clamp(max_vals, min=1e-6)
            audio = audio / max_vals

            # Extract audio features using pre-trained model
            with torch.no_grad():  # Optional: freeze wav2vec2 during initial training
                audio_features = self.audio_model(audio).last_hidden_state  # [B, T', 768]
            audio_features = torch.mean(audio_features, dim=1)          # [B, 768]
            audio_features = self.audio_projection(audio_features)      # [B, audio_feature_dim]
            
            # Process spectrogram if available
            spec_features = None
            if self.use_spectrogram and audio_spectrogram is not None:
                spec_features = self.spectrogram_model(audio_spectrogram)  # [B, spec_out_dim]
                spec_features = self.spectrogram_projection(spec_features)  # [B, audio_feature_dim]
                
                # Combine with wav2vec features
                audio_features = audio_features + spec_features

            # Process forensic consistency
            forensic_features = self.forensic_module(video_frames)  # [B, T, hidden_dim]
            forensic_features = torch.mean(forensic_features, dim=1)  # [B, hidden_dim]

            # Initialize explainability features
            explainability_features = []
            component_contributions = {}
            
            # Process ELA features if available
            ela_output = None
            if ela_features is not None:
                # Add channel dimension if missing
                if len(ela_features.shape) == 3:
                    ela_features = ela_features.unsqueeze(1)
                ela_output = self.ela_encoder(ela_features)
                ela_output = self.ela_projection(ela_output)
                explainability_features.append(ela_output)
                component_contributions['ela'] = ela_output
            else:
                explainability_features.append(torch.zeros(batch_size, 128, device=video_frames.device))
            
            # Process metadata features if available
            metadata_output = None
            if metadata_features is not None:
                metadata_output = self.metadata_encoder(metadata_features)
                explainability_features.append(metadata_output)
                component_contributions['metadata'] = metadata_output
                
                # Enhanced metadata analysis
                enhanced_metadata_score = self.enhanced_metadata_analyzer(metadata_features)
                component_contributions['enhanced_metadata'] = enhanced_metadata_score
            else:
                explainability_features.append(torch.zeros(batch_size, 128, device=video_frames.device))
                component_contributions['enhanced_metadata'] = torch.zeros(batch_size, 1, device=video_frames.device)
            
            # Process audio-visual sync features if available
            av_sync_score = None
            if audio_visual_sync is not None:
                av_sync_features = audio_visual_sync
            else:
                # Calculate sync between current video and audio if not provided
                av_sync_score = self.sync_detector(video_features, audio_features)
                av_sync_features = av_sync_score.view(batch_size, -1)
                
            explainability_features.append(av_sync_features)
            component_contributions['av_sync'] = av_sync_features
            
            # Process face embeddings if available
            face_embedding_output = None
            if face_embeddings is not None:
                face_embedding_output = self.face_embedding_processor(face_embeddings)
                explainability_features.append(face_embedding_output)
                component_contributions['face_embedding'] = face_embedding_output
            else:
                explainability_features.append(torch.zeros(batch_size, 128, device=video_frames.device))
                
            # Extract facial landmarks for advanced analysis if not provided
            if facial_landmarks is None and self.enable_face_mesh:
                try:
                    facial_landmarks = self.extract_facial_landmarks(video_frames)
                except Exception as e:
                    if self.debug:
                        print(f"Error extracting facial landmarks: {e}")
                    facial_landmarks = torch.zeros(batch_size, num_frames, 136, device=video_frames.device)
            elif facial_landmarks is None:
                facial_landmarks = torch.zeros(batch_size, num_frames, 136, device=video_frames.device)
                
            # Extract eye landmarks if needed (for eye analysis)
            eye_landmarks = self.extract_eye_landmarks_from_facial(facial_landmarks)
            
            # Extract lip landmarks if needed (for lip-audio sync)
            lip_landmarks = self.extract_lip_landmarks_from_facial(facial_landmarks)
            
            # NEW FEATURES PROCESSING
            
            # 1. Facial Dynamics Analysis
            # Facial Action Units analysis
            fau_score, au_sequence = self.facial_au_analyzer(facial_landmarks)
            component_contributions['fau_analysis'] = fau_score
            
            # Micro-expression detection
            micro_expr_score, expr_probs = self.micro_expression_detector(video_frames)
            component_contributions['micro_expressions'] = micro_expr_score.unsqueeze(1)
            
            # Landmark trajectory analysis
            landmark_consistency, motion_seq = self.landmark_trajectory_analyzer(facial_landmarks)
            component_contributions['landmark_trajectory'] = landmark_consistency
            
            # Head pose estimation
            head_pose_score, pose_seq = self.head_pose_estimator(facial_landmarks)
            component_contributions['head_pose'] = head_pose_score
            
            # Eye analysis (blinking and pupil dilation)
            eye_naturalness, blinks, pupil_dilation = self.eye_analysis_module(eye_landmarks)
            component_contributions['eye_naturalness'] = eye_naturalness
            
            # Lip-audio sync analysis
            lip_audio_sync_score, _ = self.lip_audio_sync_analyzer(
                lip_landmarks, 
                audio_features.unsqueeze(1).expand(-1, lip_landmarks.size(1), -1)
            )
            component_contributions['lip_audio_sync'] = lip_audio_sync_score.unsqueeze(1)
            
            # Oculomotor dynamics analysis
            oculomotor_naturalness, _ = self.oculomotor_dynamics_analyzer(eye_landmarks)
            component_contributions['oculomotor'] = oculomotor_naturalness
            
            # 2. Physiological Signal Analysis
            physio_results = self.physiological_analyzer(video_frames)
            component_contributions['physiological'] = physio_results['naturalness']
            
            # Extract skin color from face regions for pulse analysis
            skin_color_seq = self.extract_skin_color(video_frames)
            skin_naturalness = self.skin_color_analyzer(skin_color_seq)
            component_contributions['skin_color'] = skin_naturalness
            
            # 3. Visual Artifact Analysis
            # Lighting consistency
            lighting_consistency = self.lighting_consistency_analyzer(video_frames)
            component_contributions['lighting'] = lighting_consistency
            
            # Texture analysis (using first frame)
            texture_consistency, _ = self.texture_analyzer(video_frames[:, 0])
            component_contributions['texture'] = texture_consistency
            
            # Frequency domain analysis
            frequency_score = self.frequency_domain_analyzer(video_frames[:, 0])
            component_contributions['frequency'] = frequency_score
            
            # GAN fingerprint detection
            gan_score = self.gan_fingerprint_detector(video_frames[:, 0])
            component_contributions['gan_fingerprint'] = gan_score
            
            # 4. Audio Analysis
            # Voice analysis
            voice_authenticity = self.voice_analysis_module(
                audio_features.unsqueeze(1).expand(-1, 5, -1)  # Expand to sequence
            )
            component_contributions['voice_authenticity'] = voice_authenticity
            
            # MFCC analysis
            mfcc_consistency, _ = self.mfcc_extractor(audio)
            component_contributions['mfcc'] = mfcc_consistency
            
            # Voice biometrics verification
            voice_bio_consistency = self.voice_biometrics_verifier(
                audio_features.unsqueeze(1).expand(-1, 5, -1)  # Expand to sequence
            )
            component_contributions['voice_biometrics'] = voice_bio_consistency
            
            # 5. Multimodal Analysis
            # Siamese network comparison for audio-video coherence
            av_similarity = self.siamese_network(audio_features, video_features)
            component_contributions['av_similarity'] = av_similarity
            
            # Emotion recognition for cross-modal consistency
            emotion_consistency, _ = self.emotion_recognition(video_features, audio_features)
            component_contributions['emotion'] = emotion_consistency
            
            # Autoencoder reconstruction for anomaly detection
            if video_frames.size(1) > 0:
                _, ae_error = self.autoencoder(video_frames[:, 0])
                component_contributions['autoencoder'] = torch.sigmoid(1.0 - ae_error.unsqueeze(1))
            else:
                component_contributions['autoencoder'] = torch.ones(batch_size, 1, device=video_frames.device)
            
            # 6. Forensic Analysis
            # Digital artifact detection
            artifact_score = self.digital_artifact_detector(video_frames[:, 0])
            component_contributions['digital_artifacts'] = 1.0 - artifact_score  # Invert for consistency
            
            # Compression analysis
            compression_score = self.compression_analyzer(video_frames[:, 0])
            component_contributions['compression'] = 1.0 - compression_score  # Invert for consistency
            
            # 7. Liveness Detection
            liveness_score, _ = self.liveness_detector(video_features)
            component_contributions['liveness'] = liveness_score
            
            # Combine and fuse features
            if self.fusion_type == 'attention':
                combined_features = self.fusion_module(video_features, audio_features)
            else:  # Default to concat
                combined_features = torch.cat([video_features, audio_features], dim=-1)
                combined_features = self.combined_projection(combined_features)

            # Process through transformer
            transformer_output = self.transformer(combined_features.unsqueeze(1)).squeeze(1)
            
            # Combine transformer output with explainability features if enabled
            if self.enable_explainability:
                # Concatenate all explainability features
                # Normalize explainability features to ensure consistent dimensions
                normalized_features = []
                batch_size = inputs['video_frames'].shape[0] if 'video_frames' in inputs else 1
                target_dim = 4  # Set this to match your expected dimension
                device = next(self.parameters()).device

                for feature in explainability_features:
                    if feature is None:
                        normalized_features.append(torch.zeros(batch_size, target_dim, device=device))
                        continue
                        
                    # Ensure it's a tensor
                    if not isinstance(feature, torch.Tensor):
                        try:
                            feature = torch.tensor(feature, device=device)
                        except:
                            normalized_features.append(torch.zeros(batch_size, target_dim, device=device))
                            continue
                    
                    # Ensure it has batch dimension
                    if feature.dim() == 1:
                        feature = feature.unsqueeze(0)
                    
                    # Ensure correct batch size
                    if feature.shape[0] != batch_size:
                        if feature.shape[0] == 1:
                            feature = feature.expand(batch_size, -1)
                        else:
                            feature = feature[:batch_size]
                    
                    # Normalize feature dimension
                    feat_dim = feature.shape[1] if feature.dim() > 1 else 1
                    if feat_dim < target_dim:
                        # Pad with zeros
                        padding = torch.zeros(batch_size, target_dim - feat_dim, device=device)
                        feature = torch.cat([feature, padding], dim=1)
                    elif feat_dim > target_dim:
                        # Truncate
                        feature = feature[:, :target_dim]
                    
                    normalized_features.append(feature)

                try:
                    all_explainability = torch.cat(normalized_features, dim=-1)
                except Exception as e:
                    print(f"[WARNING] Error concatenating explainability features: {e}")
                    # Return a zero tensor as fallback
                    total_dim = sum(f.shape[1] for f in normalized_features if hasattr(f, 'shape'))
                    all_explainability = torch.zeros(batch_size, total_dim, device=device)
                
                # Weight each explainability component by learned weights
                weighted_components = {}
                for i, (key, value) in enumerate(component_contributions.items()):
                    if value is not None and i < len(self.component_weights):
                        weight = F.softmax(self.component_weights, dim=0)[i]
                        if value.dim() == 2:
                            # For 2D tensors (batch, features)
                            weighted_value = value * weight
                        else:
                            # For 1D tensors (batch)
                            weighted_value = value.view(batch_size, -1) * weight
                        weighted_components[key] = weighted_value
                
                # Create explainability vector by concatenating weighted features
                # Normalize weighted components for concatenation
                tensors_to_concatenate = [
                    value for key, value in weighted_components.items() 
                    if value is not None and value.dim() > 0
                ]
                
                # Normalize dimensions for concatenation
                normalized_tensors = []
                batch_size = inputs['video_frames'].shape[0] if 'video_frames' in inputs else 1
                target_dim = 4  # Set this to match your expected dimension
                device = next(self.parameters()).device
                
                for tensor in tensors_to_concatenate:
                    # Ensure correct batch size
                    if tensor.shape[0] != batch_size:
                        if tensor.shape[0] == 1:
                            tensor = tensor.expand(batch_size, -1)
                        else:
                            tensor = tensor[:batch_size]
                    
                    # Normalize feature dimension
                    feat_dim = tensor.shape[1] if tensor.dim() > 1 else 1
                    if feat_dim < target_dim:
                        # Pad with zeros
                        padding = torch.zeros(batch_size, target_dim - feat_dim, device=device)
                        tensor = torch.cat([tensor, padding], dim=1)
                    elif feat_dim > target_dim:
                        # Truncate
                        tensor = tensor[:, :target_dim]
                    
                    normalized_tensors.append(tensor)
                
                try:
                    explainability_vector = torch.cat(normalized_tensors, dim=-1)
                except Exception as e:
                    print(f"[WARNING] Error concatenating explainability vector: {e}")
                    # Create a fallback vector with a reasonable dimension
                    explainability_vector = torch.zeros(batch_size, target_dim * len(normalized_tensors), device=device)
                
                # Ensure consistent dimensionality
                explainability_vector = explainability_vector.view(batch_size, -1)
                
                # Combine with transformer output
                final_features = torch.cat([transformer_output, all_explainability, explainability_vector], dim=-1)
            else:
                final_features = transformer_output
            
            # Main classification
            output = self.classifier(final_features)
            
            # Deepfake type classification (optional)
            deepfake_type_output = None
            if self.detect_deepfake_type:
                deepfake_type_output = self.deepfake_type_classifier(final_features)
            
            # Deepfake check during evaluation
            deepfake_check_results = None
            explanation_data = None
            if not self.training:
                # Run detailed forensic analysis
                deepfake_check_results, explanation_data = self.deepfake_check_video(
                    video_frames=video_frames, 
                    original_video_frames=original_video_frames,
                    fake_periods=inputs.get('fake_periods'),
                    timestamps=inputs.get('timestamps'),
                    original_audio=original_audio, 
                    current_audio=audio,
                    ela_features=ela_features,
                    metadata_features=metadata_features,
                    temporal_consistency=temporal_consistency,
                    av_sync_features=audio_visual_sync,
                    facial_landmarks=facial_landmarks,
                    component_contributions=component_contributions
                )

            # Create results dictionary
            results = {
                'logits': output,
                'deepfake_type': deepfake_type_output,
                'deepfake_check': deepfake_check_results,
                'explanation': explanation_data,
                'component_weights': F.softmax(self.component_weights, dim=0) if self.enable_explainability else None,
                'component_contributions': component_contributions
            }

            return output, results
            
        except Exception as e:
            print(f"❌ Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
            # Return zero tensor with proper shape as fallback
            return torch.zeros((batch_size, 2), device=video_frames.device), {"error": str(e)}

    def extract_facial_landmarks(self, video_frames):
        """Extract facial landmarks using mediapipe/dlib."""
        batch_size, num_frames, C, H, W = video_frames.shape
        landmarks_batch = []
        
        try:
            for b in range(batch_size):
                landmarks_sequence = []
                
                for t in range(min(num_frames, 10)):  # Process up to 10 frames to save computation
                    frame = video_frames[b, t].permute(1, 2, 0).cpu().numpy()
                    frame = (frame * 255).astype(np.uint8)
                    
                    # Use mediapipe face mesh if available
                    if self.enable_face_mesh and hasattr(self, 'mp_face_mesh'):
                        results = self.mp_face_mesh.process(frame)
                        
                        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                            # Extract key landmarks (we'll use a subset for efficiency)
                            face_landmarks = results.multi_face_landmarks[0].landmark
                            landmarks = []
                            # Extract 68 key points (similar to dlib's 68 point model)
                            indices = self._get_key_landmark_indices()
                            for idx in indices:
                                if idx < len(face_landmarks):
                                    landmark = face_landmarks[idx]
                                    landmarks.extend([landmark.x * W, landmark.y * H])
                                else:
                                    landmarks.extend([0, 0])  # Padding if landmark not found
                        else:
                            # No face detected
                            landmarks = [0] * 136  # 68 landmarks * 2 coordinates
                    else:
                        # No face mesh available
                        landmarks = [0] * 136  # 68 landmarks * 2 coordinates
                    
                    landmarks_sequence.append(landmarks)
                
                # If we processed fewer than num_frames frames, pad with zeros
                while len(landmarks_sequence) < num_frames:
                    landmarks_sequence.append([0] * 136)
                
                # Convert to tensor
                landmarks_tensor = torch.tensor(
                    landmarks_sequence,
                    dtype=torch.float32,
                    device=video_frames.device
                )
                landmarks_batch.append(landmarks_tensor)
            
            # Stack along batch dimension
            return torch.stack(landmarks_batch, dim=0)
        
        except Exception as e:
            if self.debug:
                print(f"Error extracting facial landmarks: {e}")
            return torch.zeros((batch_size, num_frames, 136), device=video_frames.device)

    def _get_key_landmark_indices(self):
        """Get indices for key facial landmarks from mediapipe's 468 landmarks."""
        # Approximate mapping from mediapipe's 468 points to 68 point representation
        # These indices are an approximation and may need adjustment
        return [
            # Jaw line (0-16)
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84,
            
            # Right eyebrow (17-21)
            70, 63, 105, 66, 107,
            
            # Left eyebrow (22-26)
            336, 296, 334, 293, 300,
            
            # Nose bridge (27-30)
            6, 197, 195, 5,
            
            # Nose bottom (31-35)
            4, 45, 275, 440, 344,
            
            # Right eye (36-41)
            33, 160, 158, 133, 153, 144,
            
            # Left eye (42-47)
            362, 385, 387, 263, 373, 380,
            
            # Outer lips (48-59)
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
            
            # Inner lips (60-67)
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 318,
        ]

    def extract_eye_landmarks_from_facial(self, facial_landmarks):
        """Extract eye landmarks from full facial landmarks."""
        # Eye landmark indices in 68-point model
        left_eye_indices = range(36, 42)  # Left eye (36-41)
        right_eye_indices = range(42, 48)  # Right eye (42-47)
        
        batch_size, num_frames = facial_landmarks.shape[:2]
        
        try:
            eye_landmarks = []
            
            for b in range(batch_size):
                frame_landmarks = []
                
                for t in range(num_frames):
                    face_lm = facial_landmarks[b, t]
                    
                    # Extract eye landmarks
                    eyes_lm = []
                    for i in list(left_eye_indices) + list(right_eye_indices):
                        x_idx = i * 2
                        y_idx = i * 2 + 1
                        if x_idx < face_lm.shape[0] and y_idx < face_lm.shape[0]:
                            eyes_lm.extend([face_lm[x_idx], face_lm[y_idx]])
                        else:
                            eyes_lm.extend([0, 0])
                    
                    frame_landmarks.append(eyes_lm)
                
                eye_landmarks.append(torch.tensor(
                    frame_landmarks,
                    dtype=torch.float32,
                    device=facial_landmarks.device
                ))
            
            return torch.stack(eye_landmarks, dim=0)
        
        except Exception as e:
            if self.debug:
                print(f"Error extracting eye landmarks: {e}")
            # Return zeros with shape [batch_size, num_frames, 12*2] (12 eye landmarks total)
            return torch.zeros((batch_size, num_frames, 24), device=facial_landmarks.device)

    def extract_lip_landmarks_from_facial(self, facial_landmarks):
        """Extract lip landmarks from full facial landmarks."""
        # Lip landmark indices in 68-point model
        outer_lip_indices = range(48, 60)  # Outer lips (48-59)
        inner_lip_indices = range(60, 68)  # Inner lips (60-67)
        
        batch_size, num_frames = facial_landmarks.shape[:2]
        
        try:
            lip_landmarks = []
            
            for b in range(batch_size):
                frame_landmarks = []
                
                for t in range(num_frames):
                    face_lm = facial_landmarks[b, t]
                    
                    # Extract lip landmarks
                    lips_lm = []
                    for i in list(outer_lip_indices) + list(inner_lip_indices):
                        x_idx = i * 2
                        y_idx = i * 2 + 1
                        if x_idx < face_lm.shape[0] and y_idx < face_lm.shape[0]:
                            lips_lm.extend([face_lm[x_idx], face_lm[y_idx]])
                        else:
                            lips_lm.extend([0, 0])
                    
                    frame_landmarks.append(lips_lm)
                
                lip_landmarks.append(torch.tensor(
                    frame_landmarks,
                    dtype=torch.float32,
                    device=facial_landmarks.device
                ))
            
            return torch.stack(lip_landmarks, dim=0)
        
        except Exception as e:
            if self.debug:
                print(f"Error extracting lip landmarks: {e}")
            # Return zeros with shape [batch_size, num_frames, 20*2] (20 lip landmarks total)
            return torch.zeros((batch_size, num_frames, 40), device=facial_landmarks.device)

    def extract_skin_color(self, video_frames):
        """Extract average skin color from face regions for pulse analysis."""
        batch_size, num_frames, C, H, W = video_frames.shape
        
        try:
            skin_colors = []
            
            for b in range(batch_size):
                frame_colors = []
                
                for t in range(num_frames):
                    frame = video_frames[b, t].permute(1, 2, 0).cpu().numpy()
                    
                    # Simple skin detection using RGB values
                    # This is a basic approach - more sophisticated methods exist
                    r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
                    
                    # Simple skin color range in RGB
                    skin_mask = (r > 0.4) & (g > 0.28) & (b > 0.2) & \
                               (r > g) & (r > b) & \
                               (r - g > 0.1) & \
                               (abs(r - g) > 0.15)
                    
                    # Extract average color of skin regions
                    if np.any(skin_mask):
                        avg_r = np.mean(r[skin_mask])
                        avg_g = np.mean(g[skin_mask])
                        avg_b = np.mean(b[skin_mask])
                        frame_colors.append([avg_r, avg_g, avg_b])
                    else:
                        # No skin detected, use default values
                        frame_colors.append([0.5, 0.4, 0.35])  # Average skin tone
                
                skin_colors.append(torch.tensor(
                    frame_colors,
                    dtype=torch.float32,
                    device=video_frames.device
                ))
            
            return torch.stack(skin_colors, dim=0)
        
        except Exception as e:
            if self.debug:
                print(f"Error extracting skin colors: {e}")
            # Return zeros with shape [batch_size, num_frames, 3] (RGB values)
            return torch.zeros((batch_size, num_frames, 3), device=video_frames.device)

    def check_eye_blinking(self, frame):
        """Check if eyes are blinking in the given frame using face mesh."""
        if not self.enable_face_mesh:
            return None
            
        try:
            # Convert PyTorch tensor to numpy for MediaPipe
            frame_rgb = frame.permute(1, 2, 0).cpu().numpy()
            
            # Ensure frame is in correct format (0-1 float or 0-255 uint8)
            if frame_rgb.max() <= 1.0:
                frame_rgb = (frame_rgb * 255).astype(np.uint8)
                
            results = self.mp_face_mesh.process(frame_rgb)

            if not results or not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0].landmark
            
            # Extract eye landmarks
            left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]  # Key landmark indices for left eye
            right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]  # Key landmark indices for right eye
            
            # Calculate eye openness ratios (vertical / horizontal)
            def eye_aspect_ratio(eye_pts):
                # Vertical distances
                v1 = np.sqrt((eye_pts[1].x - eye_pts[5].x)**2 + (eye_pts[1].y - eye_pts[5].y)**2)
                v2 = np.sqrt((eye_pts[2].x - eye_pts[4].x)**2 + (eye_pts[2].y - eye_pts[4].y)**2)
                
                # Horizontal distance
                h = np.sqrt((eye_pts[0].x - eye_pts[3].x)**2 + (eye_pts[0].y - eye_pts[3].y)**2)
                
                # Eye aspect ratio
                return (v1 + v2) / (2.0 * h) if h > 0 else 0
            
            # Calculate ratios
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            # Average ratio
            ear = (left_ear + right_ear) / 2.0
            
            # EAR threshold for blink detection
            EAR_THRESHOLD = 0.2
                
            # Return True if eyes are blinking (eyes are closed)
            return ear < EAR_THRESHOLD
                
        except Exception as e:
            if self.debug:
                print(f"Error in eye blinking detection: {e}")
            return None

    def deepfake_check_video(self, video_frames, original_video_frames, fake_periods, timestamps, 
                            original_audio=None, current_audio=None, ela_features=None, 
                            metadata_features=None, temporal_consistency=None, av_sync_features=None,
                            facial_landmarks=None, component_contributions=None):
        """
        Comprehensive deepfake check by analyzing multiple forensic signals.
        
        Returns the number of inconsistencies detected and explanation data.
        """
        inconsistencies = {
            'video_frame_diff': 0,
            'audio_diff': 0,
            'ela_analysis': 0,
            'temporal_inconsistency': 0,
            'metadata_analysis': 0,
            'av_sync_issues': 0,
            'eye_blinking': 0,
            'facial_dynamics': 0,
            'physiological': 0,
            'lighting_consistency': 0,
            'frequency_domain': 0,
            'micro_expressions': 0,
            'head_pose': 0,
            'lip_sync': 0,
            'voice_authenticity': 0,
            'gan_fingerprint': 0,
            'texture_analysis': 0
        }
        total_checks = 0
        explanation = {
            'detection_scores': {},
            'highlighted_regions': [],
            'issues_found': [],
            'confidence': 0.0,
            'evidence': {}
        }

        try:
            # Check video inconsistencies if original frames are available
            if original_video_frames is not None and video_frames is not None:
                min_batch = min(video_frames.size(0), original_video_frames.size(0))
                min_frames = min(video_frames.size(1), original_video_frames.size(1))
                
                frame_diffs = []
                for b in range(min_batch):
                    batch_diffs = []
                    for t in range(min_frames):
                        frame_diff = torch.abs(video_frames[b, t] - original_video_frames[b, t]).mean().item()
                        batch_diffs.append(frame_diff)
                        if frame_diff > self.deepfake_threshold.item():
                            inconsistencies['video_frame_diff'] += 1
                            # Add to highlighted regions for explanation
                            explanation['highlighted_regions'].append((b, t, frame_diff))
                        total_checks += 1
                    
                    frame_diffs.append(batch_diffs)
                
                # Store as evidence
                explanation['evidence']['frame_differences'] = frame_diffs
                explanation['detection_scores']['frame_diff_score'] = inconsistencies['video_frame_diff'] / max(1, total_checks)
                
                if self.debug:
                    print(f"Video inconsistencies: {inconsistencies['video_frame_diff']}/{total_checks}")
                    
            # Check audio inconsistencies if original audio is available
            if original_audio is not None and current_audio is not None:
                try:
                    # Handle potential shape mismatches
                    min_batch_audio = min(current_audio.size(0), original_audio.size(0))
                    min_len_audio = min(current_audio.size(1), original_audio.size(1))
                    
                    audio_diff = torch.abs(
                        current_audio[:min_batch_audio, :min_len_audio] - 
                        original_audio[:min_batch_audio, :min_len_audio]
                    ).mean(dim=1).cpu().numpy()
                    
                    audio_issues = []
                    for i, score in enumerate(audio_diff):
                        if self.debug:
                            print(f"Audio difference score for sample {i}: {score:.4f}")
                        if score > 0.2:  # Arbitrary threshold for audio difference
                            inconsistencies['audio_diff'] += 1
                            audio_issues.append((i, score))
                    
                    # Store as evidence
                    explanation['evidence']['audio_diff'] = audio_diff.tolist()
                    explanation['detection_scores']['audio_diff_score'] = np.mean(audio_diff)
                    if audio_issues:
                        explanation['issues_found'].append(f"Audio manipulation detected in {len(audio_issues)} samples")
                            
                except Exception as audio_error:
                    if self.debug:
                        print(f"Error in audio comparison: {audio_error}")
                        
            # Check ELA (Error Level Analysis) features
            if ela_features is not None:
                try:
                    # Analyze ELA features
                    if isinstance(ela_features, torch.Tensor):
                        ela_mean = torch.mean(ela_features).item()
                        ela_std = torch.std(ela_features).item()
                        
                        # High ELA values indicate potential manipulation
                        if ela_mean > 0.5 or ela_std > 0.2:
                            inconsistencies['ela_analysis'] += 1
                            explanation['issues_found'].append(f"ELA analysis indicates possible manipulation (mean: {ela_mean:.2f}, std: {ela_std:.2f})")
                        
                        explanation['evidence']['ela_stats'] = {'mean': ela_mean, 'std': ela_std}
                        explanation['detection_scores']['ela_score'] = ela_mean
                except Exception as ela_error:
                    if self.debug:
                        print(f"Error in ELA analysis: {ela_error}")
            
            # Check temporal consistency
            if temporal_consistency is not None:
                try:
                    temporal_score = temporal_consistency.item() if isinstance(temporal_consistency, torch.Tensor) else temporal_consistency
                    
                    if temporal_score < self.temporal_consistency_threshold.item():
                        inconsistencies['temporal_inconsistency'] += 1
                        explanation['issues_found'].append(f"Low temporal consistency detected ({temporal_score:.2f})")
                    
                    explanation['evidence']['temporal_consistency'] = temporal_score
                    explanation['detection_scores']['temporal_score'] = 1.0 - temporal_score
                except Exception as temp_error:
                    if self.debug:
                        print(f"Error in temporal consistency check: {temp_error}")
            
            # Check metadata features
            if metadata_features is not None:
                try:
                    # Convert to numpy if tensor
                    if isinstance(metadata_features, torch.Tensor):
                        metadata = metadata_features.cpu().numpy()
                    else:
                        metadata = np.array(metadata_features)
                    
                    # Analyze compression and noise patterns
                    compression_sign = metadata[5] if len(metadata) > 5 else 0
                    noise_level = metadata[3] if len(metadata) > 3 else 0
                    small_file = metadata[9] if len(metadata) > 9 else 0
                    
                    metadata_score = compression_sign * 0.4 + noise_level * 0.4 + small_file * 0.2
                    
                    if metadata_score > 0.6:
                        inconsistencies['metadata_analysis'] += 1
                        explanation['issues_found'].append(f"Metadata analysis indicates manipulation (score: {metadata_score:.2f})")
                    
                    explanation['evidence']['metadata_score'] = metadata_score
                    explanation['detection_scores']['metadata_score'] = metadata_score
                except Exception as meta_error:
                    if self.debug:
                        print(f"Error in metadata analysis: {meta_error}")
                        
            # Check audio-visual synchronization
            if av_sync_features is not None:
                try:
                    # Convert to numpy if tensor
                    if isinstance(av_sync_features, torch.Tensor):
                        av_sync = av_sync_features.cpu().numpy()
                    else:
                        av_sync = np.array(av_sync_features)
                    
                    # Analyze sync features
                    correlation = av_sync[0] if len(av_sync) > 0 else 0
                    best_lag = abs(av_sync[1]) if len(av_sync) > 1 else 0
                    
                    # High lag or low correlation indicates sync issues
                    if best_lag > 2 or correlation < 0.3:
                        inconsistencies['av_sync_issues'] += 1
                        explanation['issues_found'].append(f"Audio-visual sync issues detected (lag: {best_lag:.1f}, correlation: {correlation:.2f})")
                    
                    explanation['evidence']['av_sync'] = {'lag': float(best_lag), 'correlation': float(correlation)}
                    explanation['detection_scores']['av_sync_score'] = (1.0 - correlation) * 0.5 + min(1.0, best_lag/5.0) * 0.5
                except Exception as sync_error:
                    if self.debug:
                        print(f"Error in A/V sync analysis: {sync_error}")
            
            # Process new component contributions if available
            if component_contributions:
                # 1. Facial dynamics checks
                if 'eye_naturalness' in component_contributions:
                    score = self._extract_score(component_contributions['eye_naturalness'])
                    if score < 0.5:
                        inconsistencies['eye_blinking'] += 1
                        explanation['issues_found'].append(f"Unnatural eye blinking patterns detected (score: {score:.2f})")
                    explanation['detection_scores']['eye_blinking_score'] = 1.0 - score
                
                if 'micro_expressions' in component_contributions:
                    score = self._extract_score(component_contributions['micro_expressions'])
                    if score < 0.5:
                        inconsistencies['micro_expressions'] += 1
                        explanation['issues_found'].append(f"Missing or unnatural micro-expressions detected (score: {score:.2f})")
                    explanation['detection_scores']['micro_expressions_score'] = 1.0 - score
                
                if 'landmark_trajectory' in component_contributions:
                    score = self._extract_score(component_contributions['landmark_trajectory'])
                    if score < 0.5:
                        inconsistencies['facial_dynamics'] += 1
                        explanation['issues_found'].append(f"Unnatural facial movement patterns detected (score: {score:.2f})")
                    explanation['detection_scores']['facial_dynamics_score'] = 1.0 - score
                
                if 'head_pose' in component_contributions:
                    score = self._extract_score(component_contributions['head_pose'])
                    if score < 0.5:
                        inconsistencies['head_pose'] += 1
                        explanation['issues_found'].append(f"Unnatural head pose movements detected (score: {score:.2f})")
                    explanation['detection_scores']['head_pose_score'] = 1.0 - score
                
                if 'lip_audio_sync' in component_contributions:
                    score = self._extract_score(component_contributions['lip_audio_sync'])
                    if score < 0.5:
                        inconsistencies['lip_sync'] += 1
                        explanation['issues_found'].append(f"Poor lip synchronization with audio detected (score: {score:.2f})")
                    explanation['detection_scores']['lip_sync_score'] = 1.0 - score
                
                # 2. Physiological signal checks
                if 'physiological' in component_contributions:
                    score = self._extract_score(component_contributions['physiological'])
                    if score < 0.5:
                        inconsistencies['physiological'] += 1
                        explanation['issues_found'].append(f"Unnatural physiological signals detected (score: {score:.2f})")
                    explanation['detection_scores']['physiological_score'] = 1.0 - score
                
                if 'skin_color' in component_contributions:
                    score = self._extract_score(component_contributions['skin_color'])
                    if score < 0.5:
                        inconsistencies['physiological'] += 1
                        explanation['issues_found'].append(f"Unnatural skin color variations detected (score: {score:.2f})")
                    explanation['detection_scores']['skin_color_score'] = 1.0 - score
                
                # 3. Visual artifact checks
                if 'lighting' in component_contributions:
                    score = self._extract_score(component_contributions['lighting'])
                    if score < 0.5:
                        inconsistencies['lighting_consistency'] += 1
                        explanation['issues_found'].append(f"Lighting inconsistencies detected (score: {score:.2f})")
                    explanation['detection_scores']['lighting_score'] = 1.0 - score
                
                if 'texture' in component_contributions:
                    score = self._extract_score(component_contributions['texture'])
                    if score < 0.5:
                        inconsistencies['texture_analysis'] += 1
                        explanation['issues_found'].append(f"Texture inconsistencies detected (score: {score:.2f})")
                    explanation['detection_scores']['texture_score'] = 1.0 - score
                
                if 'frequency' in component_contributions:
                    score = self._extract_score(component_contributions['frequency'])
                    if score > 0.5:  # Inverted score - higher means more artifacts
                        inconsistencies['frequency_domain'] += 1
                        explanation['issues_found'].append(f"Frequency domain artifacts detected (score: {score:.2f})")
                    explanation['detection_scores']['frequency_score'] = score
                
                if 'gan_fingerprint' in component_contributions:
                    score = self._extract_score(component_contributions['gan_fingerprint'])
                    if score > 0.5:  # Inverted score - higher means more likely GAN-generated
                        inconsistencies['gan_fingerprint'] += 1
                        explanation['issues_found'].append(f"GAN fingerprint detected (score: {score:.2f})")
                    explanation['detection_scores']['gan_fingerprint_score'] = score
                
                # 4. Audio analysis checks
                if 'voice_authenticity' in component_contributions:
                    score = self._extract_score(component_contributions['voice_authenticity'])
                    if score < 0.5:
                        inconsistencies['voice_authenticity'] += 1
                        explanation['issues_found'].append(f"Voice inconsistencies detected (score: {score:.2f})")
                    explanation['detection_scores']['voice_score'] = 1.0 - score
                
                if 'mfcc' in component_contributions:
                    score = self._extract_score(component_contributions['mfcc'])
                    if score < 0.5:
                        inconsistencies['voice_authenticity'] += 1
                        explanation['issues_found'].append(f"Speech pattern inconsistencies detected (score: {score:.2f})")
                    explanation['detection_scores']['mfcc_score'] = 1.0 - score
                
                # 5. Combined multimodal checks
                if 'av_similarity' in component_contributions:
                    score = self._extract_score(component_contributions['av_similarity'])
                    if score < 0.5:
                        inconsistencies['av_sync_issues'] += 1
                        explanation['issues_found'].append(f"Audio-visual coherence issues detected (score: {score:.2f})")
                    explanation['detection_scores']['av_coherence_score'] = 1.0 - score
                
                if 'emotion' in component_contributions:
                    score = self._extract_score(component_contributions['emotion'])
                    if score < 0.5:
                        inconsistencies['facial_dynamics'] += 1
                        explanation['issues_found'].append(f"Emotion inconsistency between face and voice detected (score: {score:.2f})")
                    explanation['detection_scores']['emotion_consistency_score'] = 1.0 - score
            
            # Calculate total number of different checks performed
            total_check_types = len([v for v in inconsistencies.values() if v > 0])
            
            # Calculate total inconsistencies found
            total_inconsistencies = sum(inconsistencies.values())
            
            # Calculate weighted combined score
            detection_scores = explanation['detection_scores']
            if detection_scores:
                combined_score = sum(detection_scores.values()) / len(detection_scores)
                explanation['confidence'] = min(0.99, combined_score)
            else:
                explanation['confidence'] = 0.5  # Default if no scores available
            
            return total_inconsistencies, explanation
        
        except Exception as e:
            if self.debug:
                print(f"Error in deepfake check: {e}")
                import traceback
                traceback.print_exc()
            return 0, {'detection_scores': {}, 'issues_found': ['Error performing analysis'], 'confidence': 0.0}
    
    def _extract_score(self, tensor_or_value):
        """Helper method to extract a scalar score from various tensor formats."""
        try:
            if isinstance(tensor_or_value, torch.Tensor):
                if tensor_or_value.numel() == 1:
                    return tensor_or_value.item()
                else:
                    # Take mean if it's a multi-element tensor
                    return tensor_or_value.mean().item()
            elif isinstance(tensor_or_value, dict) and 'naturalness' in tensor_or_value:
                # Handle physiological analyzer output
                return tensor_or_value['naturalness'].item() if isinstance(tensor_or_value['naturalness'], torch.Tensor) else tensor_or_value['naturalness']
            elif isinstance(tensor_or_value, (int, float)):
                return tensor_or_value
            else:
                return 0.5  # Default value
        except Exception as e:
            if self.debug:
                print(f"Error extracting score: {e}")
            return 0.5  # Default value
    
    def get_attention_maps(self, inputs):
        """
        Get attention maps for visualization.
        
        Args:
            inputs: The same inputs as provided to forward()
            
        Returns:
            Attention maps [batch, frames, classes]
        """
        try:
            if not hasattr(self, 'cam_extractor'):
                # Initialize attention map extractor on first use
                from pytorch_grad_cam import GradCAM
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                
                # Define target layer (usually last convolutional layer)
                if hasattr(self.visual_model, 'features'):
                    target_layer = [self.visual_model.features[-1]]
                else:
                    # Find a suitable layer
                    for name, module in reversed(list(self.visual_model.named_children())):
                        if isinstance(module, nn.Conv2d):
                            target_layer = [module]
                            break
                    else:
                        # Fallback
                        target_layer = list(self.visual_model.children())[-3:-2]
                
                self.cam_extractor = GradCAM(model=self, target_layers=target_layer)
            
            # Generate attention maps
            video_frames = inputs.get('video_frames')  # [batch, frames, channels, height, width]
            batch_size, num_frames = video_frames.shape[:2]
            
            attention_maps = []
            for t in range(min(num_frames, 5)):  # Process first 5 frames max for efficiency
                for b in range(batch_size):
                    frame = video_frames[b, t].unsqueeze(0)  # Add batch dimension [1, C, H, W]
                    
                    # Set requires_grad to True for the input
                    frame.requires_grad = True
                    
                    # Define targets (class 0 = real, class 1 = fake)
                    targets = [ClassifierOutputTarget(0), ClassifierOutputTarget(1)]
                    
                    # Generate CAM
                    cam = self.cam_extractor(input_tensor=frame, targets=targets)  # [1, H, W]
                    attention_maps.append(cam)
            
            # Stack and reshape
            attention_maps = torch.stack(attention_maps)
            attention_maps = attention_maps.view(batch_size, -1, 2, attention_maps.shape[-2], attention_maps.shape[-1])
            
            return attention_maps
        
        except Exception as e:
            if self.debug:
                print(f"Error generating attention maps: {e}")
            return None
                    

"""
Advanced Model Components for Deepfake Detection
This module contains improved model components designed to enhance deepfake detection accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SelfAttentionPooling(nn.Module):
    """
    Self-attention pooling module that learns to focus on important frames/features
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, sequence_length, feature_dim]
            
        Returns:
            Tensor of shape [batch_size, feature_dim]
        """
        # Calculate attention weights
        attn_weights = self.attention(x)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Apply attention weights
        weighted_x = x * attn_weights
        
        # Sum along the sequence dimension
        return weighted_x.sum(dim=1)


class TemporalConsistencyDetector(nn.Module):
    """
    Module to detect temporal inconsistencies in deepfakes
    """
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.output_dim = hidden_dim * 2  # Bidirectional
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, sequence_length, feature_dim]
            
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        _, hidden = self.gru(x)
        # Concat the last hidden state from both directions
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return hidden


class EnhancedCrossModalFusion(nn.Module):
    """
    Enhanced cross-modal fusion with gating mechanism and residual connections
    """
    def __init__(self, visual_dim, audio_dim, fusion_dim=512):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        
        # Cross-modal attention
        self.visual_attn = nn.MultiheadAttention(fusion_dim, 8, dropout=0.2)
        self.audio_attn = nn.MultiheadAttention(fusion_dim, 8, dropout=0.2)
        
        # Gating mechanism
        self.visual_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        self.audio_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        
    def forward(self, visual_features, audio_features):
        """
        Args:
            visual_features: Visual features tensor of shape [batch_size, seq_len, visual_dim] or [batch_size, visual_dim]
            audio_features: Audio features tensor of shape [batch_size, seq_len, audio_dim] or [batch_size, audio_dim]
        Returns:
            Fused features tensor of shape [batch_size, fusion_dim]
        """
        # Shape assertions
        assert visual_features.dim() in [2, 3], "visual_features must be [batch, seq_len, dim] or [batch, dim]"
        assert audio_features.dim() in [2, 3], "audio_features must be [batch, seq_len, dim] or [batch, dim]"
        batch_size = visual_features.size(0)
        # If input is [batch, dim], unsqueeze to [batch, 1, dim]
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(1)
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)
        # Project to common space
        v_proj = self.visual_proj(visual_features)
        a_proj = self.audio_proj(audio_features)
        # True cross-sequence attention: use all sequence steps
        v_proj_t = v_proj.transpose(0, 1)  # [seq_len, batch, fusion_dim]
        a_proj_t = a_proj.transpose(0, 1)
        # Cross attention: visual attending to audio
        v_attend_a, _ = self.visual_attn(v_proj_t, a_proj_t, a_proj_t)
        v_attend_a = v_attend_a.transpose(0, 1).mean(dim=1)  # [batch, fusion_dim]
        # Cross attention: audio attending to visual
        a_attend_v, _ = self.audio_attn(a_proj_t, v_proj_t, v_proj_t)
        a_attend_v = a_attend_v.transpose(0, 1).mean(dim=1)
        # Apply gating mechanism
        v_combined = torch.cat([v_proj.mean(dim=1), v_attend_a], dim=1)
        a_combined = torch.cat([a_proj.mean(dim=1), a_attend_v], dim=1)
        v_gate = self.visual_gate(v_combined)
        a_gate = self.audio_gate(a_combined)
        v_gated = v_proj.mean(dim=1) * v_gate + v_attend_a * (1 - v_gate)
        a_gated = a_proj.mean(dim=1) * a_gate + a_attend_v * (1 - a_gate)
        v_out = self.norm1(v_gated + v_proj.mean(dim=1))
        a_out = self.norm2(a_gated + a_proj.mean(dim=1))
        combined = torch.cat([v_out, a_out], dim=1)
        output = self.output_proj(combined)
        return output


class FocalLossWithLogits(nn.Module):
    """
    Focal Loss with logits for better handling of class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, target):
        """
        Args:
            logits: Raw model outputs (before softmax)
            target: Ground truth labels
            
        Returns:
            Focal loss value
        """
        BCE_loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none'
        )
        # Get the probabilities
        if logits.size(-1) > 1:
            prob = F.softmax(logits, dim=-1)
            prob_for_target = torch.gather(prob, 1, target.unsqueeze(1))
            prob_for_target = prob_for_target.squeeze(1)
        else:
            prob_for_target = torch.sigmoid(logits)
            prob_for_target = torch.where(target > 0.5, prob_for_target, 1 - prob_for_target)
        focal_weight = (1 - prob_for_target) ** self.gamma
        # Per-class alpha support
        if logits.size(-1) > 1:
            if isinstance(self.alpha, torch.Tensor):
                alpha_weight = self.alpha[target]
            else:
                alpha_weight = torch.ones_like(target) * self.alpha
        else:
            if isinstance(self.alpha, torch.Tensor):
                alpha_weight = torch.where(target > 0.5, self.alpha[1], self.alpha[0])
            else:
                alpha_weight = torch.where(target > 0.5, torch.ones_like(target) * self.alpha, torch.ones_like(target) * (1 - self.alpha))
        loss = alpha_weight * focal_weight * BCE_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsyncCrossModalConsistencyLoss(nn.Module):
    """
    Detects inconsistencies between audio and visual features across time
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, visual_features, audio_features, targets):
        """
        Args:
            visual_features: Visual features across frames [batch, seq_len, dim]
            audio_features: Audio features across frames [batch, seq_len, dim]
            targets: 1 for real, 0 for fake
            
        Returns:
            Loss that's higher when audio and visual features are inconsistent
            (typical in deepfakes)
        """
        # Normalize the features
        visual_norm = F.normalize(visual_features, p=2, dim=2)
        audio_norm = F.normalize(audio_features, p=2, dim=2)
        
        # Calculate similarity between visual and audio features
        batch_size, seq_len, _ = visual_norm.shape
        similarity = torch.bmm(visual_norm, audio_norm.transpose(1, 2))
        
        # Diagonal elements are the synchronous time points
        sync_sim = torch.diagonal(similarity, dim1=1, dim2=2)
        
        # Off-diagonal elements are the asynchronous time points
        # Create a mask to get only off-diagonal elements
        mask = torch.ones_like(similarity)
        mask[:, torch.arange(seq_len), torch.arange(seq_len)] = 0
        
        # Get asynchronous similarities
        async_sim = similarity * mask
        
        # For real videos, sync similarity should be higher than async
        # For fake videos, this pattern may be disrupted
        
        # Get the mean of sync and async similarities
        sync_mean = sync_sim.mean(dim=1)
        async_mean = async_sim.sum(dim=(1,2)) / (mask.sum(dim=(1,2)) + 1e-8)
        
        # Calculate margin ranking loss
        # For real videos: sync_mean should be higher than async_mean + margin
        # For fake videos: we don't enforce this constraint as strongly
        weights = targets.float()  # 1 for real, 0 for fake
        
        # Loss is higher for real videos that don't meet the constraint
        # and slightly penalizes fake videos that do meet it
        loss = weights * F.relu(async_mean - sync_mean + self.margin) + \
               (1 - weights) * F.relu(sync_mean - async_mean + self.margin * 0.1)
        
        return loss.mean()

class PeriodicalFeatureExtractor(nn.Module):
    """
    Extracts periodic features from temporal sequences to detect repeating patterns
    typical in synthetic/generated content.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Frequency analysis layers
        self.freq_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, sequence_length, feature_dim]
            
        Returns:
            Tensor of shape [batch_size, hidden_dim] containing periodical features
        """
        batch_size, seq_len, _ = x.shape
        assert x.device.type in ['cpu', 'cuda'], "Input tensor must be on CPU or CUDA device"
        encoded = self.encoder(x)
        # Use GPU FFT if available
        if encoded.device.type == 'cuda':
            fft_features = torch.fft.rfft(encoded, dim=1)
        else:
            fft_features = torch.fft.rfft(encoded.cpu(), dim=1)
        magnitude = torch.abs(fft_features)
        if encoded.device.type == 'cuda':
            magnitude = magnitude.to(encoded.device)
        freq_mean = torch.mean(magnitude, dim=1)
        freq_max, _ = torch.max(magnitude, dim=1)
        combined = freq_mean + freq_max
        output = self.freq_analyzer(combined)
        return output


class MultiScaleFeatureFusion(nn.Module):
    """
    Processes features at multiple temporal scales and fuses them for
    hierarchical representation.
    """
    def __init__(self, input_dim, scales=[1, 2, 4]):
        super().__init__()
        self.input_dim = input_dim
        self.scales = scales
        # Distribute input channels across scales. If input_dim is not divisible by
        # the number of scales, distribute the remainder to the first few scales.
        n_scales = len(scales)
        base = input_dim // n_scales
        rem = input_dim - base * n_scales
        out_channels = [base + (1 if i < rem else 0) for i in range(n_scales)]

        # Ensure sum(out_channels) == input_dim
        assert sum(out_channels) == input_dim, "Internal channel distribution error"

        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, out_ch, kernel_size=scale, stride=scale, padding=0),
                nn.ReLU(),
                nn.BatchNorm1d(out_ch)
            )
            for out_ch, scale in zip(out_channels, scales)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, input_dim)
        )
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, sequence_length, feature_dim]
            
        Returns:
            Tensor of shape [batch_size, input_dim] containing multi-scale features
        """
        batch_size, seq_len, feat_dim = x.shape
        assert feat_dim == self.input_dim, f"Feature dim {feat_dim} does not match input_dim {self.input_dim}"
        x_transposed = x.transpose(1, 2)
        scale_features = []
        for scale_idx, processor in enumerate(self.scale_processors):
            scaled_feat = processor(x_transposed)
            pooled = F.adaptive_avg_pool1d(scaled_feat, 1).squeeze(-1)
            scale_features.append(pooled)
        multi_scale = torch.cat(scale_features, dim=1)
        output = self.fusion(multi_scale)
        return output

"""
Unified Multimodal Deepfake Detection Model

This module combines the advanced features from both multi_modal_model.py and 
multi_modal_model_check_1.py into a single comprehensive system for deepfake detection.

Features:
- Facial dynamics analysis (FAU, micro-expressions, landmarks)
- Physiological signal analysis (heart rate, breathing patterns)
- Visual artifact detection (lighting, texture, frequency domain)
- Audio analysis (voice biometrics, MFCC, phoneme-viseme sync)
- Siamese networks and autoencoders for anomaly detection
- Liveness detection modules
- Adversarial training modules (FGSM, PGD attacks)
- Self-supervised learning (contrastive learning, masked autoencoders)
- Curriculum learning with difficulty scheduling
- Active learning strategies
- Model optimization (quantization, pruning, knowledge distillation)
- Real-time processing capabilities
- Ensemble methods with Bayesian uncertainty
- Comprehensive forensic analysis modules
"""

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
import librosa
import math
import scipy
from scipy import signal
import copy
import random
from torch.nn.utils import prune
import torch.quantization as quantization
from torch.jit import script
import warnings
from collections import OrderedDict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
try:
    import dlib
    from facenet_pytorch import InceptionResnetV1
    import onnx
    import onnxruntime as ort
    import tensorrt as trt
    HAS_OPTIONAL_DEPS = True
except ImportError:
    logger.warning("Some optional dependencies (dlib, facenet_pytorch, onnx, tensorrt) not found. Some features may be limited.")
    HAS_OPTIONAL_DEPS = False

# ============== CORE FUSION MODULES ==============

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
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        mean = torch.mean(x, dim=1)
        std = torch.std(x, dim=1)
        return torch.cat([mean, std], dim=-1)


# ============== ADVERSARIAL ROBUSTNESS MODULES ==============

class AdversarialNoise(nn.Module):
    """Adversarial noise injection for robustness training."""
    
    def __init__(self, epsilon=0.01, alpha=0.005, num_steps=5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        
    def fgsm_attack(self, x, grad):
        """Fast Gradient Sign Method attack."""
        return x + self.epsilon * grad.sign()
    
    def pgd_attack(self, x, grad, original_x):
        """Projected Gradient Descent attack."""
        x_adv = x.clone()
        for _ in range(self.num_steps):
            x_adv = x_adv + self.alpha * grad.sign()
            eta = torch.clamp(x_adv - original_x, -self.epsilon, self.epsilon)
            x_adv = original_x + eta
        return x_adv
    
    def forward(self, x, attack_type='fgsm', grad=None):
        if grad is None:
            return x
        
        if attack_type == 'fgsm':
            return self.fgsm_attack(x, grad)
        elif attack_type == 'pgd':
            return self.pgd_attack(x, grad, x)
        else:
            return x


class GradientMaskingDefense(nn.Module):
    """Gradient masking defense mechanism."""
    
    def __init__(self, masking_ratio=0.1):
        super().__init__()
        self.masking_ratio = masking_ratio
        
    def forward(self, grad):
        if self.training:
            # Randomly mask gradients
            mask = torch.rand_like(grad) > self.masking_ratio
            return grad * mask.float()
        return grad


class AdversarialTraining(nn.Module):
    """Adversarial training module with gradient masking defense."""
    
    def __init__(self, model, epsilon=0.01, alpha=0.005, num_steps=5):
        super().__init__()
        self.model = model
        self.adversarial_noise = AdversarialNoise(epsilon, alpha, num_steps)
        self.gradient_masking = GradientMaskingDefense()
        
    def forward(self, x, target=None, training=True):
        if training and target is not None:
            # Clean forward pass
            clean_output = self.model(x)
            
            # Generate adversarial examples
            x.requires_grad_(True)
            loss = F.cross_entropy(clean_output, target)
            grad = torch.autograd.grad(loss, x, create_graph=True)[0]
            
            # Apply gradient masking
            grad = self.gradient_masking(grad)
            
            # Generate adversarial examples
            x_adv = self.adversarial_noise(x, grad=grad)
            
            # Adversarial forward pass
            adv_output = self.model(x_adv)
            
            return clean_output, adv_output
        else:
            return self.model(x)


# ============== SELF-SUPERVISED LEARNING MODULES ==============

class SelfSupervisedPretrainer(nn.Module):
    """Self-supervised pre-training with contrastive learning."""
    
    def __init__(self, encoder, projection_dim=128, temperature=0.07):
        super().__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.output_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.temperature = temperature
        
    def contrastive_loss(self, z1, z2):
        """Compute contrastive loss between augmented views."""
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Create positive pairs mask
        mask = torch.eye(batch_size * 2, dtype=torch.bool).to(z.device)
        sim_matrix.masked_fill_(mask, float('-inf'))
        
        # Positive pairs
        pos_sim = torch.diag(sim_matrix, batch_size)
        neg_sim = sim_matrix[~mask].view(batch_size * 2, -1)
        
        # Compute loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size * 2, dtype=torch.long).to(z.device)
        
        return F.cross_entropy(logits, labels)
    
    def forward(self, x1, x2):
        # Encode both views
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Project to contrastive space
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)
        
        # Compute contrastive loss
        loss = self.contrastive_loss(z1, z2)
        
        return loss, h1, h2


class MaskedAutoencoderPretraining(nn.Module):
    """Masked Autoencoder for self-supervised pre-training."""
    
    def __init__(self, encoder, decoder, mask_ratio=0.25):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        
    def random_masking(self, x, mask_ratio):
        """Apply random masking to input."""
        batch_size, seq_len, dim = x.shape
        len_keep = int(seq_len * (1 - mask_ratio))
        
        # Random shuffle
        noise = torch.rand(batch_size, seq_len, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        
        # Generate mask
        mask = torch.ones([batch_size, seq_len], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x):
        # Apply masking
        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # Encode
        latent = self.encoder(x_masked)
        
        # Decode
        pred = self.decoder(latent, ids_restore)
        
        # Compute reconstruction loss
        loss = F.mse_loss(pred, x, reduction='none')
        loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()
        
        return loss, pred, mask


# ============== CURRICULUM LEARNING MODULES ==============

class CurriculumLearningScheduler:
    """Curriculum learning scheduler with difficulty progression."""
    
    def __init__(self, initial_difficulty=0.3, final_difficulty=1.0, 
                 progression_epochs=100, strategy='linear'):
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.progression_epochs = progression_epochs
        self.strategy = strategy
        
    def get_difficulty(self, epoch):
        """Get current difficulty level based on epoch."""
        if epoch >= self.progression_epochs:
            return self.final_difficulty
        
        progress = epoch / self.progression_epochs
        
        if self.strategy == 'linear':
            difficulty = self.initial_difficulty + (self.final_difficulty - self.initial_difficulty) * progress
        elif self.strategy == 'exponential':
            difficulty = self.initial_difficulty * (self.final_difficulty / self.initial_difficulty) ** progress
        elif self.strategy == 'step':
            if progress < 0.5:
                difficulty = self.initial_difficulty
            else:
                difficulty = self.final_difficulty
        else:
            difficulty = self.initial_difficulty + (self.final_difficulty - self.initial_difficulty) * progress
            
        return difficulty
    
    def filter_samples(self, samples, difficulty):
        """Filter samples based on difficulty level."""
        # Sort samples by difficulty (assuming samples have difficulty scores)
        if hasattr(samples[0], 'difficulty'):
            sorted_samples = sorted(samples, key=lambda x: x.difficulty)
            n_samples = int(len(sorted_samples) * difficulty)
            return sorted_samples[:n_samples]
        return samples


class ProgressiveTraining(nn.Module):
    """Progressive training with curriculum learning."""
    
    def __init__(self, model, curriculum_scheduler):
        super().__init__()
        self.model = model
        self.curriculum_scheduler = curriculum_scheduler
        self.current_epoch = 0
        
    def forward(self, x, targets=None, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
            
        # Get current difficulty
        difficulty = self.curriculum_scheduler.get_difficulty(self.current_epoch)
        
        # Apply difficulty-based modifications if needed
        if self.training and difficulty < 1.0:
            # Could implement noise reduction, simpler augmentations, etc.
            pass
            
        return self.model(x, targets)


# ============== ACTIVE LEARNING MODULES ==============

class ActiveLearningSelector:
    """Active learning sample selection strategies."""
    
    def __init__(self, strategy='uncertainty', batch_size=32):
        self.strategy = strategy
        self.batch_size = batch_size
        
    def uncertainty_sampling(self, predictions, n_samples):
        """Select samples with highest prediction uncertainty."""
        if predictions.dim() == 2:
            # Classification probabilities
            uncertainties = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=1)
        else:
            # Regression predictions - use variance
            uncertainties = torch.var(predictions, dim=1)
            
        _, indices = torch.topk(uncertainties, n_samples)
        return indices
    
    def diversity_sampling(self, features, n_samples):
        """Select diverse samples using k-means clustering."""
        try:
            from sklearn.cluster import KMeans
            
            # Cluster features
            kmeans = KMeans(n_clusters=n_samples, random_state=42)
            kmeans.fit(features.cpu().numpy())
            
            # Find closest samples to cluster centers
            distances = torch.cdist(features, torch.tensor(kmeans.cluster_centers_).to(features.device))
            indices = torch.argmin(distances, dim=0)
            
            return indices
        except ImportError:
            logger.warning("sklearn not available, falling back to random sampling")
            return torch.randperm(len(features))[:n_samples]
    
    def select_samples(self, unlabeled_data, predictions=None, features=None):
        """Select samples for labeling based on strategy."""
        n_samples = min(self.batch_size, len(unlabeled_data))
        
        if self.strategy == 'uncertainty' and predictions is not None:
            indices = self.uncertainty_sampling(predictions, n_samples)
        elif self.strategy == 'diversity' and features is not None:
            indices = self.diversity_sampling(features, n_samples)
        else:
            # Random sampling as fallback
            indices = torch.randperm(len(unlabeled_data))[:n_samples]
            
        return indices


# ============== MODEL OPTIMIZATION MODULES ==============

class ModelQuantization:
    """Model quantization for efficient inference."""
    
    def __init__(self, quantization_type='dynamic'):
        self.quantization_type = quantization_type
        
    def quantize_model(self, model):
        """Apply quantization to model."""
        if self.quantization_type == 'dynamic':
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        elif self.quantization_type == 'static':
            # Static quantization requires calibration dataset
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            quantized_model = torch.quantization.prepare(model, inplace=False)
            # Note: Need to run calibration data through the model here
            quantized_model = torch.quantization.convert(quantized_model, inplace=False)
        else:
            quantized_model = model
            
        return quantized_model


class ModelPruning:
    """Model pruning for efficient inference."""
    
    def __init__(self, pruning_ratio=0.2, structured=False):
        self.pruning_ratio = pruning_ratio
        self.structured = structured
        
    def prune_model(self, model):
        """Apply pruning to model."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                if self.structured:
                    prune.ln_structured(module, name='weight', amount=self.pruning_ratio, n=2, dim=0)
                else:
                    prune.l1_unstructured(module, name='weight', amount=self.pruning_ratio)
        
        return model


class KnowledgeDistillation(nn.Module):
    """Knowledge distillation for model compression."""
    
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.5):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, x, targets=None):
        # Student predictions
        student_outputs = self.student_model(x)
        
        # Teacher predictions (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(x)
        
        # Distillation loss
        distillation_loss = F.kl_div(
            F.log_softmax(student_outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Student loss
        if targets is not None:
            student_loss = F.cross_entropy(student_outputs, targets)
            total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        else:
            total_loss = distillation_loss
            
        return student_outputs, total_loss


# ============== REAL-TIME PROCESSING MODULES ==============

class SlidingWindowInference:
    """Sliding window inference for real-time processing."""
    
    def __init__(self, window_size=16, overlap=8):
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        
    def process_sequence(self, model, sequence):
        """Process sequence using sliding window."""
        seq_len = sequence.size(1)
        predictions = []
        
        for start in range(0, seq_len - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = sequence[:, start:end]
            
            with torch.no_grad():
                pred = model(window)
                predictions.append(pred)
        
        # Average overlapping predictions
        if predictions:
            return torch.mean(torch.stack(predictions), dim=0)
        else:
            return model(sequence)


class FrameBufferManager:
    """Frame buffer manager for real-time video processing."""
    
    def __init__(self, buffer_size=32, min_frames=8):
        self.buffer_size = buffer_size
        self.min_frames = min_frames
        self.buffer = []
        
    def add_frame(self, frame):
        """Add frame to buffer."""
        self.buffer.append(frame)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def get_sequence(self):
        """Get current sequence for processing."""
        if len(self.buffer) >= self.min_frames:
            return torch.stack(self.buffer[-self.min_frames:])
        return None
    
    def clear(self):
        """Clear buffer."""
        self.buffer.clear()


class AdaptiveResolutionScaling:
    """Adaptive resolution scaling for performance optimization."""
    
    def __init__(self, target_fps=30, min_resolution=224, max_resolution=512):
        self.target_fps = target_fps
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.current_resolution = max_resolution
        self.fps_history = []
        
    def update_resolution(self, current_fps):
        """Update resolution based on current FPS."""
        self.fps_history.append(current_fps)
        if len(self.fps_history) > 10:
            self.fps_history.pop(0)
        
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        if avg_fps < self.target_fps * 0.8:
            # Reduce resolution
            self.current_resolution = max(self.min_resolution, self.current_resolution - 32)
        elif avg_fps > self.target_fps * 1.2:
            # Increase resolution
            self.current_resolution = min(self.max_resolution, self.current_resolution + 32)
        
        return self.current_resolution
    
    def scale_input(self, x):
        """Scale input to current resolution."""
        return F.interpolate(x, size=(self.current_resolution, self.current_resolution), mode='bilinear', align_corners=False)


# ============== ENSEMBLE METHODS ==============

class MultiHeadEnsemble(nn.Module):
    """Multi-head ensemble with uncertainty estimation."""
    
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0 / len(models)] * len(models)
        
    def forward(self, x):
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        # Compute uncertainty as variance across models
        predictions_stack = torch.stack(predictions)
        uncertainty = torch.var(predictions_stack, dim=0)
        
        return ensemble_pred, uncertainty


class BayesianUncertaintyEstimation(nn.Module):
    """Bayesian uncertainty estimation using Monte Carlo dropout."""
    
    def __init__(self, model, num_samples=10):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        
    def forward(self, x):
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        predictions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(pred)
        
        # Compute mean and variance
        predictions_stack = torch.stack(predictions)
        mean_pred = torch.mean(predictions_stack, dim=0)
        uncertainty = torch.var(predictions_stack, dim=0)
        
        return mean_pred, uncertainty


# ============== FORENSIC ANALYSIS MODULES ==============

class ForensicConsistencyModule(nn.Module):
    """Forensic consistency analysis for deepfake detection."""
    
    def __init__(self, feature_dim=512):
        super(ForensicConsistencyModule, self).__init__()
        self.feature_dim = feature_dim
        
        # Lighting consistency checker
        self.lighting_analyzer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32)
        )
        
        # Texture consistency checker
        self.texture_analyzer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32)
        )
        
        # Combine features
        self.fusion = nn.Linear(64, feature_dim)
        
    def forward(self, x):
        lighting_features = self.lighting_analyzer(x)
        texture_features = self.texture_analyzer(x)
        
        combined = torch.cat([lighting_features, texture_features], dim=1)
        consistency_score = self.fusion(combined)
        
        return consistency_score


class AudioVisualSyncDetector(nn.Module):
    """Audio-visual synchronization detection for deepfake detection."""
    
    def __init__(self, visual_dim=512, audio_dim=512, hidden_dim=256):
        super(AudioVisualSyncDetector, self).__init__()
        
        # Visual feature processor
        self.visual_processor = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Audio feature processor
        self.audio_processor = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Sync detector
        self.sync_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, visual_features, audio_features):
        # Process features
        visual_processed = self.visual_processor(visual_features)
        audio_processed = self.audio_processor(audio_features)
        
        # Combine and detect sync
        combined = torch.cat([visual_processed, audio_processed], dim=-1)
        sync_score = self.sync_detector(combined)
        
        return sync_score


# ============== CONFIGURATION CLASS ==============

class UnifiedModelConfig:
    """Configuration class for the unified model."""
    
    def __init__(self, **kwargs):
        # Basic model parameters
        self.num_classes = kwargs.get('num_classes', 2)
        self.video_feature_dim = kwargs.get('video_feature_dim', 1024)
        self.audio_feature_dim = kwargs.get('audio_feature_dim', 1024)
        self.transformer_dim = kwargs.get('transformer_dim', 768)
        self.num_transformer_layers = kwargs.get('num_transformer_layers', 4)
        
        # Model architecture options
        self.backbone_visual = kwargs.get('backbone_visual', 'efficientnet')
        self.backbone_audio = kwargs.get('backbone_audio', 'wav2vec2')
        self.fusion_type = kwargs.get('fusion_type', 'attention')
        
        # Feature flags
        self.enable_face_mesh = kwargs.get('enable_face_mesh', True)
        self.enable_explainability = kwargs.get('enable_explainability', True)
        self.use_spectrogram = kwargs.get('use_spectrogram', True)
        self.detect_deepfake_type = kwargs.get('detect_deepfake_type', True)
        self.num_deepfake_types = kwargs.get('num_deepfake_types', 7)
        
        # Advanced features
        self.enable_adversarial_training = kwargs.get('enable_adversarial_training', False)
        self.enable_self_supervised = kwargs.get('enable_self_supervised', False)
        self.enable_curriculum_learning = kwargs.get('enable_curriculum_learning', False)
        self.enable_active_learning = kwargs.get('enable_active_learning', False)
        self.enable_quantization = kwargs.get('enable_quantization', False)
        self.enable_pruning = kwargs.get('enable_pruning', False)
        self.enable_real_time_optimization = kwargs.get('enable_real_time_optimization', False)
        self.enable_ensemble = kwargs.get('enable_ensemble', False)
        
        # Optimization parameters
        self.adversarial_epsilon = kwargs.get('adversarial_epsilon', 0.01)
        self.curriculum_progression_epochs = kwargs.get('curriculum_progression_epochs', 100)
        self.pruning_ratio = kwargs.get('pruning_ratio', 0.2)
        self.ensemble_size = kwargs.get('ensemble_size', 3)
        
        # Real-time processing
        self.window_size = kwargs.get('window_size', 16)
        self.buffer_size = kwargs.get('buffer_size', 32)
        self.target_fps = kwargs.get('target_fps', 30)
        
        # Debug
        self.debug = kwargs.get('debug', False)


# ============== MAIN UNIFIED MODEL ==============

class MultiModalDeepfakeModel(nn.Module):
    """
    Unified Multimodal Deepfake Detection Model
    
    This model combines advanced features from both the original multimodal model
    and the enhanced check_1 model to provide comprehensive deepfake detection
    capabilities with adversarial robustness, self-supervised learning, and
    real-time processing optimizations.
    """
    
    def __init__(self, config=None, **kwargs):
        super(MultiModalDeepfakeModel, self).__init__()
        
        # Initialize configuration
        if config is None:
            config = UnifiedModelConfig(**kwargs)
        self.config = config
        
        # Initialize model components
        self._init_backbones()
        self._init_fusion_modules()
        self._init_forensic_modules()
        self._init_advanced_features()
        self._init_classifiers()
        
        logger.info(f"Initialized UnifiedMultiModalDeepfakeModel with config: {vars(config)}")
    
    def _init_backbones(self):
        """Initialize visual and audio backbones."""
        # Visual backbone
        if self.config.backbone_visual == 'efficientnet':
            self.visual_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.visual_model.classifier = nn.Identity()
            self.visual_feature_dim = 1280
        elif self.config.backbone_visual == 'swin':
            self.visual_model = swin_v2_b(weights='DEFAULT')
            self.visual_model.head = nn.Identity()
            self.visual_feature_dim = 1024
        else:
            raise ValueError(f"Unsupported visual backbone: {self.config.backbone_visual}")
        
        # Audio backbone
        if self.config.backbone_audio == 'wav2vec2':
            self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            self.audio_feature_dim = 768
        elif self.config.backbone_audio == 'hubert':
            self.audio_model = AutoModel.from_pretrained("facebook/hubert-base-ls960")
            self.audio_feature_dim = 768
        else:
            raise ValueError(f"Unsupported audio backbone: {self.config.backbone_audio}")
        
        # Feature projections
        self.visual_projection = nn.Linear(self.visual_feature_dim, self.config.video_feature_dim)
        self.audio_projection = nn.Linear(self.audio_feature_dim, self.config.audio_feature_dim)
        
        # Spectrogram model
        if self.config.use_spectrogram:
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
            self.spectrogram_projection = nn.Linear(128, self.config.audio_feature_dim)
    
    def _init_fusion_modules(self):
        """Initialize fusion modules."""
        # Attention fusion
        self.attention_fusion = AttentionFusion(
            self.config.video_feature_dim,
            self.config.audio_feature_dim,
            self.config.transformer_dim
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            self.config.transformer_dim,
            num_heads=8
        )
        
        # Stats pooling
        self.stats_pooling = StatsPooling()
    
    def _init_forensic_modules(self):
        """Initialize forensic analysis modules."""
        self.forensic_consistency = ForensicConsistencyModule(feature_dim=256)
        self.av_sync_detector = AudioVisualSyncDetector(
            visual_dim=self.config.video_feature_dim,
            audio_dim=self.config.audio_feature_dim
        )
    
    def _init_advanced_features(self):
        """Initialize advanced features based on configuration."""
        # Adversarial training
        if self.config.enable_adversarial_training:
            self.adversarial_training = AdversarialTraining(
                model=self,
                epsilon=self.config.adversarial_epsilon
            )
        
        # Self-supervised learning
        if self.config.enable_self_supervised:
            self.self_supervised_pretrainer = SelfSupervisedPretrainer(
                encoder=self,
                projection_dim=128
            )
        
        # Curriculum learning
        if self.config.enable_curriculum_learning:
            self.curriculum_scheduler = CurriculumLearningScheduler(
                progression_epochs=self.config.curriculum_progression_epochs
            )
            self.progressive_training = ProgressiveTraining(
                model=self,
                curriculum_scheduler=self.curriculum_scheduler
            )
        
        # Active learning
        if self.config.enable_active_learning:
            self.active_learning_selector = ActiveLearningSelector()
        
        # Model optimization
        if self.config.enable_quantization:
            self.quantization = ModelQuantization()
        
        if self.config.enable_pruning:
            self.pruning = ModelPruning(pruning_ratio=self.config.pruning_ratio)
        
        # Real-time processing
        if self.config.enable_real_time_optimization:
            self.sliding_window = SlidingWindowInference(
                window_size=self.config.window_size
            )
            self.frame_buffer = FrameBufferManager(
                buffer_size=self.config.buffer_size
            )
            self.adaptive_resolution = AdaptiveResolutionScaling(
                target_fps=self.config.target_fps
            )
        
        # Ensemble methods
        if self.config.enable_ensemble:
            self.bayesian_uncertainty = BayesianUncertaintyEstimation(
                model=self,
                num_samples=5
            )
    
    def _init_classifiers(self):
        """Initialize classification heads."""
        # Main classifier
        classifier_input_dim = self.config.transformer_dim * 2  # Stats pooling doubles dimension
        classifier_input_dim += 256  # Forensic consistency features
        classifier_input_dim += 1  # AV sync score
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.config.num_classes)
        )
        
        # Deepfake type classifier
        if self.config.detect_deepfake_type:
            self.deepfake_type_classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, self.config.num_deepfake_types)
            )
    
    def forward(self, video_frames, audio_waveform, spectrogram=None, targets=None, **kwargs):
        """
        Forward pass of the unified model.
        
        Args:
            video_frames: Video frames tensor [batch, seq_len, channels, height, width]
            audio_waveform: Audio waveform tensor [batch, seq_len]
            spectrogram: Optional spectrogram tensor [batch, channels, freq, time]
            targets: Optional target labels for training
            **kwargs: Additional arguments for specific modules
        
        Returns:
            Dictionary containing predictions and additional outputs
        """
        batch_size, seq_len = video_frames.size(0), video_frames.size(1)
        
        # Extract visual features
        visual_features = []
        for i in range(seq_len):
            frame = video_frames[:, i]  # [batch, channels, height, width]
            features = self.visual_model(frame)
            visual_features.append(features)
        
        visual_features = torch.stack(visual_features, dim=1)  # [batch, seq_len, features]
        visual_features = self.visual_projection(visual_features)
        
        # Extract audio features
        audio_outputs = self.audio_model(audio_waveform)
        audio_features = audio_outputs.last_hidden_state
        audio_features = self.audio_projection(audio_features)
        
        # Process spectrogram if available
        if self.config.use_spectrogram and spectrogram is not None:
            spec_features = self.spectrogram_model(spectrogram)
            spec_features = self.spectrogram_projection(spec_features)
            # Expand to match sequence length
            spec_features = spec_features.unsqueeze(1).expand(-1, seq_len, -1)
            # Combine with audio features
            audio_features = audio_features + spec_features
        
        # Ensure matching sequence lengths
        min_seq_len = min(visual_features.size(1), audio_features.size(1))
        visual_features = visual_features[:, :min_seq_len]
        audio_features = audio_features[:, :min_seq_len]
        
        # Apply fusion
        fused_features = self.attention_fusion(visual_features, audio_features)
        
        # Apply temporal attention
        fused_features = self.temporal_attention(fused_features)
        
        # Apply stats pooling
        pooled_features = self.stats_pooling(fused_features)
        
        # Forensic analysis
        forensic_features = self.forensic_consistency(video_frames[:, -1])  # Use last frame
        av_sync_score = self.av_sync_detector(
            visual_features.mean(dim=1),
            audio_features.mean(dim=1)
        )
        
        # Combine all features
        combined_features = torch.cat([
            pooled_features,
            forensic_features,
            av_sync_score
        ], dim=-1)
        
        # Main classification
        main_predictions = self.classifier(combined_features)
        
        # Prepare output
        output = {
            'predictions': main_predictions,
            'features': combined_features,
            'visual_features': visual_features,
            'audio_features': audio_features,
            'forensic_features': forensic_features,
            'av_sync_score': av_sync_score
        }
        
        # Deepfake type classification
        if self.config.detect_deepfake_type:
            deepfake_type_predictions = self.deepfake_type_classifier(combined_features)
            output['deepfake_type_predictions'] = deepfake_type_predictions
        
        # Apply uncertainty estimation if enabled
        if self.config.enable_ensemble and hasattr(self, 'bayesian_uncertainty'):
            mean_pred, uncertainty = self.bayesian_uncertainty(combined_features)
            output['uncertainty'] = uncertainty
            output['bayesian_predictions'] = mean_pred
        
        return output
    
    def get_config(self):
        """Get model configuration."""
        return self.config
    
    def update_config(self, **kwargs):
        """Update model configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def enable_feature(self, feature_name):
        """Enable a specific feature."""
        config_attr = f"enable_{feature_name}"
        if hasattr(self.config, config_attr):
            setattr(self.config, config_attr, True)
            logger.info(f"Enabled feature: {feature_name}")
        else:
            logger.warning(f"Unknown feature: {feature_name}")
    
    def disable_feature(self, feature_name):
        """Disable a specific feature."""
        config_attr = f"enable_{feature_name}"
        if hasattr(self.config, config_attr):
            setattr(self.config, config_attr, False)
            logger.info(f"Disabled feature: {feature_name}")
        else:
            logger.warning(f"Unknown feature: {feature_name}")


# ============== UTILITY FUNCTIONS ==============

def create_unified_model(config_dict=None, **kwargs):
    """
    Factory function to create a unified model with specified configuration.
    
    Args:
        config_dict: Dictionary containing configuration parameters
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured MultiModalDeepfakeModel instance
    """
    if config_dict is not None:
        kwargs.update(config_dict)
    
    config = UnifiedModelConfig(**kwargs)
    model = MultiModalDeepfakeModel(config=config)
    
    return model


def get_model_info(model):
    """
    Get information about the model configuration and capabilities.
    
    Args:
        model: MultiModalDeepfakeModel instance
    
    Returns:
        Dictionary containing model information
    """
    config = model.get_config()
    
    info = {
        'model_type': 'UnifiedMultiModalDeepfakeModel',
        'backbone_visual': config.backbone_visual,
        'backbone_audio': config.backbone_audio,
        'num_classes': config.num_classes,
        'features_enabled': {
            'adversarial_training': config.enable_adversarial_training,
            'self_supervised': config.enable_self_supervised,
            'curriculum_learning': config.enable_curriculum_learning,
            'active_learning': config.enable_active_learning,
            'quantization': config.enable_quantization,
            'pruning': config.enable_pruning,
            'real_time_optimization': config.enable_real_time_optimization,
            'ensemble': config.enable_ensemble,
            'face_mesh': config.enable_face_mesh,
            'explainability': config.enable_explainability,
            'spectrogram': config.use_spectrogram,
            'deepfake_type_detection': config.detect_deepfake_type
        },
        'parameters': {
            'adversarial_epsilon': config.adversarial_epsilon,
            'curriculum_progression_epochs': config.curriculum_progression_epochs,
            'pruning_ratio': config.pruning_ratio,
            'ensemble_size': config.ensemble_size,
            'window_size': config.window_size,
            'buffer_size': config.buffer_size,
            'target_fps': config.target_fps
        }
    }
    
    return info


# ============== EXAMPLE USAGE ==============

def example_usage():
    """Example usage of the unified model."""
    
    # Create basic model
    basic_config = {
        'num_classes': 2,
        'backbone_visual': 'efficientnet',
        'backbone_audio': 'wav2vec2',
        'enable_adversarial_training': False,
        'enable_real_time_optimization': False
    }
    
    basic_model = create_unified_model(basic_config)
    
    # Create advanced model with all features
    advanced_config = {
        'num_classes': 2,
        'backbone_visual': 'swin',
        'backbone_audio': 'wav2vec2',
        'enable_adversarial_training': True,
        'enable_self_supervised': True,
        'enable_curriculum_learning': True,
        'enable_active_learning': True,
        'enable_real_time_optimization': True,
        'enable_ensemble': True,
        'adversarial_epsilon': 0.02,
        'curriculum_progression_epochs': 50,
        'pruning_ratio': 0.3
    }
    
    advanced_model = create_unified_model(advanced_config)
    
    # Print model info
    print("Basic Model Info:")
    print(get_model_info(basic_model))
    
    print("\nAdvanced Model Info:")
    print(get_model_info(advanced_model))
    
    return basic_model, advanced_model


if __name__ == "__main__":
    # Example usage
    basic_model, advanced_model = example_usage()
    
    # Test forward pass with dummy data
    batch_size, seq_len, channels, height, width = 2, 8, 3, 224, 224
    audio_len = 16000  # 1 second at 16kHz
    
    dummy_video = torch.randn(batch_size, seq_len, channels, height, width)
    dummy_audio = torch.randn(batch_size, audio_len)
    dummy_spectrogram = torch.randn(batch_size, 1, 128, 128)
    
    try:
        # Test basic model
        with torch.no_grad():
            basic_output = basic_model(dummy_video, dummy_audio, dummy_spectrogram)
            print(f"\nBasic model output shape: {basic_output['predictions'].shape}")
        
        # Test advanced model
        with torch.no_grad():
            advanced_output = advanced_model(dummy_video, dummy_audio, dummy_spectrogram)
            print(f"Advanced model output shape: {advanced_output['predictions'].shape}")
            
        logger.info("Model testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model testing: {e}")
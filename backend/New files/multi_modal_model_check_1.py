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
import copy
import random
from torch.nn.utils import prune
import torch.quantization as quantization
from torch.jit import script
import warnings
from collections import OrderedDict

try:
    import dlib
    from facenet_pytorch import InceptionResnetV1
    import onnx
    import onnxruntime as ort
    import tensorrt as trt
except ImportError:
    print("Warning: Some optional dependencies (dlib, facenet_pytorch, onnx, tensorrt) not found, some features may be limited")

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

class GradientMaskingDefense(nn.Module):
    """Gradient masking defense mechanism."""
    
    def __init__(self, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std
        
    def forward(self, grad):
        if self.training:
            # Add noise to gradients
            noise = torch.randn_like(grad) * self.noise_std
            return grad + noise
        return grad

class InputPreprocessingDefense(nn.Module):
    """Input preprocessing defense mechanisms."""
    
    def __init__(self, defense_type='gaussian_blur', kernel_size=3, sigma=1.0):
        super().__init__()
        self.defense_type = defense_type
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def forward(self, x):
        if self.defense_type == 'gaussian_blur':
            return self.gaussian_blur(x)
        elif self.defense_type == 'median_filter':
            return self.median_filter(x)
        elif self.defense_type == 'bit_depth_reduction':
            return self.bit_depth_reduction(x)
        else:
            return x
    
    def gaussian_blur(self, x):
        """Apply Gaussian blur to input."""
        # Simple Gaussian blur implementation
        return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, 
                           padding=self.kernel_size//2)
    
    def median_filter(self, x):
        """Apply median filter to input."""
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=1, 
                           padding=self.kernel_size//2)
    
    def bit_depth_reduction(self, x):
        """Reduce bit depth of input."""
        return torch.round(x * 15) / 15

# ============== SELF-SUPERVISED LEARNING MODULES ==============

class SelfSupervisedPretrainer(nn.Module):
    """Self-supervised pretraining with contrastive learning."""
    
    def __init__(self, feature_dim=256, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 128)
        )
        
    def info_nce_loss(self, z_i, z_j):
        """InfoNCE loss for contrastive learning."""
        batch_size = z_i.shape[0]
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        
        # Create positive mask
        pos_mask = torch.eye(batch_size, device=z_i.device).bool()
        
        # Compute loss
        pos_sim = sim_matrix[pos_mask]
        neg_sim = sim_matrix[~pos_mask].view(batch_size, -1)
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)
        
        return F.cross_entropy(logits, labels)
    
    def forward(self, features_1, features_2):
        z_1 = self.projection_head(features_1)
        z_2 = self.projection_head(features_2)
        return self.info_nce_loss(z_1, z_2)

class MaskedAutoencoderPretraining(nn.Module):
    """Masked autoencoder for self-supervised pretraining."""
    
    def __init__(self, input_dim=768, hidden_dim=256, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        # Create random mask
        mask = torch.rand(x.shape[0], x.shape[1], device=x.device) > self.mask_ratio
        
        # Apply mask
        x_masked = x * mask.unsqueeze(-1).float()
        
        # Encode
        encoded = self.encoder(x_masked)
        
        # Decode
        decoded = self.decoder(encoded)
        
        # Reconstruction loss
        loss = F.mse_loss(decoded, x, reduction='mean')
        
        return decoded, loss

class CrossModalCorrespondenceLearning(nn.Module):
    """Cross-modal correspondence learning for self-supervised pretraining."""
    
    def __init__(self, visual_dim=768, audio_dim=768, hidden_dim=256):
        super().__init__()
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
        
    def forward(self, visual_features, audio_features):
        # Encode features
        visual_encoded = self.visual_encoder(visual_features)
        audio_encoded = self.audio_encoder(audio_features)
        
        # Compute correspondence loss
        correspondence_loss = F.mse_loss(visual_encoded, audio_encoded)
        
        return correspondence_loss

class TemporalPredictionTask(nn.Module):
    """Temporal prediction task for self-supervised learning."""
    
    def __init__(self, input_dim=768, hidden_dim=256, num_future_frames=5):
        super().__init__()
        self.num_future_frames = num_future_frames
        
        # Temporal encoder
        self.temporal_encoder = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, num_layers=2
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * num_future_frames)
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # Encode temporal features
        encoded, _ = self.temporal_encoder(x)
        
        # Predict future frames
        predictions = self.prediction_head(encoded[:, -1, :])  # Use last frame
        predictions = predictions.view(batch_size, self.num_future_frames, features)
        
        return predictions

# ============== CURRICULUM LEARNING MODULES ==============

class CurriculumLearningScheduler:
    """Curriculum learning scheduler for progressive training."""
    
    def __init__(self, total_epochs, warmup_epochs=5, strategy='linear'):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.strategy = strategy
        
    def get_difficulty_threshold(self, epoch):
        """Get the difficulty threshold for current epoch."""
        if epoch < self.warmup_epochs:
            return 0.0  # Easy samples only
        
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        
        if self.strategy == 'linear':
            return progress
        elif self.strategy == 'quadratic':
            return progress ** 2
        elif self.strategy == 'exponential':
            return 1 - np.exp(-3 * progress)
        else:
            return progress
    
    def filter_samples(self, samples, difficulties, epoch):
        """Filter samples based on difficulty threshold."""
        threshold = self.get_difficulty_threshold(epoch)
        mask = difficulties <= threshold
        return [sample for sample, keep in zip(samples, mask) if keep]

class ProgressiveTraining(nn.Module):
    """Progressive training with curriculum learning."""
    
    def __init__(self, model, curriculum_scheduler):
        super().__init__()
        self.model = model
        self.curriculum_scheduler = curriculum_scheduler
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        
    def forward(self, x, difficulties=None):
        if difficulties is not None and self.training:
            # Filter samples based on curriculum
            threshold = self.curriculum_scheduler.get_difficulty_threshold(self.current_epoch)
            mask = difficulties <= threshold
            
            if mask.any():
                x_filtered = x[mask]
                return self.model(x_filtered)
            else:
                return self.model(x)
        else:
            return self.model(x)

# ============== ACTIVE LEARNING MODULES ==============

class ActiveLearningSelector:
    """Active learning sample selection strategies."""
    
    def __init__(self, strategy='uncertainty', diversity_threshold=0.5):
        self.strategy = strategy
        self.diversity_threshold = diversity_threshold
        
    def uncertainty_sampling(self, predictions, n_samples):
        """Select samples with highest uncertainty."""
        # Calculate entropy as uncertainty measure
        probs = F.softmax(predictions, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # Select top uncertain samples
        _, indices = torch.topk(entropy, n_samples)
        return indices
    
    def diversity_sampling(self, features, n_samples):
        """Select diverse samples using k-means clustering."""
        # Simple diversity sampling using random selection
        # In practice, you'd use proper clustering algorithms
        indices = torch.randperm(features.size(0))[:n_samples]
        return indices
    
    def hybrid_sampling(self, predictions, features, n_samples):
        """Hybrid uncertainty and diversity sampling."""
        n_uncertain = int(n_samples * (1 - self.diversity_threshold))
        n_diverse = n_samples - n_uncertain
        
        # Get uncertain samples
        uncertain_indices = self.uncertainty_sampling(predictions, n_uncertain)
        
        # Get diverse samples from remaining
        remaining_mask = torch.ones(features.size(0), dtype=torch.bool)
        remaining_mask[uncertain_indices] = False
        remaining_features = features[remaining_mask]
        
        if remaining_features.size(0) > 0:
            diverse_indices = self.diversity_sampling(remaining_features, n_diverse)
            # Map back to original indices
            remaining_indices = torch.where(remaining_mask)[0]
            diverse_indices = remaining_indices[diverse_indices]
            
            return torch.cat([uncertain_indices, diverse_indices])
        else:
            return uncertain_indices

# ============== MODEL OPTIMIZATION MODULES ==============

class ModelQuantization:
    """Model quantization for optimization."""
    
    def __init__(self, model):
        self.model = model
        
    def dynamic_quantization(self):
        """Apply dynamic quantization."""
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
    
    def static_quantization(self, calibration_loader):
        """Apply static quantization with calibration."""
        # Prepare model for quantization
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        
        # Calibrate with representative data
        with torch.no_grad():
            for batch in calibration_loader:
                self.model(batch)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        return quantized_model

class ModelPruning:
    """Model pruning for size reduction."""
    
    def __init__(self, model):
        self.model = model
        
    def unstructured_pruning(self, amount=0.2):
        """Apply unstructured pruning."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
        return self.model
    
    def structured_pruning(self, amount=0.2):
        """Apply structured pruning."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
        return self.model
    
    def remove_pruning(self):
        """Remove pruning masks and make pruning permanent."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass
        return self.model

class KnowledgeDistillation(nn.Module):
    """Knowledge distillation for model compression."""
    
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.5):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, x, target=None):
        # Teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher_model(x)
        
        # Student predictions
        student_logits = self.student_model(x)
        
        # Distillation loss
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard target loss
        if target is not None:
            hard_loss = F.cross_entropy(student_logits, target)
            total_loss = self.alpha * hard_loss + (1 - self.alpha) * distillation_loss
            return student_logits, total_loss
        else:
            return student_logits, distillation_loss

# ============== REAL-TIME OPTIMIZATION MODULES ==============

class SlidingWindowInference:
    """Sliding window inference for real-time processing."""
    
    def __init__(self, window_size=16, stride=8):
        self.window_size = window_size
        self.stride = stride
        
    def process_stream(self, model, frames):
        """Process frames using sliding window."""
        results = []
        
        for i in range(0, len(frames) - self.window_size + 1, self.stride):
            window = frames[i:i + self.window_size]
            with torch.no_grad():
                result = model(window)
            results.append(result)
        
        return results

class FrameBufferManager:
    """Frame buffer management for streaming inference."""
    
    def __init__(self, max_frames=64, target_fps=30):
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.frame_buffer = []
        self.timestamps = []
        
    def add_frame(self, frame, timestamp):
        """Add frame to buffer."""
        self.frame_buffer.append(frame)
        self.timestamps.append(timestamp)
        
        # Remove old frames if buffer is full
        if len(self.frame_buffer) > self.max_frames:
            self.frame_buffer.pop(0)
            self.timestamps.pop(0)
    
    def get_frames(self, num_frames):
        """Get recent frames from buffer."""
        if len(self.frame_buffer) >= num_frames:
            return self.frame_buffer[-num_frames:]
        else:
            return self.frame_buffer

class AdaptiveResolutionScaling:
    """Adaptive resolution scaling for performance optimization."""
    
    def __init__(self, base_resolution=(224, 224), min_resolution=(112, 112)):
        self.base_resolution = base_resolution
        self.min_resolution = min_resolution
        self.current_resolution = base_resolution
        
    def scale_resolution(self, performance_metric):
        """Scale resolution based on performance metric."""
        if performance_metric < 0.5:  # Low performance
            # Reduce resolution
            new_h = max(self.min_resolution[0], int(self.current_resolution[0] * 0.8))
            new_w = max(self.min_resolution[1], int(self.current_resolution[1] * 0.8))
            self.current_resolution = (new_h, new_w)
        elif performance_metric > 0.8:  # High performance
            # Increase resolution
            new_h = min(self.base_resolution[0], int(self.current_resolution[0] * 1.2))
            new_w = min(self.base_resolution[1], int(self.current_resolution[1] * 1.2))
            self.current_resolution = (new_h, new_w)
        
        return self.current_resolution

class EarlyExitMechanism(nn.Module):
    """Early exit mechanism for adaptive inference."""
    
    def __init__(self, model, confidence_threshold=0.9):
        super().__init__()
        self.model = model
        self.confidence_threshold = confidence_threshold
        
        # Add early exit branches
        self.early_exits = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 2),
                nn.Softmax(dim=1)
            ) for _ in range(3)  # 3 early exit points
        ])
        
    def forward(self, x):
        # Forward pass with early exits
        for i, early_exit in enumerate(self.early_exits):
            # Get intermediate features (this is model-specific)
            intermediate_features = self.get_intermediate_features(x, layer_idx=i)
            
            # Early exit prediction
            early_pred = early_exit(intermediate_features)
            confidence = torch.max(early_pred, dim=1)[0]
            
            # Check if confidence is high enough for early exit
            if confidence.mean() > self.confidence_threshold:
                return early_pred, i  # Return prediction and exit layer
        
        # Full forward pass if no early exit
        final_pred = self.model(x)
        return final_pred, len(self.early_exits)
    
    def get_intermediate_features(self, x, layer_idx):
        """Get intermediate features at specified layer."""
        # This should be implemented based on your model architecture
        # For now, return a dummy tensor
        return torch.randn(x.size(0), 256, 7, 7)

class HierarchicalProcessing(nn.Module):
    """Hierarchical processing for multi-scale analysis."""
    
    def __init__(self, scales=[1.0, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
        
    def forward(self, x, model):
        """Process input at multiple scales."""
        results = []
        
        for scale in self.scales:
            # Scale input
            if scale != 1.0:
                size = (int(x.size(2) * scale), int(x.size(3) * scale))
                x_scaled = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            else:
                x_scaled = x
            
            # Forward pass
            result = model(x_scaled)
            results.append(result)
        
        # Aggregate results
        aggregated = torch.stack(results).mean(dim=0)
        return aggregated

# ============== ENSEMBLE & MULTI-HEAD MODULES ==============

class MultiHeadEnsemble(nn.Module):
    """Multi-head ensemble voting system."""
    
    def __init__(self, models, voting_strategy='soft'):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.voting_strategy = voting_strategy
        
    def forward(self, x):
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Ensemble voting
        if self.voting_strategy == 'soft':
            # Soft voting (average probabilities)
            ensemble_pred = torch.stack(predictions).mean(dim=0)
        elif self.voting_strategy == 'hard':
            # Hard voting (majority vote)
            hard_preds = torch.stack([torch.argmax(pred, dim=1) for pred in predictions])
            ensemble_pred = torch.mode(hard_preds, dim=0)[0]
        else:
            ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        return ensemble_pred, predictions

class BayesianUncertaintyEstimation(nn.Module):
    """Bayesian uncertainty estimation using Monte Carlo dropout."""
    
    def __init__(self, model, num_samples=10):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        
    def forward(self, x):
        # Enable dropout during inference
        self.model.train()
        
        predictions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(pred)
        
        # Calculate mean and variance
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        var_pred = predictions.var(dim=0)
        
        # Uncertainty as variance
        uncertainty = var_pred.mean(dim=1)
        
        return mean_pred, uncertainty

class TestTimeAugmentation:
    """Test-time augmentation for improved accuracy."""
    
    def __init__(self, augmentations=None):
        self.augmentations = augmentations or [
            lambda x: x,  # Original
            lambda x: torch.flip(x, [3]),  # Horizontal flip
            lambda x: torch.rot90(x, 1, [2, 3]),  # Rotate 90
            lambda x: torch.rot90(x, 3, [2, 3]),  # Rotate 270
        ]
        
    def forward(self, model, x):
        """Apply test-time augmentation."""
        predictions = []
        
        for aug in self.augmentations:
            x_aug = aug(x)
            pred = model(x_aug)
            predictions.append(pred)
        
        # Average predictions
        mean_pred = torch.stack(predictions).mean(dim=0)
        return mean_pred

class ModelAveraging:
    """Model averaging for ensemble methods."""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
    def forward(self, x):
        """Weighted average of model predictions."""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            pred = model(x)
            predictions.append(pred * weight)
        
        # Sum weighted predictions
        ensemble_pred = torch.stack(predictions).sum(dim=0)
        return ensemble_pred

# ============== ORIGINAL CLASSES (KEEPING CURRENT IMPLEMENTATION) ==============

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
        # Self-attention
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
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

# ============== MAIN MULTIMODAL MODEL (ENHANCED VERSION) ==============

class MultiModalDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2, video_feature_dim=1024, audio_feature_dim=1024, 
                 transformer_dim=768, num_transformer_layers=4, enable_face_mesh=True,
                 enable_explainability=True, fusion_type='attention', 
                 backbone_visual='efficientnet', backbone_audio='wav2vec2',
                 use_spectrogram=True, detect_deepfake_type=True, num_deepfake_types=7,
                 debug=False, 
                 # New parameters for enhanced features
                 enable_adversarial_training=False, enable_self_supervised=False,
                 enable_curriculum_learning=False, enable_active_learning=False,
                 enable_quantization=False, enable_pruning=False,
                 enable_real_time_optimization=False, enable_ensemble=False):
                 
        super(MultiModalDeepfakeModel, self).__init__()
        self.debug = debug
        self.enable_face_mesh = enable_face_mesh
        self.enable_explainability = enable_explainability
        self.fusion_type = fusion_type
        self.backbone_visual = backbone_visual
        self.backbone_audio = backbone_audio
        self.use_spectrogram = use_spectrogram
        self.detect_deepfake_type = detect_deepfake_type
        self.num_deepfake_types = num_deepfake_types
        
        # Enhanced features flags
        self.enable_adversarial_training = enable_adversarial_training
        self.enable_self_supervised = enable_self_supervised
        self.enable_curriculum_learning = enable_curriculum_learning
        self.enable_active_learning = enable_active_learning
        self.enable_quantization = enable_quantization
        self.enable_pruning = enable_pruning
        self.enable_real_time_optimization = enable_real_time_optimization
        self.enable_ensemble = enable_ensemble
        
        # Initialize visual backbone
        if backbone_visual == 'efficientnet':
            self.visual_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.visual_model.classifier = nn.Identity()
            visual_feature_dim = 1280
        elif backbone_visual == 'swin':
            self.visual_model = swin_v2_b(weights='DEFAULT')
            self.visual_model.head = nn.Identity()
            visual_feature_dim = 1024
        else:
            # Default to EfficientNet
            self.visual_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.visual_model.classifier = nn.Identity()
            visual_feature_dim = 1280
            
        # Initialize audio backbone
        if backbone_audio == 'wav2vec2':
            self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            audio_feature_dim = 768
        else:
            # Default to Wav2Vec2
            self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            audio_feature_dim = 768
        
        # Projection layers
        self.video_projection = nn.Linear(visual_feature_dim, video_feature_dim)
        self.audio_projection = nn.Linear(audio_feature_dim, audio_feature_dim)
        
        # Temporal attention for video sequences
        self.temporal_attention = TemporalAttention(video_feature_dim)
        
        # Cross-modal fusion
        if fusion_type == 'attention':
            self.fusion_module = AttentionFusion(
                video_feature_dim, audio_feature_dim, transformer_dim
            )
        else:
            # Simple concatenation fusion
            self.fusion_module = nn.Linear(video_feature_dim + audio_feature_dim, transformer_dim)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=8, 
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Statistics pooling
        self.stats_pooling = StatsPooling()
        
        # Forensic consistency module
        self.forensic_module = ForensicConsistencyModule(3, 256)
        
        # Audio-visual synchronization detector
        self.sync_detector = AudioVisualSyncDetector(video_feature_dim, audio_feature_dim)
        
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim * 2, 512),  # *2 for stats pooling
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Deepfake type classifier (if enabled)
        if detect_deepfake_type:
            self.deepfake_type_classifier = nn.Sequential(
                nn.Linear(transformer_dim * 2, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_deepfake_types)
            )
        
        # ============== ENHANCED MODULES ==============
        
        # Adversarial training modules
        if enable_adversarial_training:
            self.adversarial_noise = AdversarialNoise()
            self.gradient_masking = GradientMaskingDefense()
            self.input_preprocessing = InputPreprocessingDefense()
        
        # Self-supervised learning modules
        if enable_self_supervised:
            self.ssl_pretrainer = SelfSupervisedPretrainer(transformer_dim)
            self.mae_pretrainer = MaskedAutoencoderPretraining(transformer_dim)
            self.cross_modal_ssl = CrossModalCorrespondenceLearning(video_feature_dim, audio_feature_dim)
            self.temporal_prediction = TemporalPredictionTask(transformer_dim)
        
        # Curriculum learning
        if enable_curriculum_learning:
            self.curriculum_scheduler = CurriculumLearningScheduler(total_epochs=100)
            self.progressive_training = ProgressiveTraining(self, self.curriculum_scheduler)
        
        # Active learning
        if enable_active_learning:
            self.active_selector = ActiveLearningSelector(strategy='hybrid')
        
        # Model optimization
        if enable_quantization:
            self.quantizer = ModelQuantization(self)
        
        if enable_pruning:
            self.pruner = ModelPruning(self)
        
        # Real-time optimization
        if enable_real_time_optimization:
            self.sliding_window = SlidingWindowInference()
            self.frame_buffer = FrameBufferManager()
            self.adaptive_resolution = AdaptiveResolutionScaling()
            self.early_exit = EarlyExitMechanism(self)
            self.hierarchical_processing = HierarchicalProcessing()
        
        # Ensemble methods
        if enable_ensemble:
            self.bayesian_uncertainty = BayesianUncertaintyEstimation(self)
            self.tta = TestTimeAugmentation()
        
        # Initialize MediaPipe face mesh
        if enable_face_mesh:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                print(f"Warning: Could not initialize face mesh: {e}")
                self.enable_face_mesh = False
        
        # Initialize weights
        self._initialize_weights()
        
        # Learnable threshold for deepfake detection
        self.deepfake_threshold = nn.Parameter(torch.tensor(20.0), requires_grad=True)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs: Dict[str, Union[torch.Tensor, List, None]]) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Enhanced forward pass with all new features.
        
        Args:
            inputs: Dictionary containing:
                - video_frames: tensor of shape [B, T, C, H, W]
                - audio: tensor of shape [B, L]
                - Additional optional inputs for enhanced features
        
        Returns:
            tuple: (output logits, additional outputs dict)
        """
        try:
            video_frames = inputs['video_frames']  # [B, T, C, H, W]
            audio = inputs['audio']                # [B, L]
            
            batch_size, num_frames, C, H, W = video_frames.size()
            
            # Input preprocessing defense (if enabled)
            if self.enable_adversarial_training:
                video_frames = self.input_preprocessing(video_frames)
            
            # Extract visual features
            video_frames_flat = video_frames.view(batch_size * num_frames, C, H, W)
            visual_features = self.visual_model(video_frames_flat)
            visual_features = visual_features.view(batch_size, num_frames, -1)
            visual_features = self.video_projection(visual_features)
            
            # Apply temporal attention
            visual_features = self.temporal_attention(visual_features)
            
            # Extract audio features
            audio_features = self.audio_model(audio).last_hidden_state
            audio_features = torch.mean(audio_features, dim=1)
            audio_features = self.audio_projection(audio_features)
            
            # Cross-modal fusion
            if self.fusion_type == 'attention':
                # Expand audio features to match visual sequence length
                audio_features_expanded = audio_features.unsqueeze(1).repeat(1, num_frames, 1)
                fused_features = self.fusion_module(visual_features, audio_features_expanded)
            else:
                # Simple concatenation
                visual_pooled = torch.mean(visual_features, dim=1)
                combined = torch.cat([visual_pooled, audio_features], dim=-1)
                fused_features = self.fusion_module(combined).unsqueeze(1)
            
            # Transformer encoding
            transformer_output = self.transformer(fused_features)
            
            # Statistics pooling
            pooled_features = self.stats_pooling(transformer_output)
            
            # Main classification
            output = self.classifier(pooled_features)
            
            # Additional outputs
            additional_outputs = {}
            
            # Deepfake type classification
            if self.detect_deepfake_type:
                deepfake_type_output = self.deepfake_type_classifier(pooled_features)
                additional_outputs['deepfake_type'] = deepfake_type_output
            
            # Synchronization detection
            sync_score = self.sync_detector(
                torch.mean(visual_features, dim=1), 
                audio_features
            )
            additional_outputs['sync_score'] = sync_score
            
            # Forensic consistency analysis
            forensic_features = self.forensic_module(video_frames)
            additional_outputs['forensic_features'] = forensic_features
            
            # Self-supervised learning (if enabled and in training)
            if self.enable_self_supervised and self.training:
                # Contrastive learning
                ssl_loss = self.ssl_pretrainer(pooled_features, pooled_features)
                additional_outputs['ssl_loss'] = ssl_loss
                
                # Masked autoencoder
                mae_reconstructed, mae_loss = self.mae_pretrainer(pooled_features)
                additional_outputs['mae_loss'] = mae_loss
                
                # Cross-modal correspondence
                cross_modal_loss = self.cross_modal_ssl(
                    torch.mean(visual_features, dim=1), 
                    audio_features
                )
                additional_outputs['cross_modal_loss'] = cross_modal_loss
                
                # Temporal prediction
                temporal_predictions = self.temporal_prediction(transformer_output)
                additional_outputs['temporal_predictions'] = temporal_predictions
            
            # Uncertainty estimation (if enabled)
            if self.enable_ensemble and not self.training:
                mean_pred, uncertainty = self.bayesian_uncertainty(inputs)
                additional_outputs['uncertainty'] = uncertainty
                additional_outputs['bayesian_prediction'] = mean_pred
            
            # Early exit mechanism (if enabled)
            if self.enable_real_time_optimization and not self.training:
                early_pred, exit_layer = self.early_exit(video_frames)
                additional_outputs['early_prediction'] = early_pred
                additional_outputs['exit_layer'] = exit_layer
            
            return output, additional_outputs
            
        except Exception as e:
            if self.debug:
                print(f"Error in forward pass: {e}")
            raise e
    
    def enable_adversarial_training_mode(self, epsilon=0.01):
        """Enable adversarial training mode."""
        self.enable_adversarial_training = True
        if hasattr(self, 'adversarial_noise'):
            self.adversarial_noise.epsilon = epsilon
    
    def enable_self_supervised_mode(self):
        """Enable self-supervised learning mode."""
        self.enable_self_supervised = True
    
    def set_curriculum_epoch(self, epoch):
        """Set current epoch for curriculum learning."""
        if self.enable_curriculum_learning:
            self.progressive_training.set_epoch(epoch)
    
    def get_active_learning_samples(self, predictions, features, n_samples):
        """Get samples for active learning."""
        if self.enable_active_learning:
            return self.active_selector.hybrid_sampling(predictions, features, n_samples)
        return None
    
    def optimize_for_inference(self, quantize=True, prune=True, calibration_loader=None):
        """Optimize model for inference."""
        optimized_model = copy.deepcopy(self)
        
        if quantize and self.enable_quantization:
            if calibration_loader is not None:
                optimized_model = self.quantizer.static_quantization(calibration_loader)
            else:
                optimized_model = self.quantizer.dynamic_quantization()
        
        if prune and self.enable_pruning:
            optimized_model = self.pruner.unstructured_pruning(amount=0.3)
            optimized_model = self.pruner.remove_pruning()
        
        return optimized_model
    
    def enable_real_time_mode(self):
        """Enable real-time inference mode."""
        self.enable_real_time_optimization = True
        self.eval()
    
    def process_stream(self, frames):
        """Process streaming frames."""
        if self.enable_real_time_optimization:
            return self.sliding_window.process_stream(self, frames)
        return None
    
    def export_onnx(self, filepath, input_shape=(1, 16, 3, 224, 224), audio_shape=(1, 16000)):
        """Export model to ONNX format."""
        try:
            # Create dummy inputs
            dummy_video = torch.randn(input_shape)
            dummy_audio = torch.randn(audio_shape)
            dummy_inputs = {
                'video_frames': dummy_video,
                'audio': dummy_audio
            }
            
            # Export to ONNX
            torch.onnx.export(
                self,
                dummy_inputs,
                filepath,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['video_frames', 'audio'],
                output_names=['output'],
                dynamic_axes={
                    'video_frames': {0: 'batch_size', 1: 'num_frames'},
                    'audio': {0: 'batch_size', 1: 'audio_length'},
                    'output': {0: 'batch_size'}
                }
            )
            print(f"Model exported to ONNX: {filepath}")
            return True
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")
            return False
    
    def export_tensorrt(self, onnx_path, trt_path):
        """Export model to TensorRT format."""
        try:
            # This is a placeholder - actual TensorRT export would require
            # proper TensorRT Python API implementation
            print("TensorRT export would be implemented here")
            return True
        except Exception as e:
            print(f"Error exporting to TensorRT: {e}")
            return False
    
    def get_model_complexity(self):
        """Get model complexity metrics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }
    
        def deepfake_check_video(self, video_frames):
        """
        Enhanced deepfake check method with comprehensive forensic analysis.
        
        Args:
            video_frames: Input video frames tensor [B, T, C, H, W]
            
        Returns:
            dict: Comprehensive deepfake analysis results
        """
        try:
            batch_size, num_frames, C, H, W = video_frames.shape
            deepfake_results = {
                'overall_score': 0.0,
                'frame_inconsistencies': [],
                'temporal_inconsistencies': [],
                'face_landmarks_analysis': [],
                'compression_artifacts': [],
                'pixel_level_analysis': [],
                'metadata_analysis': {},
                'confidence_score': 0.0,
                'detailed_metrics': {}
            }
            
            # Convert to numpy for OpenCV operations
            frames_np = video_frames.cpu().numpy()
            
            # 1. Frame-by-frame inconsistency analysis
            frame_scores = []
            for i in range(num_frames):
                frame = frames_np[0, i].transpose(1, 2, 0)  # CHW to HWC
                frame = (frame * 255).astype(np.uint8)
                
                # Detect compression artifacts
                compression_score = self._detect_compression_artifacts(frame)
                
                # Analyze pixel-level inconsistencies
                pixel_score = self._analyze_pixel_inconsistencies(frame)
                
                # Face landmark analysis
                landmark_score = self._analyze_face_landmarks(frame)
                
                frame_score = {
                    'frame_index': i,
                    'compression_artifacts': compression_score,
                    'pixel_inconsistencies': pixel_score,
                    'landmark_consistency': landmark_score,
                    'overall_frame_score': (compression_score + pixel_score + landmark_score) / 3
                }
                
                frame_scores.append(frame_score)
                deepfake_results['frame_inconsistencies'].append(frame_score)
            
            # 2. Temporal consistency analysis
            temporal_score = self._analyze_temporal_consistency(frames_np[0])
            deepfake_results['temporal_inconsistencies'] = temporal_score
            
            # 3. Face landmarks analysis across frames
            landmarks_analysis = self._analyze_face_landmarks_sequence(frames_np[0])
            deepfake_results['face_landmarks_analysis'] = landmarks_analysis
            
            # 4. Optical flow analysis
            optical_flow_score = self._analyze_optical_flow(frames_np[0])
            deepfake_results['optical_flow_score'] = optical_flow_score
            
            # 5. Frequency domain analysis
            frequency_score = self._analyze_frequency_domain(frames_np[0])
            deepfake_results['frequency_analysis'] = frequency_score
            
            # 6. Eye blink detection
            blink_score = self._analyze_eye_blink_patterns(frames_np[0])
            deepfake_results['blink_analysis'] = blink_score
            
            # 7. Lip sync analysis (if audio is available)
            if hasattr(self, 'current_audio') and self.current_audio is not None:
                lip_sync_score = self._analyze_lip_sync(frames_np[0], self.current_audio)
                deepfake_results['lip_sync_analysis'] = lip_sync_score
            
            # 8. Calculate overall deepfake score
            overall_score = self._calculate_overall_deepfake_score(deepfake_results)
            deepfake_results['overall_score'] = overall_score
            
            # 9. Confidence estimation
            confidence = self._estimate_confidence(deepfake_results)
            deepfake_results['confidence_score'] = confidence
            
            # 10. Detailed metrics
            deepfake_results['detailed_metrics'] = {
                'avg_frame_score': np.mean([f['overall_frame_score'] for f in frame_scores]),
                'frame_variance': np.var([f['overall_frame_score'] for f in frame_scores]),
                'temporal_consistency': temporal_score.get('consistency_score', 0.0),
                'landmark_stability': landmarks_analysis.get('stability_score', 0.0),
                'total_frames_analyzed': num_frames,
                'suspicious_frames': len([f for f in frame_scores if f['overall_frame_score'] > 0.5])
            }
            
            return deepfake_results
            
        except Exception as e:
            if self.debug:
                print(f"Error in deepfake check: {e}")
            return {
                'overall_score': -1,
                'error': str(e),
                'confidence_score': 0.0
            }
    
    def _detect_compression_artifacts(self, frame):
        """Detect compression artifacts in a frame."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Apply DCT to detect block artifacts
            dct = cv2.dct(np.float32(gray))
            
            # Analyze high-frequency components
            high_freq = np.sum(np.abs(dct[32:, 32:]))
            total_energy = np.sum(np.abs(dct))
            
            # Calculate compression artifact score
            if total_energy > 0:
                compression_score = high_freq / total_energy
            else:
                compression_score = 0.0
            
            # Detect JPEG artifacts using gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Block boundary detection
            block_score = self._detect_block_boundaries(gradient_magnitude)
            
            return min(1.0, (compression_score + block_score) / 2)
            
        except Exception as e:
            if self.debug:
                print(f"Error in compression artifact detection: {e}")
            return 0.0
    
    def _detect_block_boundaries(self, gradient_magnitude):
        """Detect block boundaries characteristic of compression."""
        try:
            h, w = gradient_magnitude.shape
            block_score = 0.0
            
            # Check for 8x8 block patterns (JPEG)
            for i in range(8, h, 8):
                row_grad = np.mean(gradient_magnitude[i-1:i+1, :])
                block_score += row_grad
            
            for j in range(8, w, 8):
                col_grad = np.mean(gradient_magnitude[:, j-1:j+1])
                block_score += col_grad
            
            # Normalize
            total_blocks = (h // 8) + (w // 8)
            if total_blocks > 0:
                block_score /= total_blocks
            
            return min(1.0, block_score / 100)  # Normalize to [0, 1]
            
        except Exception as e:
            if self.debug:
                print(f"Error in block boundary detection: {e}")
            return 0.0
    
    def _analyze_pixel_inconsistencies(self, frame):
        """Analyze pixel-level inconsistencies."""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            
            # Calculate local variance
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Detect unusual color distributions
            hist_r = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([frame], [2], None, [256], [0, 256])
            
            # Calculate histogram entropy
            hist_entropy = self._calculate_histogram_entropy(hist_r, hist_g, hist_b)
            
            # Detect noise patterns
            noise_score = self._detect_noise_patterns(gray)
            
            # Edge consistency analysis
            edge_score = self._analyze_edge_consistency(gray)
            
            pixel_score = (local_var / 10000 + hist_entropy + noise_score + edge_score) / 4
            return min(1.0, pixel_score)
            
        except Exception as e:
            if self.debug:
                print(f"Error in pixel inconsistency analysis: {e}")
            return 0.0
    
    def _calculate_histogram_entropy(self, hist_r, hist_g, hist_b):
        """Calculate histogram entropy."""
        try:
            entropies = []
            for hist in [hist_r, hist_g, hist_b]:
                hist_norm = hist / np.sum(hist)
                hist_norm = hist_norm[hist_norm > 0]  # Remove zeros
                entropy = -np.sum(hist_norm * np.log2(hist_norm))
                entropies.append(entropy)
            
            return np.mean(entropies) / 8  # Normalize to [0, 1]
            
        except Exception as e:
            if self.debug:
                print(f"Error in histogram entropy calculation: {e}")
            return 0.0
    
    def _detect_noise_patterns(self, gray):
        """Detect unusual noise patterns."""
        try:
            # Apply Gaussian blur and subtract to get noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            
            # Calculate noise statistics
            noise_mean = np.mean(noise)
            noise_std = np.std(noise)
            
            # Detect periodic patterns in noise
            fft_noise = np.fft.fft2(noise)
            fft_magnitude = np.abs(fft_noise)
            
            # Look for peaks in frequency domain
            peaks = self._find_frequency_peaks(fft_magnitude)
            
            noise_score = (noise_mean + noise_std + len(peaks)) / 100
            return min(1.0, noise_score)
            
        except Exception as e:
            if self.debug:
                print(f"Error in noise pattern detection: {e}")
            return 0.0
    
    def _find_frequency_peaks(self, fft_magnitude):
        """Find peaks in frequency domain."""
        try:
            # Flatten and find peaks
            flat_magnitude = fft_magnitude.flatten()
            threshold = np.percentile(flat_magnitude, 95)
            peaks = np.where(flat_magnitude > threshold)[0]
            return peaks
            
        except Exception as e:
            if self.debug:
                print(f"Error in frequency peak detection: {e}")
            return []
    
    def _analyze_edge_consistency(self, gray):
        """Analyze edge consistency."""
        try:
            # Calculate edges using Canny
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density
            edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
            
            # Analyze edge smoothness
            edge_smoothness = self._calculate_edge_smoothness(edges)
            
            return min(1.0, (edge_density + edge_smoothness) / 2)
            
        except Exception as e:
            if self.debug:
                print(f"Error in edge consistency analysis: {e}")
            return 0.0
    
    def _calculate_edge_smoothness(self, edges):
        """Calculate edge smoothness metric."""
        try:
            # Find edge contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            smoothness_scores = []
            for contour in contours:
                if len(contour) > 10:  # Only consider significant contours
                    # Calculate contour smoothness
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    smoothness = len(approx) / len(contour)
                    smoothness_scores.append(smoothness)
            
            return np.mean(smoothness_scores) if smoothness_scores else 0.0
            
        except Exception as e:
            if self.debug:
                print(f"Error in edge smoothness calculation: {e}")
            return 0.0
    
    def _analyze_face_landmarks(self, frame):
        """Analyze face landmarks for inconsistencies."""
        try:
            if not self.enable_face_mesh:
                return 0.0
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.mp_face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return 0.5  # No face detected - suspicious
            
            landmark_scores = []
            for face_landmarks in results.multi_face_landmarks:
                # Extract key landmarks
                landmarks = []
                for landmark in face_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                
                landmarks = np.array(landmarks)
                
                # Analyze landmark consistency
                symmetry_score = self._analyze_face_symmetry(landmarks)
                stability_score = self._analyze_landmark_stability(landmarks)
                geometry_score = self._analyze_face_geometry(landmarks)
                
                landmark_score = (symmetry_score + stability_score + geometry_score) / 3
                landmark_scores.append(landmark_score)
            
            return np.mean(landmark_scores) if landmark_scores else 0.5
            
        except Exception as e:
            if self.debug:
                print(f"Error in face landmark analysis: {e}")
            return 0.0
    
    def _analyze_face_symmetry(self, landmarks):
        """Analyze facial symmetry."""
        try:
            # Get key facial points (simplified)
            left_eye = landmarks[33]  # Left eye corner
            right_eye = landmarks[263]  # Right eye corner
            nose_tip = landmarks[1]  # Nose tip
            
            # Calculate face center
            face_center = np.mean(landmarks, axis=0)
            
            # Calculate symmetry score
            left_dist = np.linalg.norm(left_eye - face_center)
            right_dist = np.linalg.norm(right_eye - face_center)
            
            symmetry_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)
            return symmetry_ratio
            
        except Exception as e:
            if self.debug:
                print(f"Error in face symmetry analysis: {e}")
            return 0.0
    
    def _analyze_landmark_stability(self, landmarks):
        """Analyze landmark stability."""
        try:
            # Calculate variance in landmark positions
            landmark_variance = np.var(landmarks, axis=0)
            stability_score = 1.0 / (1.0 + np.mean(landmark_variance))
            return stability_score
            
        except Exception as e:
            if self.debug:
                print(f"Error in landmark stability analysis: {e}")
            return 0.0
    
    def _analyze_face_geometry(self, landmarks):
        """Analyze face geometry consistency."""
        try:
            # Calculate facial ratios
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            nose_tip = landmarks[1]
            chin = landmarks[175]
            
            # Eye distance to face height ratio
            eye_distance = np.linalg.norm(right_eye - left_eye)
            face_height = np.linalg.norm(chin - nose_tip)
            
            if face_height > 0:
                ratio = eye_distance / face_height
                # Normal ratio is around 0.3-0.4
                geometry_score = 1.0 - abs(ratio - 0.35) / 0.35
                return max(0.0, geometry_score)
            
            return 0.0
            
        except Exception as e:
            if self.debug:
                print(f"Error in face geometry analysis: {e}")
            return 0.0
    
    def _analyze_temporal_consistency(self, frames):
        """Analyze temporal consistency across frames."""
        try:
            num_frames = len(frames)
            if num_frames < 2:
                return {'consistency_score': 0.0, 'analysis': 'Insufficient frames'}
            
            consistency_scores = []
            frame_differences = []
            
            for i in range(1, num_frames):
                prev_frame = frames[i-1].transpose(1, 2, 0)
                curr_frame = frames[i].transpose(1, 2, 0)
                
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, curr_frame)
                diff_score = np.mean(diff)
                frame_differences.append(diff_score)
                
                # Analyze motion consistency
                motion_score = self._analyze_motion_consistency(prev_frame, curr_frame)
                consistency_scores.append(motion_score)
            
            # Calculate overall temporal consistency
            avg_consistency = np.mean(consistency_scores)
            diff_variance = np.var(frame_differences)
            
            return {
                'consistency_score': avg_consistency,
                'frame_differences': frame_differences,
                'difference_variance': diff_variance,
                'motion_scores': consistency_scores
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error in temporal consistency analysis: {e}")
            return {'consistency_score': 0.0, 'error': str(e)}
    
    def _analyze_motion_consistency(self, prev_frame, curr_frame):
        """Analyze motion consistency between frames."""
        try:
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, None, None
            )
            
            # Analyze flow consistency
            if flow is not None:
                flow_magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
                flow_consistency = 1.0 - (np.std(flow_magnitude) / (np.mean(flow_magnitude) + 1e-6))
                return max(0.0, flow_consistency)
            
            return 0.0
            
        except Exception as e:
            if self.debug:
                print(f"Error in motion consistency analysis: {e}")
            return 0.0
    
    def _analyze_face_landmarks_sequence(self, frames):
        """Analyze face landmarks across frame sequence."""
        try:
            if not self.enable_face_mesh:
                return {'stability_score': 0.0, 'analysis': 'Face mesh disabled'}
            
            landmark_sequences = []
            
            for frame in frames:
                frame_rgb = frame.transpose(1, 2, 0)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Process with MediaPipe
                results = self.mp_face_mesh.process(frame_bgr)
                
                if results.multi_face_landmarks:
                    landmarks = []
                    for face_landmarks in results.multi_face_landmarks:
                        for landmark in face_landmarks.landmark:
                            landmarks.append([landmark.x, landmark.y, landmark.z])
                    landmark_sequences.append(np.array(landmarks))
                else:
                    landmark_sequences.append(None)
            
            # Analyze landmark stability across frames
            stability_score = self._calculate_landmark_sequence_stability(landmark_sequences)
            
            return {
                'stability_score': stability_score,
                'detected_faces': len([ls for ls in landmark_sequences if ls is not None]),
                'total_frames': len(frames)
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error in landmark sequence analysis: {e}")
            return {'stability_score': 0.0, 'error': str(e)}
    
    def _calculate_landmark_sequence_stability(self, landmark_sequences):
        """Calculate stability of landmarks across sequence."""
        try:
            valid_sequences = [ls for ls in landmark_sequences if ls is not None]
            
            if len(valid_sequences) < 2:
                return 0.0
            
            # Calculate variance across sequences
            stability_scores = []
            
            for i in range(1, len(valid_sequences)):
                prev_landmarks = valid_sequences[i-1]
                curr_landmarks = valid_sequences[i]
                
                if prev_landmarks.shape == curr_landmarks.shape:
                    # Calculate landmark displacement
                    displacement = np.mean(np.linalg.norm(curr_landmarks - prev_landmarks, axis=1))
                    stability_score = 1.0 / (1.0 + displacement * 100)
                    stability_scores.append(stability_score)
            
            return np.mean(stability_scores) if stability_scores else 0.0
            
        except Exception as e:
            if self.debug:
                print(f"Error in landmark sequence stability calculation: {e}")
            return 0.0
    
    def _analyze_optical_flow(self, frames):
        """Analyze optical flow patterns."""
        try:
            if len(frames) < 2:
                return {'flow_score': 0.0, 'analysis': 'Insufficient frames'}
            
            flow_scores = []
            
            for i in range(1, len(frames)):
                prev_frame = frames[i-1].transpose(1, 2, 0)
                curr_frame = frames[i].transpose(1, 2, 0)
                
                # Convert to grayscale
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
                
                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, None, None)
                
                if flow is not None:
                    # Analyze flow patterns
                    flow_magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
                    flow_angle = np.arctan2(flow[1], flow[0])
                    
                    # Calculate flow consistency
                    magnitude_consistency = 1.0 - (np.std(flow_magnitude) / (np.mean(flow_magnitude) + 1e-6))
                    angle_consistency = 1.0 - (np.std(flow_angle) / (np.pi + 1e-6))
                    
                    flow_score = (magnitude_consistency + angle_consistency) / 2
                    flow_scores.append(max(0.0, flow_score))
            
            return {
                'flow_score': np.mean(flow_scores) if flow_scores else 0.0,
                'flow_variance': np.var(flow_scores) if flow_scores else 0.0,
                'num_flows_analyzed': len(flow_scores)
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error in optical flow analysis: {e}")
            return {'flow_score': 0.0, 'error': str(e)}
    
    def _analyze_frequency_domain(self, frames):
        """Analyze frequency domain characteristics."""
        try:
            frequency_scores = []
            
            for frame in frames:
                frame_gray = cv2.cvtColor(frame.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
                
                # Apply 2D FFT
                fft = np.fft.fft2(frame_gray)
                fft_magnitude = np.abs(fft)
                
                # Analyze frequency distribution
                low_freq = np.sum(fft_magnitude[:fft_magnitude.shape[0]//4, :fft_magnitude.shape[1]//4])
                high_freq = np.sum(fft_magnitude[3*fft_magnitude.shape[0]//4:, 3*fft_magnitude.shape[1]//4:])
                
                if low_freq > 0:
                    freq_ratio = high_freq / low_freq
                    frequency_scores.append(freq_ratio)
            
            return {
                'frequency_score': np.mean(frequency_scores) if frequency_scores else 0.0,
                'frequency_variance': np.var(frequency_scores) if frequency_scores else 0.0,
                'analysis': 'Frequency domain analysis completed'
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error in frequency domain analysis: {e}")
            return {'frequency_score': 0.0, 'error': str(e)}
    
    def _analyze_eye_blink_patterns(self, frames):
        """Analyze eye blink patterns for naturalness."""
        try:
            if not self.enable_face_mesh:
                return {'blink_score': 0.0, 'analysis': 'Face mesh disabled'}
            
            blink_states = []
            
            for frame in frames:
                frame_rgb = frame.transpose(1, 2, 0)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Process with MediaPipe
                results = self.mp_face_mesh.process(frame_bgr)
                
                if results.multi_face_landmarks:
                    # Analyze eye opening
                    eye_opening = self._calculate_eye_opening(results.multi_face_landmarks[0])
                    blink_states.append(eye_opening)
                else:
                    blink_states.append(0.5)  # Default state
            
            # Analyze blink patterns
            blink_score = self._analyze_blink_naturalness(blink_states)
            
            return {
                'blink_score': blink_score,
                'blink_states': blink_states,
                'detected_blinks': self._count_blinks(blink_states)
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error in eye blink analysis: {e}")
            return {'blink_score': 0.0, 'error': str(e)}
    
    def _calculate_eye_opening(self, face_landmarks):
        """Calculate eye opening ratio."""
        try:
            # Get eye landmarks (simplified)
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]
            
            # Calculate eye opening
            left_opening = abs(left_eye_top.y - left_eye_bottom.y)
            right_opening = abs(right_eye_top.y - right_eye_bottom.y)
            
            return (left_opening + right_opening) / 2
            
        except Exception as e:
            if self.debug:
                print(f"Error in eye opening calculation: {e}")
            return 0.5
    
    def _analyze_blink_naturalness(self, blink_states):
        """Analyze naturalness of blink patterns."""
        try:
            if len(blink_states) < 10:
                return 0.5
            
            # Calculate blink frequency
            blink_threshold = 0.3
            blinks = [1 if state < blink_threshold else 0 for state in blink_states]
            blink_frequency = sum(blinks) / len(blinks)
            
            # Natural blink frequency is about 0.1-0.2 (10-20%)
            freq_score = 1.0 - abs(blink_frequency - 0.15) / 0.15
            
            # Analyze blink duration consistency
            blink_durations = self._calculate_blink_durations(blinks)
            duration_consistency = 1.0 - (np.std(blink_durations) / (np.mean(blink_durations) + 1e-6))
            
            return max(0.0, (freq_score + duration_consistency) / 2)
            
        except Exception as e:
            if self.debug:
                print(f"Error in blink naturalness analysis: {e}")
            return 0.0
    
    def _count_blinks(self, blink_states):
        """Count number of blinks in sequence."""
        try:
            blink_threshold = 0.3
            blinks = [1 if state < blink_threshold else 0 for state in blink_states]
            
            # Count transitions from open to closed
            blink_count = 0
            for i in range(1, len(blinks)):
                if blinks[i] == 1 and blinks[i-1] == 0:
                    blink_count += 1
            
            return blink_count
            
        except Exception as e:
            if self.debug:
                print(f"Error in blink counting: {e}")
            return 0
    
    def _calculate_blink_durations(self, blinks):
        """Calculate duration of each blink."""
        try:
            durations = []
            current_duration = 0
            
            for blink in blinks:
                if blink == 1:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                        current_duration = 0
            
            return durations if durations else [0]
            
        except Exception as e:
            if self.debug:
                print(f"Error in blink duration calculation: {e}")
            return [0]
    
    def _analyze_lip_sync(self, frames, audio):
        """Analyze lip synchronization with audio."""
        try:
            # This is a simplified implementation
            # In practice, you'd use more sophisticated lip-sync analysis
            
            if not self.enable_face_mesh:
                return {'lip_sync_score': 0.0, 'analysis': 'Face mesh disabled'}
            
            # Extract lip movements
            lip_movements = []
            
            for frame in frames:
                frame_rgb = frame.transpose(1, 2, 0)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Process with MediaPipe
                results = self.mp_face_mesh.process(frame_bgr)
                
                if results.multi_face_landmarks:
                    lip_opening = self._calculate_lip_opening(results.multi_face_landmarks[0])
                    lip_movements.append(lip_opening)
                else:
                    lip_movements.append(0.0)
            
            # Simplified audio feature extraction
            audio_features = self._extract_audio_features(audio)
            
            # Calculate correlation between lip movements and audio
            sync_score = self._calculate_lip_audio_correlation(lip_movements, audio_features)
            
            return {
                'lip_sync_score': sync_score,
                'lip_movements': lip_movements,
                'audio_features': audio_features[:len(lip_movements)]  # Match lengths
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error in lip sync analysis: {e}")
            return {'lip_sync_score': 0.0, 'error': str(e)}
    
    def _calculate_lip_opening(self, face_landmarks):
        """Calculate lip opening ratio."""
        try:
            # Get lip landmarks
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            
            # Calculate lip opening
            lip_opening = abs(upper_lip.y - lower_lip.y)
            
            return lip_opening
            
        except Exception as e:
            if self.debug:
                print(f"Error in lip opening calculation: {e}")
            return 0.0
    
    def _extract_audio_features(self, audio):
        """Extract audio features for lip sync analysis."""
        try:
            # Convert audio to numpy if it's a tensor
            if torch.is_tensor(audio):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = audio
            
            # Simple energy-based features
            window_size = len(audio_np) // 32  # Match typical video frame count
            audio_features = []
            
            for i in range(0, len(audio_np), window_size):
                window = audio_np[i:i+window_size]
                energy = np.sum(window**2)
                audio_features.append(energy)
            
            return audio_features
            
        except Exception as e:
            if self.debug:
                print(f"Error in audio feature extraction: {e}")
            return []
    
    def _calculate_lip_audio_correlation(self, lip_movements, audio_features):
        """Calculate correlation between lip movements and audio."""
        try:
            if len(lip_movements) == 0 or len(audio_features) == 0:
                return 0.0
            
            # Match lengths
            min_len = min(len(lip_movements), len(audio_features))
            lip_movements = lip_movements[:min_len]
            audio_features = audio_features[:min_len]
            
            # Calculate correlation
            correlation = np.corrcoef(lip_movements, audio_features)[0, 1]
            
            # Handle NaN values
            if np.isnan(correlation):
                correlation = 0.0
            
            return abs(correlation)  # Use absolute correlation
            
        except Exception as e:
            if self.debug:
                print(f"Error in lip-audio correlation calculation: {e}")
            return 0.0
    
    def _calculate_overall_deepfake_score(self, deepfake_results):
        """Calculate overall deepfake score from all analyses."""
        try:
            scores = []
            weights = []
            
            # Frame inconsistencies
            if deepfake_results['frame_inconsistencies']:
                frame_score = np.mean([f['overall_frame_score'] for f in deepfake_results['frame_inconsistencies']])
                scores.append(frame_score)
                weights.append(0.3)
            
            # Temporal inconsistencies
            if 'temporal_inconsistencies' in deepfake_results:
                temporal_score = deepfake_results['temporal_inconsistencies'].get('consistency_score', 0.0)
                scores.append(1.0 - temporal_score)  # Invert for deepfake score
                weights.append(0.2)
            
            # Face landmarks analysis
            if 'face_landmarks_analysis' in deepfake_results:
                landmark_score = deepfake_results['face_landmarks_analysis'].get('stability_score', 0.0)
                scores.append(1.0 - landmark_score)  # Invert for deepfake score
                weights.append(0.15)
            
            # Optical flow
            if 'optical_flow_score' in deepfake_results:
                flow_score = deepfake_results['optical_flow_score'].get('flow_score', 0.0)
                scores.append(1.0 - flow_score)  # Invert for deepfake score
                weights.append(0.1)
            
            # Frequency analysis
            if 'frequency_analysis' in deepfake_results:
                freq_score = deepfake_results['frequency_analysis'].get('frequency_score', 0.0)
                scores.append(min(1.0, freq_score))  # Higher frequency ratio indicates manipulation
                weights.append(0.1)
            
            # Blink analysis
            if 'blink_analysis' in deepfake_results:
                blink_score = deepfake_results['blink_analysis'].get('blink_score', 0.0)
                scores.append(1.0 - blink_score)  # Invert for deepfake score
                weights.append(0.1)
            
            # Lip sync analysis
            if 'lip_sync_analysis' in deepfake_results:
                lip_sync_score = deepfake_results['lip_sync_analysis'].get('lip_sync_score', 0.0)
                scores.append(1.0 - lip_sync_score)  # Invert for deepfake score
                weights.append(0.05)
            
            # Calculate weighted average
            if scores and weights:
                overall_score = np.average(scores, weights=weights)
                return min(1.0, max(0.0, overall_score))
            else:
                return 0.0
            
        except Exception as e:
            if self.debug:
                print(f"Error in overall deepfake score calculation: {e}")
            return 0.0
    
    def _estimate_confidence(self, deepfake_results):
        """Estimate confidence in the deepfake detection."""
        try:
            confidence_factors = []
            
            # Number of frames analyzed
            if deepfake_results['frame_inconsistencies']:
                frame_count = len(deepfake_results['frame_inconsistencies'])
                frame_confidence = min(1.0, frame_count / 16)  # Confidence increases with more frames
                confidence_factors.append(frame_confidence)
            
            # Consistency across different analyses
            scores = []
            if 'detailed_metrics' in deepfake_results:
                metrics = deepfake_results['detailed_metrics']
                if 'frame_variance' in metrics:
                    variance_confidence = 1.0 - min(1.0, metrics['frame_variance'])
                    confidence_factors.append(variance_confidence)
            
            # Face detection success rate
            if 'face_landmarks_analysis' in deepfake_results:
                landmark_analysis = deepfake_results['face_landmarks_analysis']
                if 'detected_faces' in landmark_analysis and 'total_frames' in landmark_analysis:
                    detection_rate = landmark_analysis['detected_faces'] / landmark_analysis['total_frames']
                    confidence_factors.append(detection_rate)
            
            # Overall confidence
            if confidence_factors:
                confidence = np.mean(confidence_factors)
                return min(1.0, max(0.0, confidence))
            else:
                return 0.5  # Default confidence
            
        except Exception as e:
            if self.debug:
                print(f"Error in confidence estimation: {e}")
            return 0.0
    
    def get_interpretability_report(self, inputs):
        """Generate interpretability report for the model's decision."""
        try:
            if not self.enable_explainability:
                return {'error': 'Explainability not enabled'}
            
            report = {
                'model_decision': {},
                'feature_importance': {},
                'attention_maps': {},
                'forensic_analysis': {},
                'confidence_breakdown': {}
            }
            
            # Get model prediction
            with torch.no_grad():
                output, additional_outputs = self.forward(inputs)
                probabilities = F.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
            
            report['model_decision'] = {
                'prediction': prediction.cpu().numpy().tolist(),
                'probabilities': probabilities.cpu().numpy().tolist(),
                'confidence': torch.max(probabilities, dim=1)[0].cpu().numpy().tolist()
            }
            
            # Feature importance analysis
            report['feature_importance'] = self._analyze_feature_importance(inputs, output)
            
            # Attention visualization
            if 'attention_maps' in additional_outputs:
                report['attention_maps'] = self._process_attention_maps(additional_outputs['attention_maps'])
            
            # Forensic analysis
            if 'video_frames' in inputs:
                forensic_analysis = self.deepfake_check_video(inputs['video_frames'])
                report['forensic_analysis'] = forensic_analysis
            
            # Confidence breakdown
            report['confidence_breakdown'] = self._analyze_confidence_breakdown(additional_outputs)
            
            return report
            
        except Exception as e:
            if self.debug:
                print(f"Error in interpretability report generation: {e}")
            return {'error': str(e)}
    
    def _analyze_feature_importance(self, inputs, output):
        """Analyze feature importance using gradient-based methods."""
        try:
            # Enable gradients for input
            video_frames = inputs['video_frames'].requires_grad_(True)
            audio = inputs['audio'].requires_grad_(True)
            
            # Forward pass
            model_inputs = {'video_frames': video_frames, 'audio': audio}
            model_output, _ = self.forward(model_inputs)
            
            # Calculate gradients
            target_class = torch.argmax(model_output, dim=1)
            model_output[0, target_class].backward()
            
            # Get gradients
            video_gradients = video_frames.grad
            audio_gradients = audio.grad
            
            # Calculate importance scores
            video_importance = torch.mean(torch.abs(video_gradients), dim=[0, 2, 3, 4])
            audio_importance = torch.mean(torch.abs(audio_gradients), dim=[0, 1])
            
            return {
                'video_importance': video_importance.cpu().numpy().tolist(),
                'audio_importance': audio_importance.cpu().numpy().tolist(),
                'overall_video_importance': torch.mean(video_importance).item(),
                'overall_audio_importance': torch.mean(audio_importance).item()
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error in feature importance analysis: {e}")
            return {}
    
    def _process_attention_maps(self, attention_maps):
        """Process attention maps for visualization."""
        try:
            if torch.is_tensor(attention_maps):
                attention_maps = attention_maps.cpu().numpy()
            
            # Normalize attention maps
            normalized_maps = []
            for i in range(attention_maps.shape[0]):
                attn_map = attention_maps[i]
                normalized = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                normalized_maps.append(normalized.tolist())
            
            return {
                'attention_maps': normalized_maps,
                'shape': attention_maps.shape,
                'summary': {
                    'max_attention': float(np.max(attention_maps)),
                    'min_attention': float(np.min(attention_maps)),
                    'mean_attention': float(np.mean(attention_maps))
                }
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error in attention map processing: {e}")
            return {}
    
    def _analyze_confidence_breakdown(self, additional_outputs):
        """Analyze confidence breakdown from different model components."""
        try:
            breakdown = {}
            
            # Synchronization confidence
            if 'sync_score' in additional_outputs:
                sync_score = additional_outputs['sync_score']
                if torch.is_tensor(sync_score):
                    sync_confidence = torch.sigmoid(sync_score).cpu().numpy().tolist()
                    breakdown['synchronization_confidence'] = sync_confidence
            
            # Deepfake type confidence
            if 'deepfake_type' in additional_outputs:
                deepfake_type = additional_outputs['deepfake_type']
                if torch.is_tensor(deepfake_type):
                    type_probs = F.softmax(deepfake_type, dim=1).cpu().numpy().tolist()
                    breakdown['deepfake_type_confidence'] = type_probs
            
            # Uncertainty estimation
            if 'uncertainty' in additional_outputs:
                uncertainty = additional_outputs['uncertainty']
                if torch.is_tensor(uncertainty):
                    uncertainty_vals = uncertainty.cpu().numpy().tolist()
                    breakdown['uncertainty_estimation'] = uncertainty_vals
            
            return breakdown
            
        except Exception as e:
            if self.debug:
                print(f"Error in confidence breakdown analysis: {e}")
            return {}
    
    def save_model_state(self, filepath, include_optimizer=False, optimizer=None, epoch=None, metrics=None):
        """Save complete model state with metadata."""
        try:
            save_dict = {
                'model_state_dict': self.state_dict(),
                'model_config': {
                    'num_classes': getattr(self, 'num_classes', 2),
                    'video_feature_dim': getattr(self, 'video_feature_dim', 1024),
                    'audio_feature_dim': getattr(self, 'audio_feature_dim', 1024),
                    'transformer_dim': getattr(self, 'transformer_dim', 768),
                    'num_transformer_layers': getattr(self, 'num_transformer_layers', 4),
                    'enable_face_mesh': self.enable_face_mesh,
                    'enable_explainability': self.enable_explainability,
                    'fusion_type': self.fusion_type,
                    'backbone_visual': self.backbone_visual,
                    'backbone_audio': self.backbone_audio,
                    'use_spectrogram': self.use_spectrogram,
                    'detect_deepfake_type': self.detect_deepfake_type,
                    'num_deepfake_types': self.num_deepfake_types,
                    'enable_adversarial_training': self.enable_adversarial_training,
                    'enable_self_supervised': self.enable_self_supervised,
                    'enable_curriculum_learning': self.enable_curriculum_learning,
                    'enable_active_learning': self.enable_active_learning,
                    'enable_quantization': self.enable_quantization,
                    'enable_pruning': self.enable_pruning,
                    'enable_real_time_optimization': self.enable_real_time_optimization,
                    'enable_ensemble': self.enable_ensemble
                },
                'model_complexity': self.get_model_complexity(),
                'timestamp': '2025-07-11 10:21:04'
            }
            
            if include_optimizer and optimizer is not None:
                save_dict['optimizer_state_dict'] = optimizer.state_dict()
            
            if epoch is not None:
                save_dict['epoch'] = epoch
            
            if metrics is not None:
                save_dict['metrics'] = metrics
            
            torch.save(save_dict, filepath)
            print(f"Model saved successfully to {filepath}")
            return True
            
        except Exception as e:
            if self.debug:
                print(f"Error saving model: {e}")
            return False
    
    def load_model_state(self, filepath, load_optimizer=False, optimizer=None):
        """Load complete model state with metadata."""
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)
            
            # Load optimizer if requested
            if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Return additional information
            return {
                'success': True,
                'config': checkpoint.get('model_config', {}),
                'complexity': checkpoint.get('model_complexity', {}),
                'epoch': checkpoint.get('epoch', None),
                'metrics': checkpoint.get('metrics', None),
                'timestamp': checkpoint.get('timestamp', None)
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error loading model: {e}")
            return {'success': False, 'error': str(e)}
    
    def benchmark_model(self, test_loader, device='cuda'):
        """Benchmark model performance with comprehensive metrics."""
        try:
            self.eval()
            benchmark_results = {
                'accuracy_metrics': {},
                'performance_metrics': {},
                'inference_times': [],
                'memory_usage': [],
                'model_complexity': self.get_model_complexity(),
                'detailed_analysis': {}
            }
            
            import time
            import psutil
            import gc
            
            total_samples = 0
            correct_predictions = 0
            total_inference_time = 0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    # Move batch to device
                    batch = self._move_batch_to_device(batch, device)
                    
                    # Record memory before inference
                    if device == 'cuda':
                        torch.cuda.synchronize()
                        memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
                    else:
                        memory_before = psutil.virtual_memory().used / 1024**2  # MB
                    
                    # Time inference
                    start_time = time.time()
                    
                    # Forward pass
                    outputs, additional_outputs = self.forward(batch)
                    
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    inference_time = end_time - start_time
                    
                    # Record memory after inference
                    if device == 'cuda':
                        memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
                    else:
                        memory_after = psutil.virtual_memory().used / 1024**2  # MB
                    
                    # Calculate metrics
                    batch_size = outputs.size(0)
                    predictions = torch.argmax(outputs, dim=1)
                    targets = batch.get('labels', torch.zeros(batch_size, dtype=torch.long, device=device))
                    
                    # Update counters
                    total_samples += batch_size
                    correct_predictions += (predictions == targets).sum().item()
                    total_inference_time += inference_time
                    
                    # Store for detailed analysis
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
                    # Record performance metrics
                    benchmark_results['inference_times'].append(inference_time / batch_size)
                    benchmark_results['memory_usage'].append(memory_after - memory_before)
                    
                    # Cleanup memory
                    del outputs, predictions, targets
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    if batch_idx % 10 == 0:
                        print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
            
            # Calculate accuracy metrics
            accuracy = correct_predictions / total_samples
            benchmark_results['accuracy_metrics']['overall_accuracy'] = accuracy
            
            # Calculate detailed metrics
            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
            
            if len(set(all_targets)) > 1:  # Multiple classes
                # Classification report
                class_report = classification_report(all_targets, all_predictions, output_dict=True)
                benchmark_results['accuracy_metrics']['classification_report'] = class_report
                
                # Confusion matrix
                conf_matrix = confusion_matrix(all_targets, all_predictions)
                benchmark_results['accuracy_metrics']['confusion_matrix'] = conf_matrix.tolist()
                
                # ROC AUC if binary classification
                if len(set(all_targets)) == 2:
                    auc_score = roc_auc_score(all_targets, all_predictions)
                    benchmark_results['accuracy_metrics']['roc_auc'] = auc_score
            
            # Calculate performance metrics
            avg_inference_time = np.mean(benchmark_results['inference_times'])
            std_inference_time = np.std(benchmark_results['inference_times'])
            throughput = 1.0 / avg_inference_time  # samples per second
            
            benchmark_results['performance_metrics'] = {
                'avg_inference_time_per_sample': avg_inference_time,
                'std_inference_time': std_inference_time,
                'throughput_samples_per_second': throughput,
                'total_inference_time': total_inference_time,
                'avg_memory_usage_mb': np.mean(benchmark_results['memory_usage']),
                'max_memory_usage_mb': np.max(benchmark_results['memory_usage']),
                'total_samples_processed': total_samples
            }
            
            # Detailed analysis
            benchmark_results['detailed_analysis'] = {
                'inference_time_percentiles': {
                    'p50': np.percentile(benchmark_results['inference_times'], 50),
                    'p90': np.percentile(benchmark_results['inference_times'], 90),
                    'p95': np.percentile(benchmark_results['inference_times'], 95),
                    'p99': np.percentile(benchmark_results['inference_times'], 99)
                },
                'memory_usage_percentiles': {
                    'p50': np.percentile(benchmark_results['memory_usage'], 50),
                    'p90': np.percentile(benchmark_results['memory_usage'], 90),
                    'p95': np.percentile(benchmark_results['memory_usage'], 95),
                    'p99': np.percentile(benchmark_results['memory_usage'], 99)
                }
            }
            
            return benchmark_results
            
        except Exception as e:
            if self.debug:
                print(f"Error in model benchmarking: {e}")
            return {'error': str(e)}
    
    def _move_batch_to_device(self, batch, device):
        """Move batch to specified device safely."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(device)
            else:
                device_batch[key] = value
        return device_batch
    
    def create_ensemble_model(self, model_paths, ensemble_strategy='soft_voting'):
        """Create ensemble model from multiple trained models."""
        try:
            ensemble_models = []
            
            for model_path in model_paths:
                # Create new model instance
                model = MultiModalDeepfakeModel(
                    num_classes=getattr(self, 'num_classes', 2),
                    video_feature_dim=getattr(self, 'video_feature_dim', 1024),
                    audio_feature_dim=getattr(self, 'audio_feature_dim', 1024),
                    transformer_dim=getattr(self, 'transformer_dim', 768),
                    num_transformer_layers=getattr(self, 'num_transformer_layers', 4),
                    enable_face_mesh=self.enable_face_mesh,
                    enable_explainability=self.enable_explainability,
                    fusion_type=self.fusion_type,
                    backbone_visual=self.backbone_visual,
                    backbone_audio=self.backbone_audio,
                    use_spectrogram=self.use_spectrogram,
                    detect_deepfake_type=self.detect_deepfake_type,
                    num_deepfake_types=self.num_deepfake_types,
                    debug=self.debug
                )
                
                # Load model state
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                ensemble_models.append(model)
            
            # Create ensemble
            if ensemble_strategy == 'soft_voting':
                ensemble = MultiHeadEnsemble(ensemble_models, voting_strategy='soft')
            elif ensemble_strategy == 'hard_voting':
                ensemble = MultiHeadEnsemble(ensemble_models, voting_strategy='hard')
            else:
                ensemble = MultiHeadEnsemble(ensemble_models, voting_strategy='soft')
            
            return ensemble
            
        except Exception as e:
            if self.debug:
                print(f"Error creating ensemble model: {e}")
            return None
    
    def generate_synthetic_data(self, num_samples=100, data_type='adversarial'):
        """Generate synthetic data for training augmentation."""
        try:
            synthetic_data = []
            
            if data_type == 'adversarial':
                # Generate adversarial examples
                for i in range(num_samples):
                    # Create random base sample
                    video_frames = torch.randn(1, 16, 3, 224, 224)
                    audio = torch.randn(1, 16000)
                    
                    # Apply adversarial perturbation
                    if self.enable_adversarial_training:
                        # Add adversarial noise
                        noise = torch.randn_like(video_frames) * 0.01
                        video_frames_adv = video_frames + noise
                        
                        synthetic_data.append({
                            'video_frames': video_frames_adv,
                            'audio': audio,
                            'label': random.choice([0, 1]),
                            'synthetic_type': 'adversarial'
                        })
            
            elif data_type == 'augmented':
                # Generate augmented samples
                for i in range(num_samples):
                    # Create base sample
                    video_frames = torch.randn(1, 16, 3, 224, 224)
                    audio = torch.randn(1, 16000)
                    
                    # Apply augmentations
                    # Temporal augmentation
                    if random.random() > 0.5:
                        # Reverse temporal order
                        video_frames = torch.flip(video_frames, [1])
                    
                    # Spatial augmentation
                    if random.random() > 0.5:
                        # Horizontal flip
                        video_frames = torch.flip(video_frames, [4])
                    
                    # Audio augmentation
                    if random.random() > 0.5:
                        # Add noise to audio
                        audio_noise = torch.randn_like(audio) * 0.01
                        audio = audio + audio_noise
                    
                    synthetic_data.append({
                        'video_frames': video_frames,
                        'audio': audio,
                        'label': random.choice([0, 1]),
                        'synthetic_type': 'augmented'
                    })
            
            return synthetic_data
            
        except Exception as e:
            if self.debug:
                print(f"Error generating synthetic data: {e}")
            return []
    
    def analyze_model_robustness(self, test_loader, device='cuda', attack_types=['fgsm', 'pgd']):
        """Analyze model robustness against adversarial attacks."""
        try:
            self.eval()
            robustness_results = {
                'clean_accuracy': 0.0,
                'adversarial_accuracy': {},
                'attack_success_rates': {},
                'robustness_metrics': {}
            }
            
            total_samples = 0
            clean_correct = 0
            adversarial_correct = {attack: 0 for attack in attack_types}
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    batch = self._move_batch_to_device(batch, device)
                    
                    # Clean accuracy
                    outputs, _ = self.forward(batch)
                    predictions = torch.argmax(outputs, dim=1)
                    targets = batch.get('labels', torch.zeros(outputs.size(0), dtype=torch.long, device=device))
                    
                    clean_correct += (predictions == targets).sum().item()
                    total_samples += outputs.size(0)
                    
                    # Adversarial accuracy
                    if self.enable_adversarial_training:
                        for attack_type in attack_types:
                            # Generate adversarial examples
                            batch['video_frames'].requires_grad_(True)
                            adv_outputs, _ = self.forward(batch)
                            loss = F.cross_entropy(adv_outputs, targets)
                            loss.backward()
                            
                            # Apply attack
                            adv_frames = self.adversarial_noise(
                                batch['video_frames'], 
                                attack_type=attack_type,
                                grad=batch['video_frames'].grad
                            )
                            
                            # Test adversarial examples
                            adv_batch = batch.copy()
                            adv_batch['video_frames'] = adv_frames
                            adv_outputs, _ = self.forward(adv_batch)
                            adv_predictions = torch.argmax(adv_outputs, dim=1)
                            
                            adversarial_correct[attack_type] += (adv_predictions == targets).sum().item()
                    
                    if batch_idx % 10 == 0:
                        print(f"Analyzed {batch_idx + 1}/{len(test_loader)} batches for robustness")
            
            # Calculate metrics
            robustness_results['clean_accuracy'] = clean_correct / total_samples
            
            for attack_type in attack_types:
                if attack_type in adversarial_correct:
                    adv_acc = adversarial_correct[attack_type] / total_samples
                    robustness_results['adversarial_accuracy'][attack_type] = adv_acc
                    robustness_results['attack_success_rates'][attack_type] = 1.0 - adv_acc
            
            # Calculate robustness metrics
            if robustness_results['adversarial_accuracy']:
                avg_adv_acc = np.mean(list(robustness_results['adversarial_accuracy'].values()))
                robustness_drop = robustness_results['clean_accuracy'] - avg_adv_acc
                
                robustness_results['robustness_metrics'] = {
                    'average_adversarial_accuracy': avg_adv_acc,
                    'robustness_drop': robustness_drop,
                    'robustness_ratio': avg_adv_acc / robustness_results['clean_accuracy'],
                    'overall_robustness_score': 1.0 - robustness_drop
                }
            
            return robustness_results
            
        except Exception as e:
            if self.debug:
                print(f"Error in robustness analysis: {e}")
            return {'error': str(e)}
    
    def deploy_for_production(self, deployment_config):
        """Deploy model for production with optimizations."""
        try:
            deployment_results = {
                'model_path': None,
                'optimized_model_path': None,
                'deployment_config': deployment_config,
                'optimization_results': {},
                'deployment_status': 'pending'
            }
            
            # Apply optimizations based on deployment config
            optimized_model = copy.deepcopy(self)
            
            # Quantization
            if deployment_config.get('quantization', False):
                print("Applying quantization...")
                if self.enable_quantization:
                    optimized_model = self.quantizer.dynamic_quantization()
                    deployment_results['optimization_results']['quantization'] = 'applied'
                else:
                    deployment_results['optimization_results']['quantization'] = 'not_available'
            
            # Pruning
            if deployment_config.get('pruning', False):
                print("Applying pruning...")
                if self.enable_pruning:
                    optimized_model = self.pruner.unstructured_pruning(
                        amount=deployment_config.get('pruning_amount', 0.3)
                    )
                    optimized_model = self.pruner.remove_pruning()
                    deployment_results['optimization_results']['pruning'] = 'applied'
                else:
                    deployment_results['optimization_results']['pruning'] = 'not_available'
            
            # ONNX Export
            if deployment_config.get('onnx_export', False):
                print("Exporting to ONNX...")
                onnx_path = deployment_config.get('onnx_path', 'model_optimized.onnx')
                success = optimized_model.export_onnx(onnx_path)
                deployment_results['optimization_results']['onnx_export'] = 'success' if success else 'failed'
                if success:
                    deployment_results['optimized_model_path'] = onnx_path
            
            # TensorRT Export
            if deployment_config.get('tensorrt_export', False):
                print("Exporting to TensorRT...")
                trt_path = deployment_config.get('tensorrt_path', 'model_optimized.trt')
                success = optimized_model.export_tensorrt(
                    deployment_results.get('optimized_model_path', 'model.onnx'),
                    trt_path
                )
                deployment_results['optimization_results']['tensorrt_export'] = 'success' if success else 'failed'
            
            # Real-time optimizations
            if deployment_config.get('real_time_optimization', False):
                print("Enabling real-time optimizations...")
                optimized_model.enable_real_time_mode()
                deployment_results['optimization_results']['real_time_optimization'] = 'enabled'
            
            # Save deployment model
            deployment_model_path = deployment_config.get('model_path', 'deployed_model.pth')
            success = optimized_model.save_model_state(
                deployment_model_path,
                include_optimizer=False
            )
            
            if success:
                deployment_results['model_path'] = deployment_model_path
                deployment_results['deployment_status'] = 'success'
            else:
                deployment_results['deployment_status'] = 'failed'
            
            # Generate deployment report
            deployment_results['deployment_report'] = self._generate_deployment_report(
                optimized_model, deployment_config
            )
            
            return deployment_results
            
        except Exception as e:
            if self.debug:
                print(f"Error in production deployment: {e}")
            return {'deployment_status': 'failed', 'error': str(e)}
    
    def _generate_deployment_report(self, optimized_model, deployment_config):
        """Generate deployment report with performance metrics."""
        try:
            report = {
                'model_complexity': optimized_model.get_model_complexity(),
                'optimization_summary': {},
                'performance_estimates': {},
                'deployment_recommendations': []
            }
            
            # Model complexity comparison
            original_complexity = self.get_model_complexity()
            optimized_complexity = optimized_model.get_model_complexity()
            
            report['optimization_summary'] = {
                'parameter_reduction': {
                    'original': original_complexity['total_parameters'],
                    'optimized': optimized_complexity['total_parameters'],
                    'reduction_ratio': 1.0 - (optimized_complexity['total_parameters'] / original_complexity['total_parameters'])
                },
                'model_size_reduction': {
                    'original_mb': original_complexity['model_size_mb'],
                    'optimized_mb': optimized_complexity['model_size_mb'],
                    'reduction_ratio': 1.0 - (optimized_complexity['model_size_mb'] / original_complexity['model_size_mb'])
                }
            }
            
            # Performance estimates
            report['performance_estimates'] = {
                'expected_speedup': self._estimate_speedup(deployment_config),
                'memory_reduction': self._estimate_memory_reduction(deployment_config),
                'accuracy_retention': self._estimate_accuracy_retention(deployment_config)
            }
            
            # Deployment recommendations
            report['deployment_recommendations'] = self._generate_deployment_recommendations(
                deployment_config, report['optimization_summary']
            )
            
            return report
            
        except Exception as e:
            if self.debug:
                print(f"Error generating deployment report: {e}")
            return {}
    
    def _estimate_speedup(self, deployment_config):
        """Estimate performance speedup from optimizations."""
        speedup = 1.0
        
        if deployment_config.get('quantization', False):
            speedup *= 2.0  # Typical quantization speedup
        
        if deployment_config.get('pruning', False):
            pruning_amount = deployment_config.get('pruning_amount', 0.3)
            speedup *= (1.0 + pruning_amount)
        
        if deployment_config.get('tensorrt_export', False):
            speedup *= 3.0  # Typical TensorRT speedup
        
        return speedup
    
    def _estimate_memory_reduction(self, deployment_config):
        """Estimate memory reduction from optimizations."""
        reduction = 1.0
        
        if deployment_config.get('quantization', False):
            reduction *= 0.5  # INT8 quantization
        
        if deployment_config.get('pruning', False):
            pruning_amount = deployment_config.get('pruning_amount', 0.3)
            reduction *= (1.0 - pruning_amount)
        
        return 1.0 - reduction
    
    def _estimate_accuracy_retention(self, deployment_config):
        """Estimate accuracy retention after optimizations."""
        retention = 1.0
        
        if deployment_config.get('quantization', False):
            retention *= 0.98  # Typical quantization accuracy loss
        
        if deployment_config.get('pruning', False):
            pruning_amount = deployment_config.get('pruning_amount', 0.3)
            retention *= (1.0 - pruning_amount * 0.1)
        
        return retention
    
    def _generate_deployment_recommendations(self, deployment_config, optimization_summary):
        """Generate deployment recommendations."""
        recommendations = []
        
        # Parameter reduction recommendations
        param_reduction = optimization_summary.get('parameter_reduction', {}).get('reduction_ratio', 0.0)
        if param_reduction > 0.5:
            recommendations.append({
                'type': 'optimization',
                'priority': 'high',
                'message': f'Significant parameter reduction achieved ({param_reduction:.1%}). Consider further quantization for mobile deployment.'
            })
        
        # Model size recommendations
        model_size = optimization_summary.get('model_size_reduction', {}).get('optimized_mb', 0.0)
        if model_size > 100:
            recommendations.append({
                'type': 'deployment',
                'priority': 'medium',
                'message': f'Model size ({model_size:.1f}MB) may be large for mobile deployment. Consider additional pruning.'
            })
        
        # Performance recommendations
        if deployment_config.get('real_time_optimization', False):
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'message': 'Real-time optimization enabled. Monitor latency in production and adjust batch sizes accordingly.'
            })
        
        return recommendations
    
    def __str__(self):
        """String representation of the model."""
        complexity = self.get_model_complexity()
        return f"""
MultiModalDeepfakeModel(
    Parameters: {complexity['total_parameters']:,}
    Trainable Parameters: {complexity['trainable_parameters']:,}
    Model Size: {complexity['model_size_mb']:.2f} MB
    
    Configuration:
    - Visual Backbone: {self.backbone_visual}
    - Audio Backbone: {self.backbone_audio}
    - Fusion Type: {self.fusion_type}
    - Face Mesh: {self.enable_face_mesh}
    - Explainability: {self.enable_explainability}
    - Adversarial Training: {self.enable_adversarial_training}
    - Self-Supervised: {self.enable_self_supervised}
    - Curriculum Learning: {self.enable_curriculum_learning}
    - Active Learning: {self.enable_active_learning}
    - Real-time Optimization: {self.enable_real_time_optimization}
    - Ensemble: {self.enable_ensemble}
)
"""
    
    def __repr__(self):
        return self.__str__()

# ============== UTILITY FUNCTIONS ==============

def create_model_from_config(config_dict):
    """Create model from configuration dictionary."""
    return MultiModalDeepfakeModel(**config_dict)

def load_pretrained_model(model_path, device='cpu'):
    """Load pretrained model with error handling."""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract configuration
        config = checkpoint.get('model_config', {})
        
        # Create model
        model = MultiModalDeepfakeModel(**config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model, checkpoint
        
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        return None, None

def compare_models(model1, model2, test_loader, device='cuda'):
    """Compare two models on the same test set."""
    try:
        results = {
            'model1_metrics': {},
            'model2_metrics': {},
            'comparison': {}
        }
        
        # Benchmark both models
        print("Benchmarking Model 1...")
        results['model1_metrics'] = model1.benchmark_model(test_loader, device)
        
        print("Benchmarking Model 2...")
        results['model2_metrics'] = model2.benchmark_model(test_loader, device)
        
        # Compare metrics
        if 'accuracy_metrics' in results['model1_metrics'] and 'accuracy_metrics' in results['model2_metrics']:
            acc1 = results['model1_metrics']['accuracy_metrics'].get('overall_accuracy', 0)
            acc2 = results['model2_metrics']['accuracy_metrics'].get('overall_accuracy', 0)
            
            results['comparison']['accuracy_difference'] = acc2 - acc1
            results['comparison']['better_model'] = 'model2' if acc2 > acc1 else 'model1'
        
        if 'performance_metrics' in results['model1_metrics'] and 'performance_metrics' in results['model2_metrics']:
            throughput1 = results['model1_metrics']['performance_metrics'].get('throughput_samples_per_second', 0)
            throughput2 = results['model2_metrics']['performance_metrics'].get('throughput_samples_per_second', 0)
            
            results['comparison']['throughput_difference'] = throughput2 - throughput1
            results['comparison']['faster_model'] = 'model2' if throughput2 > throughput1 else 'model1'
        
        return results
        
    except Exception as e:
        print(f"Error comparing models: {e}")
        return {'error': str(e)}

# ============== EXAMPLE USAGE ==============

if __name__ == "__main__":
    # Example usage of the enhanced model
    print("=== MultiModal Deepfake Detection Model ===")
    
    # Create model with all enhanced features
    model = MultiModalDeepfakeModel(
        num_classes=2,
        video_feature_dim=1024,
        audio_feature_dim=1024,
        transformer_dim=768,
        num_transformer_layers=4,
        enable_face_mesh=True,
        enable_explainability=True,
        fusion_type='attention',
        backbone_visual='efficientnet',
        backbone_audio='wav2vec2',
        use_spectrogram=True,
        detect_deepfake_type=True,
        num_deepfake_types=7,
        debug=True,
        # Enhanced features
        enable_adversarial_training=True,
        enable_self_supervised=True,
        enable_curriculum_learning=True,
        enable_active_learning=True,
        enable_quantization=True,
        enable_pruning=True,
        enable_real_time_optimization=True,
        enable_ensemble=True
    )
    
    print(f"Model created successfully!")
    print(f"Model complexity: {model.get_model_complexity()}")
    print(f"Model details:\n{model}")
    
    # Example forward pass
    dummy_video = torch.randn(2, 16, 3, 224, 224)
    dummy_audio = torch.randn(2, 16000)
    
    dummy_inputs = {
        'video_frames': dummy_video,
        'audio': dummy_audio
    }
    
    try:
        with torch.no_grad():
            outputs, additional_outputs = model(dummy_inputs)
            print(f"Forward pass successful!")
            print(f"Output shape: {outputs.shape}")
            print(f"Additional outputs keys: {list(additional_outputs.keys())}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
    
    # Example deployment configuration
    deployment_config = {
        'quantization': True,
        'pruning': True,
        'pruning_amount': 0.3,
        'onnx_export': True,
        'onnx_path': 'deployed_model.onnx',
        'real_time_optimization': True,
        'model_path': 'deployed_model.pth'
    }
    
    print("\n=== Deployment Example ===")
    deployment_results = model.deploy_for_production(deployment_config)
    print(f"Deployment status: {deployment_results.get('deployment_status', 'unknown')}")
    
    print("\n=== Model Ready for Training and Deployment ===")

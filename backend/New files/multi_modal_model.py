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
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Combined features dimension
        combined_dim = transformer_dim
        if self.enable_explainability:
            combined_dim += 128 * 4  # ELA + metadata + sync + face embeddings
        
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
            self.component_weights = nn.Parameter(torch.ones(5), requires_grad=True)

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
            except Exception as e:
                print(f"⚠️ Warning: Failed to initialize face mesh: {e}")
                self.enable_face_mesh = False
        
        if self.debug:
            print(f"Model initialized with video_feature_dim={self.actual_video_feature_dim}, audio_feature_dim={self.actual_audio_feature_dim}")

    # Rest of the methods remain unchanged...

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
            else:
                explainability_features.append(torch.zeros(batch_size, 128, device=video_frames.device))
            
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
                all_explainability = torch.cat(explainability_features, dim=-1)
                
                # Weight each explainability component by learned weights
                weighted_components = {}
                for key, value in component_contributions.items():
                    if value is not None:
                        idx = list(component_contributions.keys()).index(key)
                        weight = F.softmax(self.component_weights, dim=0)[idx]
                        weighted_components[key] = value * weight
                
                # Create explainability vector by concatenating weighted features
                explainability_vector = torch.cat([value for value in weighted_components.values()], dim=-1)
                
                # Combine with transformer output
                final_features = torch.cat([transformer_output, all_explainability], dim=-1)
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
                    av_sync_features=audio_visual_sync
                )

            # Create results dictionary
            results = {
                'logits': output,
                'deepfake_type': deepfake_type_output,
                'deepfake_check': deepfake_check_results,
                'explanation': explanation_data,
                'component_weights': F.softmax(self.component_weights, dim=0) if self.enable_explainability else None,
            }

            return output, results
            
        except Exception as e:
            print(f"❌ Error in forward pass: {e}")
            # Return zero tensor with proper shape as fallback
            return torch.zeros((video_frames.size(0), 2), device=video_frames.device), {"error": str(e)}

    def deepfake_check_video(self, video_frames, original_video_frames, fake_periods, timestamps, 
                            original_audio=None, current_audio=None, ela_features=None, 
                            metadata_features=None, temporal_consistency=None, av_sync_features=None):
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
            'eye_blinking': 0
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
            
            # Run facial analysis for eye blinking if face_mesh is enabled
            if self.enable_face_mesh and video_frames is not None:
                batch_size = video_frames.size(0)
                for b in range(min(batch_size, 4)):  # Limit to first 4 samples for efficiency
                    blink_detected = False
                    for t in range(min(10, video_frames.size(1))):  # Check only a few frames
                        eye_blink = self.check_eye_blinking(video_frames[b, t])
                        if eye_blink:
                            blink_detected = True
                            break
                    
                    if not blink_detected:
                        # No blinking detected in this sequence
                        inconsistencies['eye_blinking'] += 1
                        explanation['issues_found'].append(f"No eye blinking detected in sample {b}")
                        
                explanation['detection_scores']['eye_blinking_score'] = inconsistencies['eye_blinking'] / max(1, batch_size)
                        
            # Calculate total inconsistency score and confidence
            total_inconsistencies = sum(inconsistencies.values())
            max_possible = len(inconsistencies) * 2  # Arbitrary scaling factor
            
            # Calculate confidence score from individual signals
            if explanation['detection_scores']:
                confidence_score = np.mean(list(explanation['detection_scores'].values()))
                explanation['confidence'] = min(0.99, confidence_score)
            else:
                explanation['confidence'] = total_inconsistencies / max_possible

            # Add total inconsistencies to explanation
            explanation['total_inconsistencies'] = total_inconsistencies
            
            # Debug log
            if self.debug:
                if isinstance(fake_periods, list):
                    print(f"Fake periods detected: {fake_periods[:5]}...")
                if isinstance(timestamps, list):
                    print(f"Timestamps analyzed: {timestamps[:5]}...")
                print(f"Total inconsistencies: {total_inconsistencies}")
                
            return total_inconsistencies, explanation
            
        except Exception as e:
            print(f"❌ Error in deepfake check: {e}")
            return 0, {"error": str(e)}

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

    def get_attention_maps(self, inputs):
        """
        Generate attention maps for visualization and explainability.
        
        This is useful for highlighting which parts of the input are most important
        for the model's decision.
        """
        try:
            video_frames = inputs['video_frames']
            batch_size, num_frames = video_frames.shape[:2]
            
            # Get feature activations
            video_frames_flat = video_frames.view(batch_size * num_frames, *video_frames.shape[2:])
            frame_features = self.visual_model.features(video_frames_flat)
            
            # Get weights from the last layer
            classifier_weights = self.classifier[0].weight.data
            
            # Generate class activation maps
            batch_cams = []
            
            for b in range(batch_size):
                frame_cams = []
                for t in range(num_frames):
                    idx = b * num_frames + t
                    features = frame_features[idx]
                    
                    # Create CAM for each class
                    cams = []
                    for class_idx in range(classifier_weights.shape[0]):
                        cam = torch.mm(classifier_weights[class_idx:class_idx+1], features.view(features.size(0), -1))
                        cam = cam.view(features.shape[1:])
                        
                        # Normalize CAM
                        cam = F.interpolate(
                            cam.unsqueeze(0).unsqueeze(0),
                            size=video_frames.shape[3:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                        
                        # Normalize between 0 and 1
                        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                        cams.append(cam)
                    
                    frame_cams.append(torch.stack(cams))
                batch_cams.append(torch.stack(frame_cams))
            
            return torch.stack(batch_cams)  # [batch, frames, classes, H, W]
        except Exception as e:
            if self.debug:
                print(f"Error generating attention maps: {e}")
            return None

    def extract_forensic_features(self, inputs):
        """
        Extract detailed forensic features for external analysis.
        
        Returns a dictionary of forensic features that can be used for
        more detailed analysis or visualization.
        """
        forensic_features = {}
        
        try:
            # Extract frame differences if original frames available
            if 'video_frames' in inputs and 'original_video_frames' in inputs and inputs['original_video_frames'] is not None:
                video_frames = inputs['video_frames']
                original_video_frames = inputs['original_video_frames']
                
                # Calculate frame differences
                min_batch = min(video_frames.size(0), original_video_frames.size(0))
                min_frames = min(video_frames.size(1), original_video_frames.size(1))
                
                diffs = []
                for b in range(min_batch):
                    batch_diffs = []
                    for t in range(min_frames):
                        diff = torch.abs(video_frames[b, t] - original_video_frames[b, t])
                        batch_diffs.append(diff.mean().item())
                    diffs.append(batch_diffs)
                
                forensic_features['frame_diffs'] = diffs
            
            # Extract noise analysis
            if 'video_frames' in inputs:
                video_frames = inputs['video_frames']
                batch_size, num_frames = video_frames.shape[:2]
                
                # Noise extraction (simple high-pass filter)
                noise_levels = []
                for b in range(batch_size):
                    frame_noise = []
                    for t in range(num_frames - 1):  # Analyze consecutive frames
                        current = video_frames[b, t]
                        next_frame = video_frames[b, t + 1]
                        
                        # Extract noise as high-frequency components
                        diff = torch.abs(current - next_frame)
                        noise_level = torch.mean(diff).item()
                        frame_noise.append(noise_level)
                    
                    noise_levels.append(frame_noise)
                
                forensic_features['noise_levels'] = noise_levels
            
            # Extract frequency domain analysis (FFT)
            if 'video_frames' in inputs:
                video_frames = inputs['video_frames']
                batch_size, num_frames = video_frames.shape[:2]
                
                fft_features = []
                for b in range(min(batch_size, 4)):  # Limit to 4 samples to save computation
                    frame_ffts = []
                    for t in range(min(num_frames, 5)):  # Limit to 5 frames
                        frame = video_frames[b, t].mean(dim=0)  # Convert to grayscale
                        frame_fft = torch.fft.fft2(frame).abs()
                        
                        # Calculate radial average
                        h, w = frame_fft.shape
                        center_h, center_w = h // 2, w // 2
                        Y, X = torch.meshgrid(torch.arange(h) - center_h, torch.arange(w) - center_w, indexing='ij')
                        R = torch.sqrt(X**2 + Y**2)
                        R = R.to(frame_fft.device)
                        
                        # Collect values at different radii
                        max_radius = min(center_h, center_w)
                        radial_profile = []
                        for r in range(1, max_radius, max(1, max_radius // 10)):
                            mask = ((R >= r - 0.5) & (R < r + 0.5))
                            if mask.sum() > 0:
                                radial_avg = frame_fft[mask].mean().item()
                                radial_profile.append(radial_avg)
                        
                        frame_ffts.append(radial_profile)
                    fft_features.append(frame_ffts)
                
                forensic_features['fft_features'] = fft_features
            
            # Extract compression artifacts
            if 'ela_features' in inputs and inputs['ela_features'] is not None:
                forensic_features['ela'] = inputs['ela_features'].cpu().numpy()
            
            # Add metadata features
            if 'metadata_features' in inputs and inputs['metadata_features'] is not None:
                forensic_features['metadata'] = inputs['metadata_features'].cpu().numpy()
            
            return forensic_features
            
        except Exception as e:
            if self.debug:
                print(f"Error extracting forensic features: {e}")
            return {"error": str(e)}

    def interpret_prediction(self, output, results):
        """
        Interpret the model's prediction and provide human-readable explanation.
        
        Args:
            output: Model output logits
            results: Dictionary with additional results
            
        Returns:
            Dictionary with interpretable results
        """
        interpretation = {
            "prediction": "real" if output.argmax(1)[0].item() == 0 else "fake",
            "confidence": F.softmax(output, dim=1)[0][output.argmax(1)[0]].item(),
            "explanation": "This video appears to be ",
            "evidence": []
        }
        
        # Add deepfake check results if available
        if 'deepfake_check' in results and results['deepfake_check'] is not None:
            inconsistencies = results['deepfake_check']
            interpretation["inconsistencies_found"] = inconsistencies
            
            # Add explanation from deepfake check
            if 'explanation' in results and results['explanation'] is not None:
                exp = results['explanation']
                
                # Add issues found
                if 'issues_found' in exp and exp['issues_found']:
                    interpretation["issues"] = exp['issues_found']
                    
                    # Add first few issues to the explanation
                    for issue in exp['issues_found'][:3]:
                        interpretation["evidence"].append(issue)
                
                # Add confidence
                if 'confidence' in exp:
                    interpretation["detection_confidence"] = exp['confidence']
        
        # Add deepfake type if available
        if 'deepfake_type' in results and results['deepfake_type'] is not None:
            deepfake_types = ['unknown', 'face_swap', 'face_reenactment', 'lip_sync', 
                             'audio_only', 'entire_synthesis', 'attribute_manipulation']
            
            type_idx = results['deepfake_type'].argmax(1)[0].item()
            type_confidence = F.softmax(results['deepfake_type'], dim=1)[0][type_idx].item()
            
            if type_idx > 0:  # Not unknown
                interpretation["deepfake_type"] = deepfake_types[type_idx]
                interpretation["type_confidence"] = type_confidence
                interpretation["explanation"] += f"manipulated using {deepfake_types[type_idx]} technique. "
        
        # Complete explanation based on prediction
        if interpretation["prediction"] == "real":
            interpretation["explanation"] += "authentic with no signs of manipulation."
        else:
            # Add evidence to explanation
            if interpretation["evidence"]:
                evidence_text = ". ".join(interpretation["evidence"])
                interpretation["explanation"] += f"fake based on the following evidence: {evidence_text}"
            else:
                interpretation["explanation"] += "fake based on analysis of visual and audio features."
        
        return interpretation                
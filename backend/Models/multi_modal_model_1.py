import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional, Union


class MultiModalDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2, video_feature_dim=1024, audio_feature_dim=1024, transformer_dim=768, 
                 num_transformer_layers=4, enable_face_mesh=False, debug=False):
        super(MultiModalDeepfakeModel, self).__init__()
        self.debug = debug
        self.enable_face_mesh = enable_face_mesh

        # EfficientNet for video frame features
        self.visual_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.visual_model.classifier = nn.Identity()
        self.video_projection = nn.Linear(1280, video_feature_dim)

        # Wav2Vec2 for audio features
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.audio_projection = nn.Linear(768, audio_feature_dim)

        # Fusion and transformer
        self.combined_projection = nn.Linear(video_feature_dim + audio_feature_dim, transformer_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=8, batch_first=True),
            num_layers=num_transformer_layers
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

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

        # Learnable inconsistency threshold
        self.deepfake_threshold = nn.Parameter(torch.tensor(20.0), requires_grad=True)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs: Dict[str, Union[torch.Tensor, List, None]]) -> Tuple[torch.Tensor, Optional[int]]:
        """
        Forward pass for the model.
        
        Args:
            inputs: Dictionary containing:
                - video_frames: tensor of shape [B, T, C, H, W]
                - audio: tensor of shape [B, L]
                - original_video_frames: optional tensor of original frames
                - original_audio: optional tensor of original audio
                - fake_periods: optional list of fake periods
                - timestamps: optional list of timestamps
                - fake_mask: optional list or tensor of fake masks
        
        Returns:
            tuple: (output logits, deepfake check result)
        """
        try:
            video_frames = inputs['video_frames']  # [B, T, C, H, W]
            audio = inputs['audio']                # [B, L]
            original_video_frames = inputs.get('original_video_frames', None)
            original_audio = inputs.get('original_audio', None)
            fake_periods = inputs.get('fake_periods', None)
            timestamps = inputs.get('timestamps', None)
            # Handle fake_mask regardless of whether it's a list or tensor
            fake_mask = inputs.get('fake_mask', None)

            batch_size, num_frames, C, H, W = video_frames.size()

            # Debugging shapes
            if self.debug:
                print(f"Video frames shape: {video_frames.shape}")
                print(f"Audio shape: {audio.shape}")
                if isinstance(fake_mask, list):
                    print(f"Fake mask: List of {len(fake_mask)} items")
                elif fake_mask is not None:
                    print(f"Fake mask shape: {fake_mask.shape}")

            # Visual features
            video_frames_flat = video_frames.view(batch_size * num_frames, C, H, W)
            visual_features = self.visual_model(video_frames_flat)
            visual_features = visual_features.view(batch_size, num_frames, -1)
            video_features = torch.mean(visual_features, dim=1)  # [B, 1280]
            video_features = self.video_projection(video_features)  # [B, video_feature_dim]

            # Normalize audio to [-1, 1] and ensure float32
            audio = audio.float()
            max_vals = torch.abs(audio).max(dim=1, keepdim=True)[0]
            # Avoid division by zero
            max_vals = torch.clamp(max_vals, min=1e-6)
            audio = audio / max_vals

            # Audio features
            with torch.no_grad():  # Optional: freeze wav2vec2 during initial training
                audio_features = self.audio_model(audio).last_hidden_state  # [B, T', 768]
            audio_features = torch.mean(audio_features, dim=1)          # [B, 768]
            audio_features = self.audio_projection(audio_features)      # [B, audio_feature_dim]

            # Combine and fuse features
            combined_features = torch.cat([video_features, audio_features], dim=-1)  # [B, video+audio]
            combined_features = self.combined_projection(combined_features)          # [B, transformer_dim]
            transformer_output = self.transformer(combined_features.unsqueeze(1)).squeeze(1)  # [B, transformer_dim]
            output = self.classifier(transformer_output)  # [B, num_classes]

            # Deepfake check during evaluation
            deepfake_check = None
            if not self.training:
                deepfake_check = self.deepfake_check_video(
                    video_frames, original_video_frames, fake_periods, timestamps, original_audio, audio
                )

            return output, deepfake_check
            
        except Exception as e:
            print(f"❌ Error in forward pass: {e}")
            # Return zero tensor with proper shape as fallback
            return torch.zeros((video_frames.size(0), 2), device=video_frames.device), None

    def deepfake_check_video(self, video_frames, original_video_frames, fake_periods, timestamps, original_audio=None, current_audio=None):
        """
        Check for signs of deepfake by comparing original and current frames/audio.
        
        Returns the number of inconsistencies detected.
        """
        inconsistencies = 0
        total_checks = 0

        try:
            # Check video inconsistencies if original frames are available
            if original_video_frames is not None:
                min_batch = min(video_frames.size(0), original_video_frames.size(0))
                min_frames = min(video_frames.size(1), original_video_frames.size(1))
                
                for b in range(min_batch):
                    for t in range(min_frames):
                        frame_diff = torch.abs(video_frames[b, t] - original_video_frames[b, t]).mean().item()
                        if frame_diff > self.deepfake_threshold.item():
                            inconsistencies += 1
                        total_checks += 1
                
                if self.debug:
                    print(f"Video inconsistencies: {inconsistencies}/{total_checks}")
                    
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
                    
                    for i, score in enumerate(audio_diff):
                        if self.debug:
                            print(f"Audio difference score for sample {i}: {score:.4f}")
                        if score > 0.2:  # Arbitrary threshold for audio difference
                            inconsistencies += 1
                            
                except Exception as audio_error:
                    if self.debug:
                        print(f"Error in audio comparison: {audio_error}")
                        
            # Optional: Check for facial landmarks inconsistencies using face mesh
            if self.enable_face_mesh and video_frames is not None:
                for b in range(min(4, video_frames.size(0))):  # Limit to first 4 samples for efficiency
                    for t in range(min(5, video_frames.size(1))):  # Check only a few frames
                        eye_blink = self.check_eye_blinking(video_frames[b, t])
                        if eye_blink is not None and not eye_blink:
                            # No blinking detected in this sequence
                            if self.debug:
                                print(f"No eye blinking detected in sample {b}, frame {t}")
                                
            # Debug log
            if self.debug:
                if isinstance(fake_periods, list):
                    print(f"Fake periods detected: {fake_periods[:5]}...")
                if isinstance(timestamps, list):
                    print(f"Timestamps analyzed: {timestamps[:5]}...")
                    
            return inconsistencies
            
        except Exception as e:
            print(f"❌ Error in deepfake check: {e}")
            return 0

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
            left_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in range(33, 42)])
            right_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in range(42, 51)])

            # Calculate eye openness
            left_eye_open = np.linalg.norm(left_eye[1] - left_eye[5])
            right_eye_open = np.linalg.norm(right_eye[1] - right_eye[5])
            
            # Normalize by inter-ocular distance
            inter_ocular_dist = np.linalg.norm(left_eye[0] - right_eye[3])
            
            if inter_ocular_dist > 0:
                left_eye_open /= inter_ocular_dist
                right_eye_open /= inter_ocular_dist
                
                # Return True if eyes are blinking (eyes are closed)
                return left_eye_open < 0.1 or right_eye_open < 0.1
            else:
                return None
                
        except Exception as e:
            if self.debug:
                print(f"Error in eye blinking detection: {e}")
            return None

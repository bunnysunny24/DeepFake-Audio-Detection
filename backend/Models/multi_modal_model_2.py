import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import mediapipe as mp
import numpy as np


class MultiModalDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2, video_feature_dim=1024, audio_feature_dim=1024, transformer_dim=768, num_transformer_layers=4):
        super(MultiModalDeepfakeModel, self).__init__()

        # EfficientNet for video frame features
        self.visual_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.visual_model.classifier = nn.Identity()  # Remove final classification layer
        self.video_projection = nn.Linear(1280, video_feature_dim)  # Project EfficientNet features

        # Wav2Vec 2.0 for audio features
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.audio_projection = nn.Linear(768, audio_feature_dim)  # Project Wav2Vec2 features

        # Linear layer for combined feature projection
        self.combined_projection = nn.Linear(video_feature_dim + audio_feature_dim, transformer_dim)

        # Transformer Encoder for feature fusion
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=8, batch_first=True),
            num_layers=num_transformer_layers
        )

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # Learnable threshold for deepfake checks
        self.deepfake_threshold = nn.Parameter(torch.tensor(20.0), requires_grad=True)

        # Mediapipe for facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def forward(self, video_frames, audio):
        batch_size, num_frames, C, H, W = video_frames.size()

        # Flatten frames and extract visual features
        video_frames_flat = video_frames.view(batch_size * num_frames, C, H, W)
        visual_features = self.visual_model(video_frames_flat)
        visual_features = visual_features.view(batch_size, num_frames, -1)  # Reshape back to batch size
        video_features = torch.mean(visual_features, dim=1)  # Aggregate over frames
        video_features = self.video_projection(video_features)  # Project to fixed size

        # Extract audio features
        audio_features = self.audio_model(audio).last_hidden_state
        audio_features = torch.mean(audio_features, dim=1)  # Aggregate over sequence
        audio_features = self.audio_projection(audio_features)  # Project to fixed size

        # Combine features
        combined_features = torch.cat([video_features, audio_features], dim=-1)
        combined_features = self.combined_projection(combined_features)  # Project for Transformer input

        # Pass through Transformer Encoder
        transformer_output = self.transformer(combined_features.unsqueeze(1)).squeeze(1)

        # Classify the combined features
        output = self.classifier(transformer_output)

        # Only perform deepfake analysis during evaluation mode
        deepfake_check = None
        if not self.training:
            deepfake_check = self.deepfake_check_video(video_frames)

        return output, deepfake_check

    def deepfake_check_video(self, video_frames):
        """
        Improved inconsistency check using a learnable threshold.
        """
        inconsistencies = 0
        total_checks = 0

        # Remove batch dim if present
        if video_frames.dim() == 5:
            frames = video_frames[0]  # (N, C, H, W)
        else:
            frames = video_frames  # Already 4D: (N, C, H, W)

        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]

            # Print shape if needed
            # print("Frame shape:", frame1.shape)

            # Convert CHW to HWC for NumPy
            frame1_np = frame1.permute(1, 2, 0).cpu().numpy()
            frame2_np = frame2.permute(1, 2, 0).cpu().numpy()

            diff = np.abs(frame1_np - frame2_np).mean()
            if diff > self.deepfake_threshold.item():  # Use learnable threshold
                inconsistencies += 1

            total_checks += 1

        return inconsistencies

    def check_eye_blinking(self, frame):
        """
        Improved blink detection using Mediapipe for facial landmarks.
        """
        frame_rgb = frame.permute(1, 2, 0).cpu().numpy()  # Convert frame to RGB
        results = self.mp_face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None  # No faces detected

        landmarks = results.multi_face_landmarks[0].landmark

        # Extract eye landmarks (normalized coordinates)
        left_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in range(33, 42)])
        right_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in range(42, 51)])

        # Calculate vertical distances for eye openings
        left_eye_open = np.linalg.norm(left_eye[1] - left_eye[5])
        right_eye_open = np.linalg.norm(right_eye[1] - right_eye[5])

        # Calculate inter-ocular distance for normalization
        inter_ocular_dist = np.linalg.norm(left_eye[0] - right_eye[3])

        # Normalize eye opening distances
        left_eye_open /= inter_ocular_dist
        right_eye_open /= inter_ocular_dist

        # Return True if either eye appears to be closed (threshold is arbitrary)
        return left_eye_open < 0.1 or right_eye_open < 0.1

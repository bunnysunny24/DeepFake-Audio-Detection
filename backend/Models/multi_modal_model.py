import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from torchvision.models import efficientnet_b0
import cv2
import dlib
import numpy as np

class MultiModalDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MultiModalDeepfakeModel, self).__init__()

        # EfficientNet for video frame features
        self.visual_model = efficientnet_b0(weights="IMAGENET1K_V1")
        self.visual_model.classifier = nn.Identity()  # Remove final classification layer

        # Wav2Vec 2.0 for audio features
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # Linear projection to align dimensions with Transformer’s d_model
        self.feature_projection = nn.Linear(2048, 768)

        # Transformer Encoder for combining features
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True),
            num_layers=4
        )

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # Facial landmark detector (dlib)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def forward(self, video_frames, audio):
        batch_size, num_frames, C, H, W = video_frames.size()

        # Flatten the video frames for EfficientNet input
        video_frames_flat = video_frames.view(batch_size * num_frames, C, H, W)
        visual_features = self.visual_model(video_frames_flat)

        # Reshape visual features back to batch size and num_frames
        visual_features = visual_features.view(batch_size, num_frames, -1)

        # Only perform deepfake analysis during evaluation mode
        deepfake_check = None
        if not self.training:
            deepfake_check = self.deepfake_check_video(video_frames)

        # Aggregate video features by averaging over frames
        video_features = torch.mean(visual_features, dim=1)

        # Extract audio features
        audio_features = self.audio_model(audio).last_hidden_state
        audio_features = torch.mean(audio_features, dim=1)

        # Combine video and audio features
        combined_features = torch.cat([video_features, audio_features], dim=1)

        # Project combined features
        combined_features = self.feature_projection(combined_features)

        # Pass combined features through the Transformer Encoder
        transformer_output = self.transformer(combined_features.unsqueeze(1)).squeeze(1)

        # Classify the combined features
        output = self.classifier(transformer_output)

        return output, deepfake_check

    def deepfake_check_video(self, video_frames):
        """
        Basic inconsistency check by comparing pixel differences across frames.
        Only called during eval/inference.
        """
        inconsistencies = 0
        total_checks = 0

        frames = video_frames.squeeze(0)  # Remove batch dim -> (N, C, H, W)

        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]

            if frame1.dim() == 4:
                frame1 = frame1[0]
            if frame2.dim() == 4:
                frame2 = frame2[0]
                
            # print(f"[DEBUG] frame1 shape: {frame1.shape}")

            frame1_np = frame1.permute(1, 2, 0).cpu().numpy()
            frame2_np = frame2.permute(1, 2, 0).cpu().numpy()
            

            diff = np.abs(frame1_np - frame2_np).mean()
            if diff > 20:  # Arbitrary threshold
                inconsistencies += 1

            total_checks += 1

        return inconsistencies

    def check_eye_blinking(self, landmarks):
        """
        Simple blink detection based on vertical eye distance change.
        """
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        eye_open_left = abs(left_eye[1][1] - left_eye[5][1])
        eye_open_right = abs(right_eye[1][1] - right_eye[5][1])

        return eye_open_left < 5 or eye_open_right < 5

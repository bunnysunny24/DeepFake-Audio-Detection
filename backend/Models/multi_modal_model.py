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
        self.visual_model = efficientnet_b0(pretrained=True)
        self.visual_model.classifier = nn.Identity()  # Remove final classification layer

        # Wav2Vec 2.0 for audio features
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # Transformer Encoder for combining features
        self.transformer = nn.Transformer(
            d_model=768,
            nhead=8,
            num_encoder_layers=4
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
        
        # Flatten the video frames for efficientNet input
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

        # Extract audio features (Wav2Vec2.0 model output)
        audio_features = self.audio_model(audio).last_hidden_state
        audio_features = torch.mean(audio_features, dim=1)

        # Combine video and audio features
        combined_features = torch.cat([video_features, audio_features], dim=1)

        # Pass combined features through the transformer
        combined_features = self.transformer(combined_features.unsqueeze(1)).squeeze(1)

        # Classify the combined features
        output = self.classifier(combined_features)

        return output, deepfake_check

    def deepfake_check_video(self, frames):
        """
        Check for inconsistencies in the video frames (e.g., facial landmarks, eye blinking patterns, etc.)
        Only called during eval/inference.
        """
        deepfake_inconsistency = False

        for frame in frames:
            frame_np = frame.permute(1, 2, 0).cpu().numpy()
            gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            if len(faces) == 0:
                continue

            # Check each face for blinking inconsistencies
            for face in faces:
                shape = self.predictor(gray, face)
                landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

                # Check for eye blinking using landmarks
                if self.check_eye_blinking(landmarks):
                    deepfake_inconsistency = True
                    break

        return deepfake_inconsistency

    def check_eye_blinking(self, landmarks):
        """
        A simple check to detect blinking based on the vertical eye distance change.
        """
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # Calculate eye openness by measuring vertical distance between eyelids
        eye_open_left = abs(left_eye[1][1] - left_eye[5][1])
        eye_open_right = abs(right_eye[1][1] - right_eye[5][1])

        # If the eye opening is below a threshold, consider it as blinking
        if eye_open_left < 5 or eye_open_right < 5:
            return True
        return False

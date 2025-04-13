import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, ViTModel

class MultiModalDeepfakeDetector(nn.Module):
    def __init__(self):
        super(MultiModalDeepfakeDetector, self).__init__()
        
        # Video branch: Vision Transformer
        self.video_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.video_fc = nn.Linear(self.video_model.config.hidden_size, 256)  # Reduce dimensionality
        
        # Audio branch: Wav2Vec2
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_fc = nn.Linear(self.audio_model.config.hidden_size, 256)  # Reduce dimensionality
        
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, video_frames, audio_features):
        # Video processing
        video_embeddings = self.video_model(pixel_values=video_frames).last_hidden_state  # (batch_size, seq_len, hidden_dim)
        video_embeddings = self.video_fc(video_embeddings[:, 0, :])  # CLS token
        
        # Audio processing
        audio_embeddings = self.audio_model(audio_features).last_hidden_state  # (batch_size, seq_len, hidden_dim)
        audio_embeddings = self.audio_fc(audio_embeddings[:, 0, :])  # CLS token
        
        # Cross-modal attention
        video_embeddings = video_embeddings.unsqueeze(0)  # Add sequence dimension
        audio_embeddings = audio_embeddings.unsqueeze(0)  # Add sequence dimension
        fused_embeddings, _ = self.attention(video_embeddings, audio_embeddings, audio_embeddings)
        
        # Classification
        logits = self.classifier(fused_embeddings.squeeze(0))
        return logits
import torch
import torch.nn as nn
import timm  # Pretrained models
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, cnn_dim, num_heads=4):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x1, x2):
        x2 = self.cnn_proj(x2)  # Project EfficientNet features to ViT space
        x1 = x1.unsqueeze(0)  # Convert to (seq, batch, dim) for attention
        x2 = x2.unsqueeze(0)
        attn_output, _ = self.multihead_attn(x1, x2, x2)
        return attn_output.squeeze(0) + x1.squeeze(0)  # Residual connection

class HybridViT_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridViT_CNN, self).__init__()
        
        # EfficientNet-B7 (CNN) for local features
        self.cnn = timm.create_model("tf_efficientnet_b7", pretrained=True, num_classes=0)
        cnn_out_dim = self.cnn.num_features
        
        # Vision Transformer (ViT) for global features
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        vit_out_dim = self.vit.num_features
        
        # Cross-Attention
        self.cross_attention = CrossAttention(embed_dim=vit_out_dim, cnn_dim=cnn_out_dim)
        
        # Feature processing layers
        self.segmented_fc = nn.Linear(cnn_out_dim, 256)
        self.degmented_fc = nn.Linear(cnn_out_dim, 256)
        self.heatmap_mlp = nn.Sequential(
            nn.Linear(224 * 224, 512), nn.ReLU(), nn.Linear(512, 256)
        )
        self.ear_fc = nn.Linear(1, 32)
        self.flow_fc = nn.Linear(2, 32)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        
        # Final classifier with gating mechanism
        self.fc = nn.Sequential(
            nn.Linear(vit_out_dim + cnn_out_dim + 256 + 256 + 256 + 32 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, segmented, degmented, heatmap, ear, optical_flow):
        cnn_features = self.cnn(image)
        vit_features = self.vit(image)
        fused_features = self.cross_attention(vit_features, cnn_features)
        
        segmented_features = self.segmented_fc(self.cnn(segmented))
        degmented_features = self.degmented_fc(self.cnn(degmented))
        heatmap_features = self.heatmap_mlp(heatmap.view(heatmap.size(0), -1))
        ear_features = self.ear_fc(ear.unsqueeze(1))
        flow_features = self.flow_fc(optical_flow)
        _, (lstm_output, _) = self.lstm(flow_features.unsqueeze(1))
        lstm_output = lstm_output.squeeze(0)
        
        final_features = torch.cat([
            fused_features, cnn_features, segmented_features, degmented_features,
            heatmap_features, ear_features, lstm_output
        ], dim=1)
        
        return self.fc(final_features)

# Model testing with dummy input
if __name__ == "__main__":
    model = HybridViT_CNN(num_classes=2)
    image = torch.randn(2, 3, 224, 224)
    segmented = torch.randn(2, 3, 224, 224)
    degmented = torch.randn(2, 3, 224, 224)
    heatmap = torch.randn(2, 1, 224, 224)
    ear = torch.randn(2, 1)
    optical_flow = torch.randn(2, 2)
    output = model(image, segmented, degmented, heatmap, ear, optical_flow)
    print(output.shape)  # Expected: (2, num_classes)

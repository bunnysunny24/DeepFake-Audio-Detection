import torch
import torch.nn as nn
import timm  # Pretrained models

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, cnn_dim):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, embed_dim)  # Adjust CNN output size
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5  # Scaling factor

    def forward(self, x1, x2):
        x2 = self.cnn_proj(x2)  # Project EfficientNet features to ViT space

        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)  # (batch_size, 768)
        return attn_output + x1  # Residual connection


class HybridViT_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridViT_CNN, self).__init__()

        # EfficientNet-B7 (CNN) for local features
        self.cnn = timm.create_model("tf_efficientnet_b7", pretrained=True, num_classes=0)
        cnn_out_dim = self.cnn.num_features

        # Vision Transformer (ViT) for global features
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        vit_out_dim = self.vit.num_features

        # MLP for Heatmaps
        self.mlp = nn.Sequential(
            nn.Linear(224 * 224, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # EAR Feature Processing
        self.ear_fc = nn.Linear(1, 32)  # Maps EAR value to 32D

        # Optical Flow Feature Processing
        self.flow_fc = nn.Linear(2, 32)  # Maps optical flow (2D) to 32D

        # LSTM for Optical Flow
        self.lstm = nn.LSTM(32, 64, batch_first=True)  # 64 hidden units

        # Cross-Attention
        self.cross_attention = CrossAttention(embed_dim=768, cnn_dim=2560)

        # Final Classifier
        self.fc = nn.Sequential(
            nn.Linear(vit_out_dim + cnn_out_dim + 256 + 32 + 64, 512),  # Updated size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, heatmap, ear, optical_flow):
        # CNN Features
        cnn_features = self.cnn(image)

        # ViT Features
        vit_features = self.vit(image)

        # Apply Cross-Attention
        fused_features = self.cross_attention(vit_features, cnn_features)

        # Heatmap Features
        heatmap = heatmap.view(heatmap.size(0), -1)  # Flatten heatmap
        heatmap_features = self.mlp(heatmap)

        # EAR Features
        ear_features = self.ear_fc(ear.unsqueeze(1))

        # Optical Flow Features
        flow_features = self.flow_fc(optical_flow)

        # Pass Optical Flow through LSTM
        _, (lstm_output, _) = self.lstm(flow_features.unsqueeze(1))
        lstm_output = lstm_output.squeeze(0)  # Extract last hidden state

        # Concatenate All Features
        final_features = torch.cat([fused_features, cnn_features, heatmap_features, ear_features, lstm_output], dim=1)

        # Fully Connected Layer
        output = self.fc(final_features)
        return output


# Test the model with dummy input
if __name__ == "__main__":
    model = HybridViT_CNN(num_classes=2)
    image = torch.randn(2, 3, 224, 224)  # Batch size of 2, RGB images
    heatmap = torch.randn(2, 1, 224, 224)  # Batch size of 2, single-channel heatmaps
    ear = torch.randn(2, 1)  # Batch size of 2, single EAR values
    optical_flow = torch.randn(2, 2)  # Batch size of 2, (x, y) optical flow values

    output = model(image, heatmap, ear, optical_flow)
    print(output.shape)  # Expected: (2, num_classes)

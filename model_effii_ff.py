import torch
import torch.nn as nn
import timm  # Pretrained models

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output + x1  # Residual Connection

class HybridViT_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridViT_CNN, self).__init__()

        # EfficientNet-B7 (CNN) for local features
        self.cnn = timm.create_model("efficientnet_b7", pretrained=True, num_classes=0)
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

        # Cross-Attention
        self.cross_attention = CrossAttention(embed_dim=vit_out_dim)

        # Final Classifier
        self.fc = nn.Sequential(
            nn.Linear(vit_out_dim + cnn_out_dim + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, heatmap):
        # CNN Features
        cnn_features = self.cnn(image)

        # ViT Features
        vit_features = self.vit(image)

        # Apply Cross-Attention
        fused_features = self.cross_attention(vit_features, cnn_features)

        # Heatmap Features
        heatmap = heatmap.view(heatmap.size(0), -1)  # Flatten heatmap
        heatmap_features = self.mlp(heatmap)

        # Concatenate All Features
        final_features = torch.cat([fused_features, cnn_features, heatmap_features], dim=1)

        # Fully Connected Layer
        output = self.fc(final_features)
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SkinColorAnalyzer(nn.Module):
    def __init__(self, feature_dim=32):
        super(SkinColorAnalyzer, self).__init__()
        self.feature_dim = feature_dim
        
    def detect_skin(self, frames):
        """
        Detect skin regions in RGB frames using proper tensor operations.
        Args:
            frames: torch.Tensor of shape [batch_size, num_frames, 3, height, width]
        Returns:
            torch.Tensor: Binary skin mask
        """
        device = frames.device
        dtype = frames.dtype
        
        # Ensure frames are in float format and normalized to [0, 1]
        if frames.max() > 1.0:
            frames = frames / 255.0
            
        # Extract RGB channels properly
        r = frames[..., 0, :, :].to(dtype=torch.float32)
        g = frames[..., 1, :, :].to(dtype=torch.float32)
        b = frames[..., 2, :, :].to(dtype=torch.float32)
        
        # Create skin mask using strict type handling
        with torch.no_grad():
            # RGB thresholds for skin detection
            skin_mask = ((r > 0.4).to(dtype=torch.bool) & 
                        (g > 0.28).to(dtype=torch.bool) & 
                        (b > 0.2).to(dtype=torch.bool) &
                        (r > g).to(dtype=torch.bool) & 
                        (r > b).to(dtype=torch.bool))
            
            # Additional refinements with explicit type casting
            diff_rg = (r - g).abs()
            skin_mask = skin_mask & (diff_rg > 0.1).to(dtype=torch.bool)
            
        return skin_mask.to(device=device)
        
    def forward(self, frames):
        """
        Extract skin color features with robust error handling.
        Args:
            frames: torch.Tensor of shape [batch_size, num_frames, 3, height, width]
        Returns:
            torch.Tensor: Skin color features
        """
        batch_size, num_frames = frames.shape[:2]
        device = frames.device
        
        try:
            # Detect skin regions
            skin_mask = self.detect_skin(frames)
            
            # Extract color features for skin regions
            skin_features = []
            
            for b in range(batch_size):
                frame_features = []
                for t in range(num_frames):
                    mask = skin_mask[b, t]
                    if mask.any():
                        # Safe indexing with boolean mask
                        r_mean = torch.mean(frames[b, t, 0][mask])
                        g_mean = torch.mean(frames[b, t, 1][mask])
                        b_mean = torch.mean(frames[b, t, 2][mask])
                        frame_feat = torch.stack([r_mean, g_mean, b_mean])
                    else:
                        frame_feat = torch.tensor([0.5, 0.4, 0.35], device=device)
                    frame_features.append(frame_feat)
                
                # Stack frame features
                skin_features.append(torch.stack(frame_features))
            
            # Stack batch features
            return torch.stack(skin_features)
            
        except Exception as e:
            print(f"Error in skin color analysis: {str(e)}")
            # Return fallback tensor
            return torch.zeros((batch_size, num_frames, 3), device=device)

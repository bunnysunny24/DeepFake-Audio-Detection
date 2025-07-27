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
            # Compute all conditions separately and ensure boolean type
            r_thresh = (r > 0.4).bool()
            g_thresh = (g > 0.28).bool()
            b_thresh = (b > 0.2).bool()
            r_g_cond = (r > g).bool()
            r_b_cond = (r > b).bool()
            
            # Combine conditions
            skin_mask = r_thresh & g_thresh & b_thresh & r_g_cond & r_b_cond
            
            # Additional refinements
            diff_rg = (r - g).abs()
            diff_mask = (diff_rg > 0.1).bool()
            skin_mask = skin_mask & diff_mask
            
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
                    mask = skin_mask[b, t].bool()  # Ensure mask is boolean
                    if mask.any():
                        # Safe indexing with boolean mask
                        r_vals = frames[b, t, 0][mask]
                        g_vals = frames[b, t, 1][mask]
                        b_vals = frames[b, t, 2][mask]
                        
                        # Compute means only if we have valid values
                        if r_vals.numel() > 0:
                            r_mean = torch.mean(r_vals)
                            g_mean = torch.mean(g_vals)
                            b_mean = torch.mean(b_vals)
                            frame_feat = torch.stack([r_mean, g_mean, b_mean])
                        else:
                            frame_feat = torch.tensor([0.5, 0.4, 0.35], device=device)
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

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
            skin_mask = self.detect_skin(frames)  # [B, T, H, W]
            # Prepare output tensor
            skin_features = torch.full((batch_size, num_frames, 3), 0.0, device=device)
            # Reshape for vectorized computation
            # frames: [B, T, 3, H, W], skin_mask: [B, T, H, W]
            mask_flat = skin_mask.view(batch_size, num_frames, -1)  # [B, T, H*W]
            frames_flat = frames.view(batch_size, num_frames, 3, -1)  # [B, T, 3, H*W]
            # For each batch and frame, compute mean for each channel where mask is True
            for c in range(3):
                channel_vals = frames_flat[:, :, c, :]  # [B, T, H*W]
                masked_vals = channel_vals * mask_flat.float()  # zeros out non-skin, ensure float mask
                # Count number of skin pixels per frame
                skin_counts = mask_flat.sum(dim=-1)  # [B, T]
                # Avoid division by zero
                skin_counts_safe = skin_counts.clone()
                skin_counts_safe[skin_counts_safe == 0] = 1
                # Sum over skin pixels
                skin_sum = masked_vals.sum(dim=-1)  # [B, T]
                skin_mean = skin_sum / skin_counts_safe  # [B, T]
                # Where no skin, set to default value
                default_val = torch.tensor([0.5, 0.4, 0.35], device=device)[c]
                skin_mean = torch.where(skin_counts > 0, skin_mean, default_val)
                skin_features[:, :, c] = skin_mean
            return skin_features
        except Exception as e:
            print(f"Error in skin color analysis: {str(e)}")
            # Return fallback tensor
            return torch.zeros((batch_size, num_frames, 3), device=device)

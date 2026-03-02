import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class SkinColorAnalyzer(nn.Module):
    def __init__(self, feature_dim=32):
        super(SkinColorAnalyzer, self).__init__()
        self.feature_dim = feature_dim
        # Add BatchNorm and a learnable projection for trainable features
        self.bn = nn.BatchNorm1d(3)
        self.proj = nn.Linear(3, self.feature_dim)
    def detect_skin(self, frames):
        """
        Detect skin regions in RGB frames using YCrCb heuristic.
        Args:
            frames: torch.Tensor of shape [batch_size, num_frames, 3, height, width]
        Returns:
            torch.Tensor: Binary skin mask
        """
        # Vectorized, torch-based skin heuristic to avoid CPU roundtrips
        # Heuristic based on RGB thresholds (scaled for [0,1] input)
        device = frames.device
        dtype = frames.dtype
        if frames.max() > 1.0:
            frames = frames / 255.0

        # frames: [B, T, 3, H, W]
        b, t, c, h, w = frames.shape
        # Split channels
        r = frames[:, :, 0, :, :]
        g = frames[:, :, 1, :, :]
        b_ch = frames[:, :, 2, :, :]

        # Precompute per-pixel max and min across channels
        maxc = torch.max(frames, dim=2).values  # [B, T, H, W]
        minc = torch.min(frames, dim=2).values  # [B, T, H, W]

        # Thresholds (original thresholds in 0-255 space): R>95, G>40, B>20, (max-min)>15, R>G, R>B, (R-G)>15
        # Scaled to [0,1]
        thr_r = 95.0 / 255.0
        thr_g = 40.0 / 255.0
        thr_b = 20.0 / 255.0
        thr_diff = 15.0 / 255.0

        cond = (
            (r > thr_r)
            & (g > thr_g)
            & (b_ch > thr_b)
            & ((maxc - minc) > thr_diff)
            & (r > g)
            & (r > b_ch)
            & ((r - g) > thr_diff)
        )

        skin_mask = cond.to(dtype=dtype, device=device).contiguous()  # [B, T, H, W]
        return skin_mask
    def forward(self, frames):
        """
        Extract skin color features with robust error handling.
        Args:
            frames: torch.Tensor of shape [batch_size, num_frames, 3, height, width]
        Returns:
            torch.Tensor: Skin color features
        """
        assert frames.dim() == 5 and frames.shape[2] == 3, f"Input must be [batch, num_frames, 3, height, width], got {frames.shape}"
        batch_size, num_frames = frames.shape[:2]
        device = frames.device
        try:
            # Detect skin regions
            skin_mask = self.detect_skin(frames)  # [B, T, H, W]

            # Vectorize feature computation to avoid Python loops and repeated small allocations
            mask_flat = skin_mask.view(batch_size, num_frames, -1).to(dtype=frames.dtype)  # [B, T, H*W]
            frames_flat = frames.view(batch_size, num_frames, 3, -1)  # [B, T, 3, H*W]

            # Broadcast mask over channels and compute per-channel sums: [B, T, 3]
            masked = frames_flat * mask_flat.unsqueeze(2)  # [B, T, 3, H*W]
            skin_sum = masked.sum(dim=-1)  # [B, T, 3]

            # Counts per frame: [B, T]
            skin_counts = mask_flat.sum(dim=-1)
            skin_counts_safe = skin_counts.clamp_min(1).unsqueeze(-1)  # [B, T, 1]

            skin_mean = skin_sum / skin_counts_safe  # [B, T, 3]

            # Pre-create default values once on correct device/dtype
            default_val = torch.tensor([0.5, 0.4, 0.35], device=device, dtype=frames.dtype).view(1, 1, 3)

            # Where there are zero skin pixels, substitute default color means
            mask_has_skin = skin_counts.unsqueeze(-1) > 0
            skin_mean = torch.where(mask_has_skin, skin_mean, default_val)

            # Apply BatchNorm and projection
            skin_features_bn = self.bn(skin_mean.contiguous().view(-1, 3)).view(batch_size, num_frames, 3)
            skin_features_proj = self.proj(skin_features_bn)
            return skin_features_proj
        except Exception as e:
            logger.exception("Error in skin color analysis")
            return torch.zeros((batch_size, num_frames, self.feature_dim), device=device, dtype=frames.dtype)
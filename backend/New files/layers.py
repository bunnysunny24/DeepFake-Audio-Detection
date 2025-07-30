
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianNoise(nn.Module):
    """Add Gaussian noise during training for regularization."""
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
        self.training = True
        
    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x

class MixupLayer(nn.Module):
    """Mixup augmentation layer."""
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.training = True
    
    def forward(self, x, target):
        if not self.training or self.alpha <= 0:
            return x, target
        
        batch_size = x.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        shuffle_idx = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[shuffle_idx]
        mixed_target = lam * target + (1 - lam) * target[shuffle_idx]
        
        return mixed_x, mixed_target

class CutMixLayer(nn.Module):
    """CutMix augmentation layer."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.training = True
    
    def forward(self, x, target):
        if not self.training or self.alpha <= 0:
            return x, target
        
        batch_size = x.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Generate random bbox
        W = x.size(2)
        H = x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        shuffle_idx = torch.randperm(batch_size, device=x.device)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[shuffle_idx, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        mixed_target = lam * target + (1 - lam) * target[shuffle_idx]
        
        return x, mixed_target

class StochasticDepth(nn.Module):
    """Stochastic Depth regularization."""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.training = True
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class LabelSmoothing(nn.Module):
    """Label smoothing loss."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class FeatureDropout(nn.Module):
    """Feature-level dropout for regularization."""
    def __init__(self, drop_prob=0.1, feature_size=None):
        super().__init__()
        self.drop_prob = drop_prob
        self.feature_size = feature_size
        self.training = True
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
            
        # Create feature dropout mask
        if self.feature_size is None:
            self.feature_size = x.size(1)
        mask = torch.bernoulli(torch.full((1, self.feature_size), 1 - self.drop_prob)).to(x.device)
        return x * mask.expand_as(x)

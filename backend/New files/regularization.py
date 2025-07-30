from dataclasses import dataclass
from typing import Optional

@dataclass
class RegularizationConfig:
    """Configuration for model regularization and anti-overfitting measures."""
    
    # Dropout settings
    dropout_rate: float = 0.5
    spatial_dropout_rate: float = 0.2
    feature_dropout_rate: float = 0.1
    
    # Noise injection
    gaussian_noise_std: float = 0.1
    
    # Weight regularization
    weight_decay: float = 0.01
    l1_regularization: float = 0.0
    
    # Gradient clipping
    gradient_clip_norm: float = 1.0
    
    # Label smoothing
    label_smoothing: float = 0.1
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.001
    
    # Data augmentation
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    use_mixup: bool = True
    use_cutmix: bool = True
    
    # Learning rate scheduling
    use_cosine_schedule: bool = True
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Stochastic Weight Averaging
    use_swa: bool = True
    swa_start: int = 10
    swa_freq: int = 5
    swa_lr: float = 1e-2
    
    # Advanced regularization
    stochastic_depth_prob: float = 0.1
    feature_size: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert 0 <= self.dropout_rate <= 1, "Dropout rate must be between 0 and 1"
        assert 0 <= self.spatial_dropout_rate <= 1, "Spatial dropout rate must be between 0 and 1"
        assert 0 <= self.feature_dropout_rate <= 1, "Feature dropout rate must be between 0 and 1"
        assert self.gaussian_noise_std >= 0, "Gaussian noise std must be non-negative"
        assert self.weight_decay >= 0, "Weight decay must be non-negative"
        assert self.gradient_clip_norm > 0, "Gradient clip norm must be positive"
        assert 0 <= self.label_smoothing < 1, "Label smoothing must be between 0 and 1"
        assert self.early_stopping_patience > 0, "Early stopping patience must be positive"
        assert self.early_stopping_delta > 0, "Early stopping delta must be positive"

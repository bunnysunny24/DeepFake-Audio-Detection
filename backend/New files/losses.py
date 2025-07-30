import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with class weights for handling imbalanced datasets.
    Combines class weighting with focal modulation to down-weight easy examples.
    Includes numerical stability improvements and proper handling of edge cases.
    """
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, reduction='mean', eps=1e-8):
        """
        Args:
            alpha (float): Weighting factor for rare class (usually the positive/fake class)
            gamma (float): Focusing parameter to down-weight easy examples 
            class_weights (tensor): Manual class weights if needed
            reduction (str): 'none' | 'mean' | 'sum'
            eps (float): Small constant to prevent numerical instability
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor of shape [B, C] with model predictions 
            targets: Tensor of shape [B] with class labels
        Returns:
            Weighted focal loss value
        """
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # (B, C, d1, d2, ...) -> (B, C, D)
            inputs = inputs.transpose(1, 2)    # (B, C, D) -> (B, D, C)
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # (B, D, C) -> (B * D, C)
        
        targets = targets.view(-1)
        
        # Get probabilities with numerical stability
        log_softmax = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_softmax)  # More stable than direct softmax
        
        # Get the probability for the target class
        pt = probs.gather(1, targets.unsqueeze(1))
        pt = pt.view(-1) + self.eps  # Add eps for numerical stability
        
        # Clamp probabilities to prevent extreme focal weights
        pt = torch.clamp(pt, min=self.eps, max=1.0 - self.eps)
        
        # Compute focal weights with controlled scaling
        focal_weights = ((1 - pt) ** self.gamma).clamp(min=0.0, max=100.0)
        
        # Compute alpha weights
        alpha = torch.ones_like(pt) * self.alpha
        alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
        
        # Combine alpha and focal weights
        weights = alpha_t * focal_weights
        
        # Add class weights if provided
        if self.class_weights is not None:
            class_weights = self.class_weights.to(inputs.device)
            weights = weights * class_weights.gather(0, targets)
        
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply weights to cross entropy
        loss = weights * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss that combines classification and regularization losses
    """
    def __init__(self, class_weights=None, type_class_weights=None, lambda_reg=0.1, lambda_aux=0.2):
        super().__init__()
        self.class_weights = class_weights
        self.type_class_weights = type_class_weights  # Separate weights for deepfake type classification
        # Main deepfake detection loss with class weighting and focal modulation
        self.main_loss = WeightedFocalLoss(class_weights=class_weights)
        # Weight for regularization losses
        self.lambda_reg = lambda_reg
        # Weight for auxiliary tasks
        self.lambda_aux = lambda_aux
        
    def forward(self, outputs, targets, aux_outputs=None, aux_targets=None):
        """
        Compute combined loss with regularization
        
        Args:
            outputs: Main model outputs
            targets: Target labels
            aux_outputs: Dict of auxiliary outputs (optional)
            aux_targets: Dict of auxiliary targets (optional)
            
        Returns:
            total_loss: Combined loss value
            losses: Dict containing individual loss components
        """
        losses = {}
        
        # Main classification loss using Weighted Focal Loss
        main_loss = self.main_loss(outputs, targets)
        losses['main'] = main_loss.item() if isinstance(main_loss, torch.Tensor) else main_loss
        total_loss = main_loss

        # Auxiliary losses if provided
        if aux_outputs is not None and aux_targets is not None:
            if isinstance(aux_outputs, dict) and isinstance(aux_targets, dict):
                if 'deepfake_type' in aux_outputs and 'deepfake_type' in aux_targets:
                    # Use weighted cross entropy for deepfake type classification
                    # Ensure weights are on the same device as the inputs
                    weights = self.type_class_weights
                    if weights is not None:
                        weights = weights.to(aux_outputs['deepfake_type'].device)
                    
                    type_loss = F.cross_entropy(
                        aux_outputs['deepfake_type'], 
                        aux_targets['deepfake_type'],
                        weight=weights
                    )
                    losses['type'] = type_loss.item()
                    total_loss += self.lambda_aux * type_loss
        
        return total_loss, losses

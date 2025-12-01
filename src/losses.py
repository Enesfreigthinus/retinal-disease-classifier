"""
Losses module for Retinal Disease Classifier.
Contains loss functions including Focal Loss for class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    
    Addresses class imbalance by down-weighting easy examples
    and focusing on hard examples.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter (0 = BCE, 2 is typical)
        alpha: Class weighting factor
        pos_weight: Weight for positive samples (for class imbalance)
        reduction: Reduction method ('mean', 'sum', 'none')
        
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(
        self, 
        gamma: float = 2.0, 
        alpha: float = None, 
        pos_weight: torch.Tensor = None, 
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Loss value
        """
        # Binary cross entropy with logits
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, 
                pos_weight=self.pos_weight, 
                reduction='none'
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, 
                reduction='none'
            )
        
        # Focal term: (1 - p_t)^gamma
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    Different focusing parameters for positive and negative samples.
    Particularly effective when negative samples dominate.
    
    Args:
        gamma_neg: Focusing parameter for negative samples
        gamma_pos: Focusing parameter for positive samples
        clip: Probability clipping value
        reduction: Reduction method
        
    Reference:
        Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021
    """
    
    def __init__(
        self, 
        gamma_neg: float = 4.0, 
        gamma_pos: float = 1.0, 
        clip: float = 0.05, 
        reduction: str = 'mean'
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Loss value
        """
        # Get probabilities
        xs_pos = torch.sigmoid(inputs)
        xs_neg = 1 - xs_pos
        
        # Asymmetric clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # Basic cross entropy
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg
        
        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w
        
        if self.reduction == 'mean':
            return -loss.mean()
        elif self.reduction == 'sum':
            return -loss.sum()
        else:
            return -loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss.
    
    Simple class-weighted BCE for handling imbalanced datasets.
    
    Args:
        pos_weight: Weight for positive samples
        reduction: Reduction method
    """
    
    def __init__(self, pos_weight: torch.Tensor = None, reduction: str = 'mean'):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Loss value
        """
        return F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


def get_loss_function(
    loss_type: str = 'focal',
    pos_weight: torch.Tensor = None,
    gamma: float = None,
    device: str = None
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('focal', 'bce', 'weighted_bce', 'asymmetric')
        pos_weight: Positive class weights for imbalanced data
        gamma: Focal loss gamma parameter
        device: Device to place loss function on
        
    Returns:
        Loss function module
    """
    if gamma is None:
        gamma = config.FOCAL_LOSS_GAMMA
    if device is None:
        device = config.DEVICE
    
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    
    if loss_type == 'focal':
        loss_fn = FocalLoss(gamma=gamma, pos_weight=pos_weight)
    elif loss_type == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()
    elif loss_type == 'weighted_bce':
        loss_fn = WeightedBCELoss(pos_weight=pos_weight)
    elif loss_type == 'asymmetric':
        loss_fn = AsymmetricLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_fn.to(device)


def print_loss_summary(loss_type: str, pos_weight: torch.Tensor = None):
    """Print loss function configuration."""
    print("=" * 60)
    print("LOSS FUNCTION")
    print("=" * 60)
    print(f"Type: {loss_type}")
    
    if loss_type == 'focal':
        print(f"Gamma: {config.FOCAL_LOSS_GAMMA}")
    
    if pos_weight is not None:
        print(f"Using positive class weights")
        print(f"  Min weight: {pos_weight.min():.2f}")
        print(f"  Max weight: {pos_weight.max():.2f}")
        print(f"  Mean weight: {pos_weight.mean():.2f}")
    
    print("=" * 60)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==============================================================================
# LOSS FUNCTIONS FOR CHANGE DETECTION
# ==============================================================================

class BinaryCrossEntropyLoss(nn.Module):
    """
    Binary Cross Entropy Loss for binary change detection.
    Uses BCEWithLogitsLoss for numerical stability (combines sigmoid + BCE).
    """
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class DiceLoss(nn.Module):
    """
    Dice Loss (Soft Dice Coefficient) for segmentation.
    Directly optimizes the Dice coefficient, good for class imbalance.
    """
    def __init__(self, smooth: float = 1e-6, squared: bool = False):
        super().__init__()
        self.smooth = smooth
        self.squared = squared

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        if self.squared:
            intersection = (probs_flat * targets_flat).sum()
            union = (probs_flat * probs_flat).sum() + (targets_flat * targets_flat).sum()
        else:
            intersection = (probs_flat * targets_flat).sum()
            union = probs_flat.sum() + targets_flat.sum()
        
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_coeff


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance.
    Down-weights easy examples and focuses on hard negatives/positives.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        
        loss = focal_weight * bce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BCEDiceLoss(nn.Module):
    """
    Hybrid loss combining Binary Cross Entropy and Dice Loss to tackle extreme pixel class imbalance
    commonly found in satellite image change detection.
    """
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
        return total_loss


class BCEFocalLoss(nn.Module):
    """
    Hybrid loss combining Binary Cross Entropy and Focal Loss.
    """
    def __init__(self, bce_weight: float = 0.5, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        focal_loss = self.focal(logits, targets)
        total_loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * focal_loss
        return total_loss


class DiceFocalLoss(nn.Module):
    """
    Hybrid loss combining Dice Loss and Focal Loss.
    """
    def __init__(self, dice_weight: float = 0.5, alpha: float = 0.25, gamma: float = 2.0, smooth: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)
        total_loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * focal_loss
        return total_loss


# Alias for backward compatibility
from typing import Optional


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Available losses:
    - 'bce': Binary Cross Entropy
    - 'dice': Dice Loss
    - 'focal': Focal Loss
    - 'bce_dice': BCE + Dice (default)
    - 'bce_focal': BCE + Focal
    - 'dice_focal': Dice + Focal
    """
    loss_name = loss_name.lower()
    
    if loss_name == "bce":
        return BinaryCrossEntropyLoss(**kwargs)
    elif loss_name == "dice":
        return DiceLoss(**kwargs)
    elif loss_name == "focal":
        return FocalLoss(**kwargs)
    elif loss_name == "bce_dice":
        return BCEDiceLoss(**kwargs)
    elif loss_name == "bce_focal":
        return BCEFocalLoss(**kwargs)
    elif loss_name == "dice_focal":
        return DiceFocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def calculate_iou(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Calculates Intersection over Union (IoU / Jaccard Index) metric for tensor batches.
    """
    if preds.shape != targets.shape:
        preds = torch.sigmoid(preds)
    else:
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.sigmoid(preds)
            
    pred_mask = (preds > threshold).float()
    
    intersection = (pred_mask * targets).sum()
    total = pred_mask.sum() + targets.sum()
    union = total - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def calculate_metrics(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Calculates Precision, Recall, F1-Score, and IoU.
    """
    if isinstance(preds, torch.Tensor):
        if preds.min() < 0 or preds.max() > 1:
            probs = torch.sigmoid(preds)
        else:
            probs = preds
        p_mask = (probs > threshold).float()
        
        tp = (p_mask * targets).sum().item()
        fp = (p_mask * (1.0 - targets)).sum().item()
        fn = ((1.0 - p_mask) * targets).sum().item()
    else:
        p_mask = (preds > threshold).astype(np.float32)
        t_mask = (targets > threshold).astype(np.float32)
        tp = np.sum(p_mask * t_mask)
        fp = np.sum(p_mask * (1.0 - t_mask))
        fn = np.sum((1.0 - p_mask) * t_mask)

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    iou = tp / (tp + fp + fn + smooth)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou
    }

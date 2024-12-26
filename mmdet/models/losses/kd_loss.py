import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from .utils import weighted_loss


def norm(feat: torch.Tensor) -> torch.Tensor:
    """Normalize the feature maps to have zero mean and unit variance."""
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    feat = (feat - mean) / (std + 1e-6)
    return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)


def calculate_mse_loss(pred, target):
    pred = norm(pred)
    target = norm(target)
    loss = F.mse_loss(pred, target, reduction='none') / 2
    return loss


def calculate_l1_loss(pred, target):
    pred = norm(pred)
    target = norm(target)
    loss = F.l1_loss(pred, target, reduction='none')
    return loss


def calculate_kl_loss(pred, target, T=3.0):

    assert pred.size() == target.size(), "Pred and target must have the same shape."

    # Apply log_softmax to the pred and softmax to the target along the channel dimension (C)
    pred_log_softmax = F.log_softmax(pred / T, dim=1) 
    target_softmax = F.softmax(target / T, dim=1)
    kd_loss = F.kl_div(pred_log_softmax, target_softmax, reduction='none', log_target=False)
    kd_loss = kd_loss.sum(dim=1)  # Now kd_loss is of shape (N, H, W)
    # Average the loss across all spatial positions (H, W)
    kd_loss = kd_loss.mean(dim=[1, 2])  # Now kd_loss is of shape (N,)

    # Scale the loss by T^2
    loss = kd_loss * (T * T)
    
    return loss # Return the mean loss over the batch


@MODELS.register_module()
class KDLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, loss_type='mse'):
        super(KDLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        # Validate loss_type
        assert self.loss_type in ['l1', 'mse', 'kl']

    def forward(self, pred, target, reduction_override=None) -> torch.Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        # Select the appropriate loss calculation function based on loss_type
        if self.loss_type == 'mse':
            loss = calculate_mse_loss(pred, target)
        elif self.loss_type == 'l1':
            loss = calculate_l1_loss(pred, target)
        elif self.loss_type == 'kl':
            loss = calculate_kl_loss(pred, target)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        loss = self.loss_weight * loss
        return loss
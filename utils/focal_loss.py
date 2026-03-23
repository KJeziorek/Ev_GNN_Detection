import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Sigmoid focal loss — replaces BCEWithLogitsLoss for obj branch."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        # pred: raw logits, target: 0 or 1
        p = pred.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # p_t = p where target=1, (1-p) where target=0
        p_t = p * target + (1 - p) * (1 - target)
        modulator = (1 - p_t) ** self.gamma

        # alpha weighting: alpha for positive, (1-alpha) for negative
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = alpha_t * modulator * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
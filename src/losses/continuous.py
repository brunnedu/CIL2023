'''
    Losses taking floating point tensors as inputs
'''

import torch.nn as nn
import torch.functional as F

class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(y_pred, y_true):
        tp = (y_true * y_pred).sum()
        p = (y_true + y_pred - y_true * y_pred).sum()
        return  tp / p

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.flatten(1)
        y_true = y_true.flatten(1)
        return 1.0 - ((2.0 * (y_pred * y_true).sum(1) + self.smooth) / ((y_pred + y_true).sum(1) + self.smooth)).mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, y_pred, y_true):
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction=self.reduction)
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            loss = alpha_t * loss

        return loss.mean()
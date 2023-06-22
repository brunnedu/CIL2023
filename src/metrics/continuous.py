"""
    Metrics taking floating point tensors as inputs.
    Expects all values to be between 0. and 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.classification


class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(y_pred, y_true):
        tp = (y_true * y_pred).sum()
        p = (y_true + y_pred - y_true * y_pred).sum()
        return tp / p


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.flatten(1)
        y_true = y_true.flatten(1)

        tp = (y_pred * y_true).sum(1)

        return 1.0 - ((2.0 * tp + self.smooth) / ((y_pred + y_true).sum(1) + self.smooth)).mean()


class BinaryF1Score(nn.Module):
    def __init__(self, smooth=1.0, reduction='average', alpha=0.0):
        super().__init__()
        '''
            Higher alpha pushes the values closer to 0 and 1
        '''

        reduction_fns = {
            'average': torch.mean,
            'sum': torch.sum,
        }

        self.smooth = smooth
        self.alpha = alpha
        self.reduction_fn = reduction_fns[reduction]

    def forward(self, y_pred, y_true):
        y_pred = y_pred.flatten(1)
        y_true = y_true.flatten(1)

        if self.alpha > 0.0:
            y_pred = torch.sigmoid((y_pred - 0.5) * self.alpha)

        tp = (y_pred * y_true).sum(1)
        fp = (y_pred * (1.0 - y_true)).sum(1)
        fn = ((1.0 - y_pred) * y_true).sum(1)

        f1scores = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        return self.reduction_fn(f1scores)


class OneMinusLossScore(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        """ Wrapper score that is simply 1.0 - loss """
        super().__init__()

        self.loss_fn = loss_fn

    def forward(self, y_pred, y_true):
        return 1.0 - self.loss_fn(y_pred, y_true)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, bce_reduction="none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = bce_reduction

    def forward(self, y_pred, y_true):
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction=self.reduction)
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            loss = alpha_t * loss

        return loss.mean()


class PatchAccuracy(torch.nn.Module):
    """
    Evaluation metric used by kaggle.
    1. Splits the prediction and target into patches of size patch_size.
    2. Binarizes every patch by comparing the mean of the patch activations to the cutoff value.
    3. Computes the accuracy over the binarized patches.
    """
    def __init__(self, patch_size: int = 16, cutoff: float = 0.25):
        super(PatchAccuracy, self).__init__()
        self.patch_size = patch_size
        self.cutoff = cutoff

    def forward(self, y_hat, y):

        patches_hat, patches = self.binarize_patches(y_hat, y)

        return (patches == patches_hat).float().mean()

    def binarize_patches(self, y_hat, y):

        h_patches = y.shape[-2] // self.patch_size
        w_patches = y.shape[-1] // self.patch_size
        patches_hat = y_hat.reshape(-1, 1, h_patches, self.patch_size, w_patches, self.patch_size).mean(
            (-1, -3)) > self.cutoff
        patches = y.reshape(-1, 1, h_patches, self.patch_size, w_patches, self.patch_size).mean((-1, -3)) > self.cutoff

        return patches_hat, patches


class PatchF1Score(PatchAccuracy):
    """
    Evaluation metric used this year.
    1. Splits the prediction and target into patches of size patch_size.
    2. Binarizes every patch by comparing the mean of the patch activations to the cutoff value.
    3. Computes the F1-Score over the binarized patches.
    """
    def __init__(self, patch_size: int = 16, cutoff: float = 0.25, eps: float = 1e-10):
        super(PatchF1Score, self).__init__(patch_size=patch_size, cutoff=cutoff)

        # TODO: fix issue with self.metric not being put on device
        self.eps = eps

    def forward(self, y_hat, y):

        patches_hat, patches = self.binarize_patches(y_hat, y)

        # Compute true positives, false positives, and false negatives
        tp = torch.sum((patches_hat == 1) & (patches == 1)).float()
        fp = torch.sum((patches_hat == 1) & (patches == 0)).float()
        fn = torch.sum((patches_hat == 0) & (patches == 1)).float()

        # Compute precision, recall, and F1 score
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1_score = 2 * precision * recall / (precision + recall + self.eps)

        return f1_score


class LabelUncertaintyLoss(torch.nn.Module):
    """
    BCE Loss with label uncertainty.
    Adapted from `Weakly-Supervised Semantic Segmentation by Learning Label Uncertainty <https://arxiv.org/pdf/2110.05926.pdf>` section 3.2.1
    """
    def __init__(self, bce_reduction: str = "mean"):
        super().__init__()
        self.reduction = bce_reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):

        # Extract mean and variance channels from y_pred
        mean = y_pred[:, 0, :, :]
        variance = y_pred[:, 1, :, :]

        # Compute the adapted logit value
        adapted_logit = (mean / (1 + torch.pi * variance ** 2 / 8).sqrt()).sigmoid().unsqueeze(1)

        # Compute binary cross entropy loss
        loss = F.binary_cross_entropy(adapted_logit, y_true, reduction=self.reduction)

        return loss

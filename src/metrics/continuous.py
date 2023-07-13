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

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.flatten(1)
        y_true = y_true.flatten(1)

        tp = (y_pred * y_true).sum(1)

        return 1.0 - ((2.0 * tp + self.smooth) / ((y_pred + y_true).sum(1) + self.smooth)).mean()

    def __repr__(self):
        return f"{self.__class__.__name__}(smooth={self.smooth})"


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

    def __repr__(self):
        return f"{self.__class__.__name__}(smooth={self.smooth}, alpha={self.alpha})"


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

    def __repr__(self):
        return f"FocalLoss(alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction})"


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

    def __repr__(self):
        return f"PatchAccuracy(patch_size={self.patch_size}, cutoff={self.cutoff})"


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

    def __repr__(self):
        return f"PatchF1Score(patch_size={self.patch_size}, cutoff={self.cutoff}, eps={self.eps})"



################################
### TOPOLOGY PRESERVING LOSS ###
################################

def soft_erode(img):
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)

def soft_dilate(img):
    return F.max_pool2d(img, (3,3), (1,1), (1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skeletonize(img, nr_of_iterations: int):
    img1 = soft_open(img)
    skel = F.relu(img-img1)
    for j in range(nr_of_iterations):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

class SoftClDice(nn.Module):
    def __init__(self, nr_of_iterations: int = 50, smooth=1.):
        super().__init__()
        self.nr_of_iterations = nr_of_iterations
        self.epsilon = smooth

    def forward(self, y_true, y_pred):
        skel_pred = soft_skeletonize(y_pred, self.nr_of_iterations)
        skel_true = soft_skeletonize(y_true, self.nr_of_iterations)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...]) + self.smooth) / (torch.sum(skel_pred[:,1:,...]) + self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...]) + self.smooth) / (torch.sum(skel_true[:,1:,...]) + self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec * tsens)/(tprec + tsens)
        return cl_dice


class TopologyPreservingLoss(nn.Module):
    """ 
        Loss tries to preserve the connectivity of components
        Implemented as described here: https://arxiv.org/pdf/2003.07311.pdf
        Adapted from https://github.com/jocpae/clDice/tree/master

        nr_of_iterations: int = 50, the maximum foreground width / how often min-maxing should be applied
        weight_cldice: float = 0.5, the weight that will be given to connectivity preserving dice loss (rest will be given to standard dice loss)
        smooth: float = 0.01, can be used to regularize the cl_loss
    """
    def __init__(self, nr_of_iterations=50, weight_cldice=0.5, smooth: float = 1):
        super().__init__()
        self.nr_of_iterations = nr_of_iterations
        self.smooth = smooth
        self.weight_cldice = weight_cldice

    def soft_dice(self, y_true, y_pred):
        intersection = torch.sum((y_true * y_pred)[:,1:,...])
        coeff = (2 * intersection + self.smooth) / (torch.sum(y_true[:,1:,...]) + torch.sum(y_pred[:,1:,...]) + self.smooth)
        return 1.0 - coeff

    def forward(self, y_true, y_pred):
        if len(y_true.shape) == 4:
            y_true = y_true.squeeze(1) # remove channel dim
            y_pred = y_pred.squeeze(1)

        dice = self.soft_dice(y_true, y_pred)
        skel_pred = soft_skeletonize(y_pred, self.nr_of_iterations)
        skel_true = soft_skeletonize(y_true, self.nr_of_iterations)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...]) + self.smooth) / (torch.sum(skel_pred[:,1:,...]) + self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...]) + self.smooth) / (torch.sum(skel_true[:,1:,...]) + self.smooth)    
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return (1.0 - self.weight_cldice) * dice + self.weight_cldice * cl_dice
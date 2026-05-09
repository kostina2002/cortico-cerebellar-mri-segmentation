import torch
import torch.nn.functional as F

def focal_loss(pred, target, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    alpha = 0.75
    return (alpha * (1 - pt) ** gamma * bce).mean()

def tversky_loss(pred, target, alpha=0.7, beta=0.3):
    pred, target = pred.reshape(-1), target.reshape(-1)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    return 1 - (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)

def dice_loss(pred, target):
    pred, target = pred.reshape(-1), target.reshape(-1)
    inter = (pred * target).sum()
    return 1 - (2. * inter + 1e-6) / (pred.sum() + target.sum() + 1e-6)

def combined_loss(pred, target):
    return (0.3 * focal_loss(pred, target) +
            0.4 * tversky_loss(pred, target) +
            0.3 * dice_loss(pred, target))
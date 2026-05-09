import torch

def dice_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()
    inter = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    return (2. * inter + 1e-6) / (union + 1e-6)
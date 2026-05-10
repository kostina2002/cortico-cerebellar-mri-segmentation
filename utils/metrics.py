import torch
import numpy as np

def dice_score_per_volume(pred, target, threshold=0.5, eps=1e-6):
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()
    
    B = pred.size(0)
    dices = []
    
    for b in range(B):
        p = pred_bin[b].reshape(-1)
        t = target_bin[b].reshape(-1)
    
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dice_b = (2. * inter + eps) / (union + eps)
        dices.append(dice_b.item())
    return float(np.mean(dices))

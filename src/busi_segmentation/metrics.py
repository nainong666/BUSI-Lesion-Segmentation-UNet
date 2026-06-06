from __future__ import annotations

import torch


def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = (torch.sigmoid(logits) >= threshold).float()
    targets = (targets >= 0.5).float()
    predictions = predictions.flatten(1)
    targets = targets.flatten(1)
    intersection = (predictions * targets).sum(dim=1)
    pred_sum = predictions.sum(dim=1)
    target_sum = targets.sum(dim=1)
    union = pred_sum + target_sum - intersection
    dice = torch.where(
        pred_sum + target_sum > 0,
        (2.0 * intersection) / (pred_sum + target_sum).clamp_min(1e-8),
        torch.ones_like(intersection),
    )
    iou = torch.where(union > 0, intersection / union.clamp_min(1e-8), torch.ones_like(union))
    return dice, iou

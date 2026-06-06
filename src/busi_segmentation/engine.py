from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .metrics import dice_iou_from_logits


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite.")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * images.size(0)
        sample_count += images.size(0)
    return loss_sum / sample_count


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module | None,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, object]], float | None]:
    model.eval()
    dice_values = []
    iou_values = []
    rows = []
    loss_sum = 0.0
    sample_count = 0

    for images, masks, paths in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            loss_sum += criterion(logits, masks).item() * images.size(0)
        batch_dice, batch_iou = dice_iou_from_logits(logits, masks)
        dice_values.extend(batch_dice.cpu().tolist())
        iou_values.extend(batch_iou.cpu().tolist())
        rows.extend(
            {"image_path": path, "dice": float(dice), "iou": float(iou)}
            for path, dice, iou in zip(paths, batch_dice.cpu().tolist(), batch_iou.cpu().tolist(), strict=True)
        )
        sample_count += images.size(0)

    metrics = {
        "dice": float(torch.tensor(dice_values).mean()),
        "iou": float(torch.tensor(iou_values).mean()),
    }
    loss = loss_sum / sample_count if criterion is not None else None
    return metrics, rows, loss

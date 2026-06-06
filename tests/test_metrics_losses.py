from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from busi_segmentation.losses import DiceLoss
from busi_segmentation.metrics import dice_iou_from_logits


class MetricsLossesTest(unittest.TestCase):
    def test_dice_and_iou_perfect_prediction(self) -> None:
        targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        logits = torch.where(targets > 0, torch.tensor(20.0), torch.tensor(-20.0))
        dice, iou = dice_iou_from_logits(logits, targets)
        self.assertAlmostEqual(float(dice.item()), 1.0, places=6)
        self.assertAlmostEqual(float(iou.item()), 1.0, places=6)

    def test_dice_and_iou_empty_masks_are_perfect_when_both_empty(self) -> None:
        targets = torch.zeros((1, 1, 2, 2))
        logits = torch.full_like(targets, -20.0)
        dice, iou = dice_iou_from_logits(logits, targets)
        self.assertAlmostEqual(float(dice.item()), 1.0, places=6)
        self.assertAlmostEqual(float(iou.item()), 1.0, places=6)

    def test_dice_loss_is_low_for_good_logits(self) -> None:
        targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        logits = torch.where(targets > 0, torch.tensor(20.0), torch.tensor(-20.0))
        loss = DiceLoss()(logits, targets)
        self.assertLess(float(loss.item()), 1e-4)


if __name__ == "__main__":
    unittest.main()

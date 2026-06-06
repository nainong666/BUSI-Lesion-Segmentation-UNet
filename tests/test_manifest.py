from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from busi_segmentation.data import BUSISegmentationDataset
from busi_segmentation.manifest import discover_segmentation_images, write_folds


def create_image(path: Path, value: int = 127) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=value).save(path)


def create_mask(path: Path, box: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((8, 8), dtype=np.uint8)
    y0, x0, y1, x1 = box
    mask[y0:y1, x0:x1] = 255
    Image.fromarray(mask, mode="L").save(path)


class ManifestTest(unittest.TestCase):
    def test_discovers_benign_and_malignant_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for cls in ("benign", "malignant", "normal"):
                create_image(root / cls / f"{cls} (0).png")
                create_mask(root / cls / f"{cls} (0)_mask.png", (1, 1, 4, 4))

            records = discover_segmentation_images(root)
            self.assertEqual(len(records), 2)
            self.assertEqual({record.label_name for record in records}, {"benign", "malignant"})

    def test_multiple_masks_are_listed_and_merged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            create_image(root / "benign" / "benign (0).png")
            create_mask(root / "benign" / "benign (0)_mask.png", (1, 1, 3, 3))
            create_mask(root / "benign" / "benign (0)_mask_1.png", (5, 5, 7, 7))
            create_image(root / "malignant" / "malignant (0).png")
            create_mask(root / "malignant" / "malignant (0)_mask.png", (2, 2, 4, 4))

            output_dir = root / "folds"
            metadata = write_folds(root, output_dir, num_folds=2, seed=3)
            self.assertEqual(metadata["multi_mask_images"], 1)
            dataset = BUSISegmentationDataset(root, output_dir / "all.csv", image_size=8)
            _, mask, _ = dataset[0]
            self.assertEqual(float(mask.sum()), 8.0)

    def test_folds_are_deterministic_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for cls, count in (("benign", 6), ("malignant", 4)):
                for index in range(count):
                    create_image(root / cls / f"{cls} ({index}).png")
                    create_mask(root / cls / f"{cls} ({index})_mask.png", (1, 1, 4, 4))

            first = write_folds(root, root / "folds-a", num_folds=2, seed=7)
            second = write_folds(root, root / "folds-b", num_folds=2, seed=7)
            self.assertEqual(first["splits"], second["splits"])
            train_paths = {
                line.split(",")[0]
                for line in (root / "folds-a" / "fold_0" / "train.csv").read_text(encoding="utf-8").splitlines()[1:]
            }
            val_paths = {
                line.split(",")[0]
                for line in (root / "folds-a" / "fold_0" / "val.csv").read_text(encoding="utf-8").splitlines()[1:]
            }
            self.assertFalse(train_paths & val_paths)


if __name__ == "__main__":
    unittest.main()

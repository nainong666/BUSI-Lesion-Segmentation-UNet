from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


class BUSISegmentationDataset(Dataset):
    def __init__(self, data_dir: Path, manifest_path: Path, image_size: int) -> None:
        self.data_dir = data_dir.resolve()
        self.image_size = image_size
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            self.records = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.records)

    def _load_mask_union(self, mask_paths: list[Path]) -> Image.Image:
        union: np.ndarray | None = None
        for mask_path in mask_paths:
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert("L")) > 0
            union = mask if union is None else np.logical_or(union, mask)
        if union is None:
            raise RuntimeError("mask_paths must not be empty")
        return Image.fromarray((union.astype(np.uint8) * 255), mode="L")

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.data_dir / record["image_path"]
        mask_paths = [self.data_dir / path for path in record["mask_paths"].split("|")]

        with Image.open(image_path) as source:
            image = source.convert("L")
        mask = self._load_mask_union(mask_paths)

        image = TF.resize(image, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
        image_tensor = TF.to_tensor(image)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        return image_tensor, mask_tensor, record["image_path"]


def make_datasets(data_dir: Path, fold_dir: Path, image_size: int):
    datasets = {}
    for split_name in ("train", "val"):
        manifest_path = fold_dir / f"{split_name}.csv"
        if manifest_path.exists():
            datasets[split_name] = BUSISegmentationDataset(data_dir, manifest_path, image_size)
    return datasets


def make_loaders(
    data_dir: Path,
    fold_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    datasets = make_datasets(data_dir, fold_dir, image_size)
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for split, dataset in datasets.items()
    }

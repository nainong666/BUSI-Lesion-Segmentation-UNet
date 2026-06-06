from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from . import SEGMENTATION_CLASSES

CLASS_TO_LABEL = {name: index for index, name in enumerate(SEGMENTATION_CLASSES)}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_PATTERN = re.compile(r"_mask(?:_\d+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class SegmentationRecord:
    image_path: str
    mask_paths: str
    label: int
    label_name: str


def is_mask_image(path: Path) -> bool:
    return bool(MASK_PATTERN.search(path.stem))


def mask_paths_for_image(image_path: Path) -> list[Path]:
    return sorted(image_path.parent.glob(f"{image_path.stem}_mask*{image_path.suffix}"))


def discover_segmentation_images(data_dir: Path) -> list[SegmentationRecord]:
    data_dir = data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"BUSI data directory does not exist: {data_dir}")

    records: list[SegmentationRecord] = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class directory: {class_dir}")
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if is_mask_image(image_path):
                continue
            masks = [path for path in mask_paths_for_image(image_path) if path.is_file()]
            if not masks:
                raise FileNotFoundError(f"No mask files found for image: {image_path}")
            records.append(
                SegmentationRecord(
                    image_path=image_path.resolve().relative_to(data_dir).as_posix(),
                    mask_paths="|".join(path.resolve().relative_to(data_dir).as_posix() for path in masks),
                    label=label,
                    label_name=class_name,
                )
            )

    if not records:
        raise RuntimeError(f"No BUSI segmentation images found under {data_dir}")
    return records


def stratified_folds(
    records: list[SegmentationRecord],
    num_folds: int,
    seed: int,
) -> list[dict[str, list[SegmentationRecord]]]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")

    rng = random.Random(seed)
    folds = [{"train": [], "val": []} for _ in range(num_folds)]
    for class_name in SEGMENTATION_CLASSES:
        class_records = [record for record in records if record.label_name == class_name]
        rng.shuffle(class_records)
        for fold_index in range(num_folds):
            val_records = class_records[fold_index::num_folds]
            val_ids = {record.image_path for record in val_records}
            train_records = [record for record in class_records if record.image_path not in val_ids]
            folds[fold_index]["train"].extend(train_records)
            folds[fold_index]["val"].extend(val_records)

    for fold in folds:
        rng.shuffle(fold["train"])
        rng.shuffle(fold["val"])
    return folds


def summarize(records: list[SegmentationRecord]) -> dict[str, int]:
    counts = Counter(record.label_name for record in records)
    return {"total": len(records), **{name: counts[name] for name in SEGMENTATION_CLASSES}}


def write_csv(path: Path, records: list[SegmentationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "mask_paths", "label", "label_name"])
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def write_folds(
    data_dir: Path,
    output_dir: Path,
    num_folds: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    records = discover_segmentation_images(data_dir)
    folds = stratified_folds(records, num_folds=num_folds, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all.csv", records)
    for fold_index, fold in enumerate(folds):
        fold_dir = output_dir / f"fold_{fold_index}"
        for split_name, split_records in fold.items():
            write_csv(fold_dir / f"{split_name}.csv", split_records)

    multi_mask_images = sum(1 for record in records if len(record.mask_paths.split("|")) > 1)
    metadata: dict[str, object] = {
        "data_dir": str(data_dir.resolve()),
        "seed": seed,
        "num_folds": num_folds,
        "classes": list(SEGMENTATION_CLASSES),
        "label_mapping": CLASS_TO_LABEL,
        "multi_mask_images": multi_mask_images,
        "splits": {
            "all": summarize(records),
            **{
                f"fold_{fold_index}": {
                    split_name: summarize(split_records) for split_name, split_records in fold.items()
                }
                for fold_index, fold in enumerate(folds)
            },
        },
        "note": "Multiple BUSI mask files for one image are merged as a binary lesion union.",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata

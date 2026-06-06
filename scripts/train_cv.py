from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from busi_segmentation.data import make_datasets, make_loaders
from busi_segmentation.engine import evaluate, train_one_epoch
from busi_segmentation.losses import build_loss
from busi_segmentation.models import build_model
from busi_segmentation.utils import load_json, resolve_device, save_json, seed_everything
from busi_segmentation.visualization import save_prediction_examples, save_training_curves

EXPERIMENTS = {
    "unet_256_bce_dice": {"image_size": 256, "loss": "bce_dice"},
    "unet_256_dice": {"image_size": 256, "loss": "dice"},
    "unet_256_bce": {"image_size": 256, "loss": "bce"},
    "unet_512_bce_dice": {"image_size": 512, "loss": "bce_dice"},
}
METRIC_KEYS = ["dice", "iou"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BUSI U-Net segmentation cross-validation.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "Dataset_BUSI_with_GT")
    parser.add_argument("--folds-dir", type=Path, default=REPO_ROOT / "data" / "folds")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs_cv")
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--examples", type=int, default=8)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def experiment_config(args: argparse.Namespace, fold_index: int) -> dict[str, object]:
    definition = EXPERIMENTS[args.experiment]
    return {
        "experiment": args.experiment,
        "fold": fold_index,
        "image_size": definition["image_size"],
        "loss": definition["loss"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "base_channels": args.base_channels,
        "seed": args.seed,
        "amp": args.amp,
    }


def load_completed_fold(output_dir: Path, expected_config: dict[str, object]) -> dict[str, object] | None:
    metrics_path = output_dir / "val" / "metrics.json"
    config_path = output_dir / "run_config.json"
    if not metrics_path.exists() or not config_path.exists():
        return None
    if load_json(config_path) != expected_config:
        return None
    return load_json(metrics_path)


def train_fold(args: argparse.Namespace, fold_index: int, device: torch.device) -> dict[str, object]:
    definition = EXPERIMENTS[args.experiment]
    image_size = definition["image_size"]
    loss_name = definition["loss"]
    seed_everything(args.seed + fold_index)

    fold_dir = args.folds_dir / f"fold_{fold_index}"
    output_dir = args.output_root / args.experiment / f"fold_{fold_index}"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loaders = make_loaders(
        args.data_dir,
        fold_dir,
        image_size,
        args.batch_size,
        args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if "train" not in loaders or "val" not in loaders:
        raise FileNotFoundError(f"Fold {fold_index} must contain train.csv and val.csv in {fold_dir}")

    model = build_model("unet", base_channels=args.base_channels).to(device)
    criterion = build_loss(loss_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: list[dict[str, float]] = []
    best_dice = float("-inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device, scaler, amp_enabled)
        val_metrics, _, val_loss = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"{args.experiment} fold={fold_index} epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
        )
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(
                {
                    "experiment": args.experiment,
                    "model_name": "unet",
                    "image_size": image_size,
                    "loss": loss_name,
                    "base_channels": args.base_channels,
                    "state_dict": model.state_dict(),
                    "best_val_dice": best_dice,
                    "epoch": epoch,
                },
                checkpoint_dir / "best.pt",
            )

    write_rows(output_dir / "history.csv", history)
    save_training_curves(history, output_dir / "training_curves.png")

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=True)
    model = build_model("unet", base_channels=checkpoint.get("base_channels", args.base_channels)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics, rows, _ = evaluate(model, loaders["val"], criterion=None, device=device)
    metrics = {**metrics, "best_epoch": best_epoch, "best_val_dice": best_dice}
    save_json(output_dir / "val" / "metrics.json", metrics)
    write_rows(output_dir / "val" / "predictions.csv", rows)
    datasets = make_datasets(args.data_dir, fold_dir, image_size)
    save_prediction_examples(model, datasets["val"], device, output_dir / "val" / "examples", args.examples)
    save_json(output_dir / "run_config.json", experiment_config(args, fold_index))
    return {"fold": fold_index, **metrics}


def aggregate_metrics(fold_metrics: list[dict[str, object]]) -> dict[str, object]:
    aggregate: dict[str, object] = {"folds": fold_metrics, "mean": {}, "std": {}}
    for key in METRIC_KEYS:
        values = torch.tensor([float(metrics[key]) for metrics in fold_metrics])
        aggregate["mean"][key] = float(values.mean())
        aggregate["std"][key] = float(values.std(unbiased=False))
    return aggregate


def save_report_summary(output_root: Path, reports_dir: Path) -> None:
    rows = []
    for aggregate_path in sorted(output_root.glob("*/aggregate_metrics.json")):
        payload = load_json(aggregate_path)
        rows.append(
            {
                "experiment": aggregate_path.parent.name,
                "dice_mean": payload["mean"]["dice"],
                "dice_std": payload["std"]["dice"],
                "iou_mean": payload["mean"]["iou"],
                "iou_std": payload["std"]["iou"],
            }
        )
    write_rows(reports_dir / "final_metrics_summary.csv", rows)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))
    device = resolve_device(args.device)
    model_dir = args.output_root / args.experiment
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        **EXPERIMENTS[args.experiment],
    }
    save_json(model_dir / "config.json", config)

    fold_metrics = []
    for fold_index in args.folds:
        output_dir = model_dir / f"fold_{fold_index}"
        expected_config = experiment_config(args, fold_index)
        completed = None if args.force else load_completed_fold(output_dir, expected_config)
        if completed is not None:
            print(f"{args.experiment} fold={fold_index} already complete with matching config; reusing metrics")
            fold_metrics.append({"fold": fold_index, **completed})
            continue
        fold_metrics.append(train_fold(args, fold_index, device))

    aggregate = aggregate_metrics(fold_metrics)
    save_json(model_dir / "aggregate_metrics.json", aggregate)
    write_rows(model_dir / "fold_metrics.csv", fold_metrics)
    save_report_summary(args.output_root, REPO_ROOT / "reports")
    print("aggregate metrics")
    for key in METRIC_KEYS:
        print(f"{key}: {aggregate['mean'][key]:.4f} +/- {aggregate['std'][key]:.4f}")


if __name__ == "__main__":
    main()

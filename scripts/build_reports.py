from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_reports(runs_root: Path, reports_dir: Path, max_examples: int = 3) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for experiment_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        aggregate_path = experiment_dir / "aggregate_metrics.json"
        if not aggregate_path.exists():
            continue
        with aggregate_path.open(encoding="utf-8") as handle:
            aggregate = json.load(handle)

        experiment_name = experiment_dir.name
        experiment_report_dir = reports_dir / "experiments" / experiment_name
        copy_if_exists(aggregate_path, experiment_report_dir / "aggregate_metrics.json")
        copy_if_exists(experiment_dir / "fold_metrics.csv", experiment_report_dir / "fold_metrics.csv")
        copy_if_exists(experiment_dir / "config.json", experiment_report_dir / "config.json")

        first_fold = experiment_dir / "fold_0"
        copy_if_exists(first_fold / "training_curves.png", experiment_report_dir / "fold_0_training_curves.png")
        examples_dir = first_fold / "val" / "examples"
        if examples_dir.exists():
            for example_path in sorted(examples_dir.glob("*.png"))[:max_examples]:
                copy_if_exists(example_path, experiment_report_dir / "examples" / example_path.name)

        rows.append(
            {
                "experiment": experiment_name,
                "dice_mean": aggregate["mean"]["dice"],
                "dice_std": aggregate["std"]["dice"],
                "iou_mean": aggregate["mean"]["iou"],
                "iou_std": aggregate["std"]["iou"],
            }
        )

    write_rows(reports_dir / "final_metrics_summary.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect lightweight BUSI segmentation reports.")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs_cv")
    parser.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "reports")
    parser.add_argument("--max-examples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_reports(args.runs_root, args.reports_dir, args.max_examples)


if __name__ == "__main__":
    main()

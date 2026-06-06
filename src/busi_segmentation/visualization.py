from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def save_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_dice"] for row in history], label="Dice")
    axes[1].plot(epochs, [row["val_iou"] for row in history], label="IoU")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


@torch.inference_mode()
def save_prediction_examples(model, dataset, device, output_dir: Path, max_images: int = 8) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for index in range(min(max_images, len(dataset))):
        image_tensor, mask_tensor, relative_path = dataset[index]
        logits = model(image_tensor.unsqueeze(0).to(device))
        prediction = (torch.sigmoid(logits)[0, 0].cpu().numpy() >= 0.5).astype(np.float32)
        image = image_tensor[0].numpy()
        mask = mask_tensor[0].numpy()

        _, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Image")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask")
        axes[2].imshow(image, cmap="gray")
        axes[2].imshow(prediction, cmap="Reds", alpha=0.45)
        axes[2].set_title("Prediction")
        for axis in axes:
            axis.axis("off")
        plt.tight_layout()
        output_path = output_dir / f"{index:03d}_{Path(relative_path).stem}.png"
        plt.savefig(output_path, dpi=160)
        plt.close()

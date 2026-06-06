# BUSI Lesion Segmentation U-Net

BUSI 乳腺超声病灶分割 baseline 项目。当前阶段只复现基础 U-Net，并只使用 `benign` 和 `malignant` 图像做二值病灶区域分割。

## Experiments

| Experiment | Input | Loss | Metrics |
| --- | ---: | --- | --- |
| `unet_256_bce_dice` | 256 x 256 | BCE + Dice | Dice, IoU |
| `unet_512_bce_dice` | 512 x 512 | BCE + Dice | Dice, IoU |
| `unet_256_dice` | 256 x 256 | Dice Loss | Dice, IoU |
| `unet_256_bce` | 256 x 256 | BCE Loss | Dice, IoU |

当前四组基础 U-Net baseline 已完成。

## Dataset

本仓库不包含 BUSI 数据集。请将原始数据集放在本地或云端，例如：

```text
Dataset_BUSI_with_GT/
├── benign/
├── malignant/
└── normal/
```

当前分割任务只使用 `benign` 和 `malignant`。同一原图存在多个 `_mask` 文件时，会将所有 mask 做像素级并集。

## Prepare Folds

```powershell
python scripts\prepare_busi_seg.py --data-dir "F:\Deep_learning\CNN vs ViT vs Swin Transformer\Dataset_BUSI_with_GT"
```

输出：

```text
data/folds/
├── all.csv
├── metadata.json
├── fold_0/train.csv, val.csv
...
└── fold_4/train.csv, val.csv
```

## Train

```powershell
python scripts\train_cv.py --experiment unet_256_bce_dice --epochs 50 --batch-size 32 --amp
python scripts\train_cv.py --experiment unet_256_dice --epochs 50 --batch-size 32 --amp
python scripts\train_cv.py --experiment unet_256_bce --epochs 50 --batch-size 32 --amp
python scripts\train_cv.py --experiment unet_512_bce_dice --epochs 50 --batch-size 8 --amp
```

云端 AutoDL 使用：

```bash
/root/miniconda3/bin/python scripts/prepare_busi_seg.py --data-dir /root/autodl-tmp/CNN_vs_ViT_vs_Swin_Transformer/Dataset_BUSI_with_GT
/root/miniconda3/bin/python scripts/train_cv.py --experiment unet_256_bce_dice --epochs 50 --batch-size 32 --num-workers 4 --amp
```

## Outputs

- `runs_cv/`: 完整训练产物、checkpoint、history、预测 CSV。
- `reports/`: 轻量汇总表和预测可视化，可上传到 GitHub。

`runs_cv/` 和权重文件不上传到 GitHub。

## Current Results

云端环境：

```text
NVIDIA GeForce RTX 4080 SUPER 32GB
PyTorch 2.7.0+cu128
batch size 32
epochs 50
5-fold cross-validation
```

| Experiment | Dice | IoU |
| --- | ---: | ---: |
| `unet_256_bce_dice` | 0.7431 +/- 0.0155 | 0.6457 +/- 0.0176 |
| `unet_256_bce` | 0.7240 +/- 0.0296 | 0.6281 +/- 0.0309 |
| `unet_256_dice` | 0.7093 +/- 0.0359 | 0.6107 +/- 0.0368 |
| `unet_512_bce_dice` | 0.6854 +/- 0.0113 | 0.5768 +/- 0.0127 |

完整 checkpoint 和 `runs_cv/` 保留在云端实例；本地只同步轻量 `reports/`、fold 清单和训练日志。

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EfficientNet-B3 角色分类模型训练脚本

基于项目 final_dataset 数据训练 EfficientNet-B3 分类模型，
产出格式与 model_loader.py 完全兼容：
  - models/efficientnet_b3/model_best.pth (state_dict checkpoint)
  - models/efficientnet_b3/class_to_idx.json
  - models/efficientnet_b3/training_results.json

模型架构与 create_model_from_name("efficientnet_b3", num_classes) 一致：
  backbone: EfficientNet-B3 (pretrained)
  classifier: Dropout(0.3) → Linear(1536→768) → ReLU → BN(768) → Dropout(0.15) → Linear(768→num_classes)
"""

import os
import sys
import json
import time
import copy
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy

# 防泄漏：以角色目录为 group 做分组切分（同角色图整体进 train 或 val，不拆散）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.core.data.split_utils import grouped_split  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─── 项目路径 ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "final_dataset"
MODEL_DIR = PROJECT_ROOT / "models" / "efficientnet_b3"

# ─── 默认训练配置 ───
DEFAULT_CONFIG = {
    "image_size": 300,       # EfficientNet-B3 推荐输入尺寸
    "batch_size": 16,
    "num_epochs": 40,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "patience": 8,           # 早停耐心
    "min_images_per_class": 30,
    "train_ratio": 0.85,
    "seed": 42,
    "use_auto_augment": True,
    "use_weighted_sampler": True,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.2,
}


def get_best_device():
    """自动检测最佳设备"""
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.backends.cuda.is_available():
        logger.info("Using CUDA GPU")
        return torch.device("cuda")
    else:
        logger.info("Using CPU only")
        return torch.device("cpu")


def create_efficientnet_b3(num_classes: int, weights=models.EfficientNet_B3_Weights.DEFAULT) -> nn.Module:
    """创建与 model_loader.py 完全一致的 EfficientNet-B3 架构

    weights: 默认使用 ImageNet 预训练权重（v1 训练 / train_clean_split 需要）；
    增量微调与评测场景应传入 weights=None，跳过 47MB 权重下载——
    因为随即 strict 加载 checkpoint 会把整个主干覆盖掉，下载的权重永不使用。
    """
    model = models.efficientnet_b3(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 768),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(768),
        nn.Dropout(p=0.15),
        nn.Linear(768, num_classes),
    )
    return model


class MixupCriterion:
    """Mixup 数据增强的损失函数"""
    def __init__(self, criterion, alpha=0.2):
        self.criterion = criterion
        self.alpha = alpha

    def __call__(self, model, inputs, targets):
        if self.alpha > 0 and torch.rand(1).item() > 0.5:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
            batch_size = inputs.size(0)
            index = torch.randperm(batch_size, device=inputs.device)
            mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
            outputs = model(mixed_inputs)
            loss = lam * self.criterion(outputs, targets) + \
                   (1 - lam) * self.criterion(outputs, targets[index])
            return loss, outputs, targets  # 返回原始 targets 用于计算准确率
        else:
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            return loss, outputs, targets


def get_transforms(image_size: int, use_auto_augment: bool = True):
    """创建训练和验证的数据变换"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET) if use_auto_augment else transforms.RandomHorizontalFlip(p=0.0),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform


def filter_dataset_by_min_images(data_dir: str, min_images: int):
    """过滤出图片数量 >= min_images 的类别"""
    valid_classes = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        count = len([f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))])
        if count >= min_images:
            valid_classes.append(class_name)
    return valid_classes


class FilteredImageDataset(torch.utils.data.Dataset):
    """自定义 Dataset，直接从 (path, label) 样本列表构建"""
    def __init__(self, samples, transform, idx_to_class):
        self.samples = samples
        self.transform = transform
        self.idx_to_class = idx_to_class

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def create_weighted_sampler_from_samples(samples):
    """创建加权随机采样器，直接从 (path, label) 样本列表计算权重，不打开图片"""
    class_counts = {}
    for _, label in samples:
        class_counts[label] = class_counts.get(label, 0) + 1

    sample_weights = [1.0 / class_counts[label] for _, label in samples]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device, phase="train"):
    """训练/验证一个 epoch"""
    if phase == "train":
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if phase == "train":
            optimizer.zero_grad(set_to_none=True)

            # Mixup
            loss, outputs, orig_labels = criterion(model, inputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss(label_smoothing=0.0)(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == orig_labels if phase == "train" else preds == labels).sum().item()
        total_samples += inputs.size(0)

        if batch_idx % 20 == 0 and phase == "train":
            logger.info(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / max(total_samples, 1)
    epoch_acc = running_corrects / max(total_samples, 1)

    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, best_acc, class_to_idx, model_dir, filename="model_best.pth"):
    """保存 checkpoint（与 model_loader.py 兼容格式）"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "class_to_idx": class_to_idx,
    }
    path = model_dir / filename
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")
    return path


def save_training_results(model_dir, class_to_idx, train_config, metrics, best_acc, training_time):
    """保存训练结果（与 model_loader.py 兼容格式）"""
    results = {
        "model_name": "efficientnet_b3",
        "architecture": "EfficientNet-B3",
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "class_names": sorted(class_to_idx.keys()),
        "image_size": train_config.get("image_size", 300),
        "best_accuracy": float(best_acc),
        "training_time_seconds": training_time,
        "training_config": train_config,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_path = model_dir / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved training results to {results_path}")

    # 同时保存 class_to_idx.json
    class_map_path = model_dir / "class_to_idx.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved class mapping to {class_map_path}")


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B3 anime character classifier")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--min-images", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--no-mixup", action="store_true")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--no-auto-augment", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--device", type=str, default=None, choices=["mps", "cpu", "cuda"])
    args = parser.parse_args()

    # 合并配置
    config = DEFAULT_CONFIG.copy()
    if args.epochs: config["num_epochs"] = args.epochs
    if args.batch_size: config["batch_size"] = args.batch_size
    if args.lr: config["learning_rate"] = args.lr
    if args.image_size: config["image_size"] = args.image_size
    if args.min_images: config["min_images_per_class"] = args.min_images
    if args.patience: config["patience"] = args.patience
    if args.no_mixup: config["mixup_alpha"] = 0.0
    if args.no_weighted_sampler: config["use_weighted_sampler"] = False
    if args.no_auto_augment: config["use_auto_augment"] = False

    # 设置随机种子
    torch.manual_seed(config["seed"])
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config["seed"])

    device = torch.device(args.device) if args.device else get_best_device()
    logger.info(f"Device: {device}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # ─── 创建模型目录 ───
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ─── 直接扫描有足够图片的类别目录 ───
    valid_classes = filter_dataset_by_min_images(str(DATA_DIR), config["min_images_per_class"])
    logger.info(f"Valid classes (>= {config['min_images_per_class']} images): {len(valid_classes)}")

    # 收集所有有效样本路径
    new_class_to_idx = {cls: idx for idx, cls in enumerate(sorted(valid_classes))}
    num_classes = len(new_class_to_idx)
    logger.info(f"Number of classes: {num_classes}")

    all_samples = []  # (image_path, class_label_idx)
    for class_name, class_idx in new_class_to_idx.items():
        class_dir = os.path.join(str(DATA_DIR), class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
                all_samples.append((os.path.join(class_dir, fname), class_idx))

    logger.info(f"Total valid samples: {len(all_samples)} images")

    # ─── 划分训练/验证集 ───
    train_transform, val_transform = get_transforms(config["image_size"], config["use_auto_augment"])
    new_idx_to_class = {idx: cls for cls, idx in new_class_to_idx.items()}

    train_ratio = config["train_ratio"]

    # ── 按角色分组切分（MUST group by character to avoid leakage）──
    # 旧逻辑按"类内逐图洗牌"会把同一角色的图同时放进 train/val（泄漏）。
    # 改为以角色目录（路径倒数第 2 段）为 group 做 GroupShuffleSplit：
    # 同一角色的所有图整体进 train 或整体进 val，杜绝 train/val 同源。
    char_groups = [Path(path).parts[-2] for path, _ in all_samples]
    train_idx, val_idx, _ = grouped_split(
        all_samples, char_groups,
        ratios=(train_ratio, 1.0 - train_ratio, 0.0),
        seed=config["seed"],
    )
    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]

    logger.info(f"Train: {len(train_samples)} images, Val: {len(val_samples)} images")
    sys.stdout.flush()

    print("[DEBUG] Creating datasets...", flush=True)
    train_dataset = FilteredImageDataset(train_samples, train_transform, new_idx_to_class)
    val_dataset = FilteredImageDataset(val_samples, val_transform, new_idx_to_class)
    print("[DEBUG] Datasets created", flush=True)

    # macOS 上统一用 num_workers=0 避免多进程 pickle 问题
    _num_workers = 0
    _pin_memory = device.type == "cuda"

    print("[DEBUG] Creating DataLoader...", flush=True)
    if config["use_weighted_sampler"]:
        sampler = create_weighted_sampler_from_samples(train_samples)
        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"],
            sampler=sampler, num_workers=_num_workers, pin_memory=_pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"],
            shuffle=True, num_workers=_num_workers, pin_memory=_pin_memory,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=_num_workers, pin_memory=_pin_memory,
    )
    print("[DEBUG] DataLoader created", flush=True)

    # ─── 创建模型 ───
    print("[DEBUG] Creating EfficientNet-B3 model...", flush=True)
    model = create_efficientnet_b3(num_classes)
    print("[DEBUG] Model created, moving to device...", flush=True)
    model = model.to(device)
    print("[DEBUG] Model on device", flush=True)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: EfficientNet-B3, Total params: {total_params}, Trainable: {trainable_params}")
    logger.info(f"Classes: {num_classes} ({new_class_to_idx})")

    # ─── 损失函数和优化器 ───
    base_criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    mixup_criterion = MixupCriterion(base_criterion, alpha=config["mixup_alpha"])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # 学习率调度器：Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    # ─── 断点续训 ───
    start_epoch = 0
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    metrics_history = []

    resume_path = MODEL_DIR / "model_best.pth"
    if args.resume and resume_path.exists():
        logger.info(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        best_model_wts = copy.deepcopy(model.state_dict())
        # 推进 scheduler 到正确位置
        for _ in range(start_epoch):
            scheduler.step()
        logger.info(f"Resumed at epoch {start_epoch}, best_acc={best_acc:.4%}")
        # 加载已有 metrics
        results_path = MODEL_DIR / "training_results.json"
        if results_path.exists():
            with open(results_path) as f:
                old_results = json.load(f)
            metrics_history = old_results.get("metrics", [])
    else:
        logger.info("Starting fresh training (no checkpoint to resume from)")

    logger.info("=" * 60)
    logger.info(f"Training: EfficientNet-B3, {num_classes} classes, {device}, start_epoch={start_epoch}")
    logger.info("=" * 60)

    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + config["num_epochs"]):
        logger.info(f"\nEpoch {epoch+1}/{start_epoch + config['num_epochs']}")

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, mixup_criterion, optimizer, device, phase="train"
        )

        # 验证
        val_loss, val_acc = train_one_epoch(
            model, val_loader, base_criterion, optimizer, device, phase="val"
        )

        # 更新学习率
        scheduler.step()

        # 记录指标
        current_lr = optimizer.param_groups[0]["lr"]
        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        logger.info(
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4%} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4%} | LR: {current_lr:.6f}"
        )

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logger.info(f"  *** Best model saved! Val Acc: {best_acc:.4%} ***")

            # 保存 checkpoint
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                new_class_to_idx, MODEL_DIR, "model_best.pth"
            )
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                logger.info(f"  Early stopping: no improvement for {config['patience']} epochs")
                break

    # ─── 加载最佳模型并保存最终结果 ───
    model.load_state_dict(best_model_wts)
    training_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info(f"Training completed in {training_time:.0f}s ({training_time/60:.1f}min)")
    logger.info(f"Best validation accuracy: {best_acc:.4%}")
    logger.info("=" * 60)

    # 保存训练结果
    save_training_results(
        MODEL_DIR, new_class_to_idx, config, metrics_history, best_acc, training_time
    )

    # 同时保存 model_full.pth（完整模型权重，model_loader 优先加载）
    checkpoint_final = {
        "epoch": len(metrics_history),
        "model_state_dict": model.state_dict(),
        "best_acc": best_acc,
        "class_to_idx": new_class_to_idx,
    }
    torch.save(checkpoint_final, MODEL_DIR / "model_full.pth")
    logger.info(f"Saved model_full.pth to {MODEL_DIR}")

    logger.info(f"\nAll training artifacts saved to {MODEL_DIR}/")
    logger.info(f"  - model_best.pth")
    logger.info(f"  - model_full.pth")
    logger.info(f"  - class_to_idx.json")
    logger.info(f"  - training_results.json")

    return best_acc


if __name__ == "__main__":
    best_acc = main()
    logger.info(f"Final best accuracy: {best_acc:.4%}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练角色分类模型 - 优化版本V2
针对过拟合和类别不平衡问题的全面优化

主要优化：
1. 增强数据增强（旋转、翻转、颜色抖动、随机裁剪、MixUp）
2. 加权损失函数（解决类别不平衡）
3. 标签平滑正则化
4. 改进的早停策略
5. 分层学习率 + 余弦退火调度
6. Dropout增强
7. 混合精度训练
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json
from datetime import datetime
from collections import Counter

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger

    logger = get_logger("train_loli_optimized_v2")
except ModuleNotFoundError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_loli_optimized_v2")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# 优化后的超参数
BATCH_SIZE = 32
NUM_EPOCHS = 80
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 0
PATIENCE = 15
EARLY_STOP_DELTA = 0.001  # 早停阈值
WEIGHT_DECAY = 1e-4  # 权重衰减
LABEL_SMOOTHING = 0.1  # 标签平滑

MODEL_TYPE = "efficientnet_b3"
DATA_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset"
MODEL_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/models"


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        class_names = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
            ]
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

        logger.info(f"数据集: {len(self.samples)} 样本, {len(self.class_to_idx)} 类别")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"加载失败 {img_path}: {e}")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
            if self.transform:
                image = self.transform(image)
            return image, label


def get_transforms(augment_level="heavy"):
    """获取增强的数据变换"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if augment_level == "heavy":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, val_transform


def calculate_class_weights(dataset):
    """计算类别权重，解决类别不平衡"""
    labels = [sample[1] for sample in dataset.samples]
    label_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(label_counts)

    weights = []
    for idx in range(num_classes):
        count = label_counts.get(idx, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    logger.info(f"类别权重计算完成，最小权重: {min(weights):.2f}, 最大权重: {max(weights):.2f}")
    return torch.tensor(weights)


def create_weighted_sampler(dataset):
    """创建加权采样器"""
    labels = [sample[1] for sample in dataset.samples]
    label_counts = Counter(labels)
    weights = []

    for label in labels:
        weights.append(1.0 / label_counts[label])

    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler


def get_model(model_type, num_classes, dropout_rate=0.3):
    """创建带Dropout增强的模型"""
    logger.info(f"创建模型: {model_type}, 类别数: {num_classes}, Dropout: {dropout_rate}")

    if model_type == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

        # 冻结前几层
        for param in model.parameters():
            param.requires_grad = False

        # 解冻后面的特征提取层
        for param in model.features[-6:].parameters():
            param.requires_grad = True

        # 修改分类器，增加Dropout
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.classifier[1].in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(1024, num_classes),
        )
    elif model_type == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        for param in model.parameters():
            param.requires_grad = False

        for param in model.features[-5:].parameters():
            param.requires_grad = True

        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes),
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model


def mixup_data(x, y, alpha=1.0):
    """MixUp数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_mixup and np.random.random() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 30 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train_model(model_type, train_loader, val_loader, num_classes, class_weights, device):
    model = get_model(model_type, num_classes, dropout_rate=0.4).to(device)

    # 使用加权损失函数解决类别不平衡
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=LABEL_SMOOTHING
    )

    # 分层学习率
    optimizer = optim.AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": LEARNING_RATE * 0.1,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": model.classifier.parameters(),
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
        ]
    )

    # 余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=LEARNING_RATE * 0.01  # 第一个周期长度  # 后续周期翻倍
    )

    best_acc = 0.0
    best_model_state = None
    patience_counter = 0
    train_history = []

    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        logger.info(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_mixup=True
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        train_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        # 早停判断
        if val_acc > best_acc + EARLY_STOP_DELTA:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info(f"  保存最佳模型，准确率: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(
                    f"  验证准确率连续 {PATIENCE} 轮未提升超过 {EARLY_STOP_DELTA}, 提前停止训练"
                )
                break

    model.load_state_dict(best_model_state)
    logger.info(f"\n训练完成，最佳验证准确率: {best_acc:.2f}%")
    return model, best_acc, train_history


def main():
    logger.info("🚀 开始训练角色分类模型 (优化版本V2)")

    # 设置设备
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"使用设备: {device}")

    # 获取数据变换
    train_transform, val_transform = get_transforms(augment_level="heavy")

    # 加载数据集
    full_dataset = CustomImageDataset(DATA_DIR, transform=None)

    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # 应用变换
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # 计算类别权重
    class_weights = calculate_class_weights(full_dataset)

    # 创建数据加载器（不使用采样器，改用采样权重在损失函数中处理）
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False
    )

    # 训练模型
    model, best_acc, train_history = train_model(
        MODEL_TYPE, train_loader, val_loader, len(full_dataset.class_to_idx), class_weights, device
    )

    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{MODEL_TYPE}_loli_optimized_v2_{timestamp}"
    model_dir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # 保存完整模型
    torch.save(model, os.path.join(model_dir, "model_full.pth"))

    # 保存状态字典
    torch.save(model.state_dict(), os.path.join(model_dir, "model_best.pth"))

    # 保存训练结果
    results = {
        "model_name": MODEL_TYPE,
        "num_classes": len(full_dataset.class_to_idx),
        "class_names": list(full_dataset.class_to_idx.keys()),
        "best_accuracy": best_acc / 100,
        "train_samples": train_size,
        "val_samples": val_size,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "train_history": train_history,
        "timestamp": timestamp,
    }

    with open(os.path.join(model_dir, "training_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n🎉 模型训练完成！")
    logger.info(f"模型保存路径: {model_dir}")
    logger.info(f"最佳验证准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    main()

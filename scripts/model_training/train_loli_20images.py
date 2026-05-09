#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练角色分类模型 - 使用每个角色20张图片的整理数据 (68个角色)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("train_loli_20images")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_loli_20images")

try:
    from src.services.notification_service import (
        send_notification,
        send_training_progress_notification,
        send_training_complete_notification,
        send_training_error_notification
    )
    NOTIFICATION_AVAILABLE = True
except ImportError:
    logger.warning("通知服务导入失败，训练通知功能将不可用")
    NOTIFICATION_AVAILABLE = False
    def send_notification(*args, **kwargs): pass
    def send_training_progress_notification(*args, **kwargs): pass
    def send_training_complete_notification(*args, **kwargs): pass
    def send_training_error_notification(*args, **kwargs): pass

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 0

MODEL_TYPES = ['mobilenet_v2', 'efficientnet_b0', 'resnet18']
DATA_DIR = './data/organized_20_images'
MODEL_DIR = './models'


class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

        logger.info(f"数据集: {len(self.samples)} 样本, {len(self.class_to_idx)} 类别")
        logger.info(f"类别: {list(self.class_to_idx.keys())}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"加载失败 {img_path}: {e}")
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
            if self.transform:
                image = self.transform(image)
            return image, label


def get_model(model_type, num_classes):
    logger.info(f"创建模型: {model_type}, 类别数: {num_classes}")

    if model_type == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
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
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model(model_type, train_loader, val_loader, num_classes, device):
    model = get_model(model_type, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        progress = (epoch + 1) / NUM_EPOCHS * 100
        send_training_progress_notification(
            stage=f"训练 {model_type}",
            progress=progress,
            message=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Val Acc: {val_acc:.2f}%",
            metrics={'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc}
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            logger.info(f"  保存最佳模型，准确率: {best_acc:.2f}%")

    model.load_state_dict(best_model_state)
    logger.info(f"训练完成，最佳验证准确率: {best_acc:.2f}%")
    return model, best_acc


def save_model(model, model_type, class_to_idx, accuracy):
    model_dir = os.path.join(MODEL_DIR, f"{model_type}_loli68_20imgs")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'model_best.pth')
    checkpoint = {
        'model_type': model_type,
        'class_to_idx': class_to_idx,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat(),
        'num_classes': len(class_to_idx),
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, model_path)

    full_model_path = os.path.join(model_dir, 'model_full.pth')
    torch.save(model, full_model_path)

    class_map_path = os.path.join(model_dir, 'class_to_idx.json')
    with open(class_map_path, 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    logger.info(f"模型已保存: {model_path}")
    return model_path


def main():
    import time
    training_start_time = time.time()

    logger.info("=" * 60)
    logger.info("开始训练角色分类模型 (68个角色, 每角色20张图片)")
    logger.info("=" * 60)

    send_notification(f"🎯 开始训练 {len(MODEL_TYPES)} 个模型\n模型类型: {', '.join(MODEL_TYPES)}\nEpochs: {NUM_EPOCHS}\nBatch Size: {BATCH_SIZE}\n数据目录: {DATA_DIR}", level="info")

    # 检查数据目录
    if not os.path.exists(DATA_DIR):
        logger.error(f"数据目录不存在: {DATA_DIR}")
        send_notification(f"❌ 数据目录不存在: {DATA_DIR}", level="error")
        return

    # 检测可用设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"使用设备: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    logger.info(f"加载数据: {DATA_DIR}")
    full_dataset = SimpleImageDataset(DATA_DIR, transform=train_transform)

    if len(full_dataset) == 0:
        logger.error("数据集为空")
        send_notification("❌ 数据集为空", level="error")
        return

    num_classes = len(full_dataset.class_to_idx)
    logger.info(f"类别数: {num_classes}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    logger.info(f"训练集: {len(train_dataset)}")
    logger.info(f"验证集: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    results = {}
    for model_type in MODEL_TYPES:
        try:
            logger.info("")
            logger.info(f"开始训练: {model_type}")
            logger.info("")

            model, best_acc = train_model(model_type, train_loader, val_loader, num_classes, device)
            model_path = save_model(model, model_type, full_dataset.class_to_idx, best_acc)

            results[model_type] = {
                'accuracy': best_acc,
                'model_path': model_path
            }

            logger.info(f"{model_type} 训练完成，准确率: {best_acc:.2f}%")

            send_training_complete_notification(
                model_name=f"{model_type}_loli68_20imgs",
                metrics={'accuracy': best_acc},
                model_path=model_path
            )

        except Exception as e:
            logger.error(f"{model_type} 训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[model_type] = {'error': str(e)}
            send_training_error_notification(stage=f"训练 {model_type}", error_message=str(e))

    logger.info("")
    logger.info("=" * 60)
    logger.info("所有模型训练完成")
    logger.info("=" * 60)

    training_elapsed = time.time() - training_start_time

    summary_message = f"✅ 全部 {len(MODEL_TYPES)} 个模型训练完成！\n总耗时: {training_elapsed / 60:.1f} 分钟\n\n📊 结果摘要:"
    has_error = False

    for model_type, result in results.items():
        if 'error' in result:
            logger.error(f"{model_type}: 训练失败 - {result['error']}")
            summary_message += f"\n❌ {model_type}: 失败"
            has_error = True
        else:
            logger.info(f"{model_type}: 准确率 {result['accuracy']:.2f}%, 路径 {result['model_path']}")
            summary_message += f"\n✅ {model_type}: {result['accuracy']:.2f}%"

    send_notification(summary_message, level="error" if has_error else "success")

    results_file = os.path.join(MODEL_DIR, 'training_results_loli68_20imgs.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"训练结果已保存: {results_file}")


if __name__ == '__main__':
    main()

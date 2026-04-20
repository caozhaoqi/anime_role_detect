#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练角色分类模型 - 使用所有已下载的8个角色
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
    logger = get_logger("train_loli_models")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_loli_models")

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 0

MODEL_TYPES = ['mobilenet_v2', 'efficientnet_b0', 'resnet18']
DATA_DIR = './data/downloaded_images'
MODEL_DIR = './models'


class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
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
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'resnet18':
        model = models.resnet18(weights=None)
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            logger.info(f"  保存最佳模型，准确率: {best_acc:.2f}%")

    model.load_state_dict(best_model_state)
    logger.info(f"训练完成，最佳验证准确率: {best_acc:.2f}%")
    return model, best_acc


def save_model(model, model_type, class_to_idx, accuracy):
    model_dir = os.path.join(MODEL_DIR, f"{model_type}_loli8")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'model_best.pth')
    checkpoint = {
        'model_type': model_type,
        'class_to_idx': class_to_idx,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
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
    logger.info("=" * 60)
    logger.info("开始训练角色分类模型 (8个角色)")
    logger.info("=" * 60)

    device = torch.device('cpu')
    logger.info(f"使用设备: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
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

        except Exception as e:
            logger.error(f"{model_type} 训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[model_type] = {'error': str(e)}

    logger.info("")
    logger.info("=" * 60)
    logger.info("所有模型训练完成")
    logger.info("=" * 60)

    for model_type, result in results.items():
        if 'error' in result:
            logger.error(f"{model_type}: 训练失败 - {result['error']}")
        else:
            logger.info(f"{model_type}: 准确率 {result['accuracy']:.2f}%, 路径 {result['model_path']}")

    results_file = os.path.join(MODEL_DIR, 'training_results_loli8.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"训练结果已保存: {results_file}")


if __name__ == '__main__':
    main()
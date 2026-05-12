#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练角色分类模型 - 抗过拟合优化版本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("train_loli_anti_overfit")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("train_loli_anti_overfit")

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 优化后的超参数
BATCH_SIZE = 16
NUM_EPOCHS = 35  # 减少训练轮数
LEARNING_RATE = 5e-5
IMAGE_SIZE = 224
NUM_WORKERS = 0
PATIENCE = 5  # 减少早停耐心值

MODEL_TYPES = ['efficientnet_b0']
DATA_DIR = './data/reorganized_dataset'
MODEL_DIR = './models/anti_overfit'


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


class CustomEfficientNet(nn.Module):
    """带Dropout正则化的EfficientNet模型"""
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # 冻结前几层
        for param in self.backbone.parameters():
            param.requires_grad = False
        # 解冻最后几层特征提取层
        for param in self.backbone.features[-5:].parameters():
            param.requires_grad = True
        # 添加Dropout层
        self.dropout = nn.Dropout(0.5)
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # 使用Dropout正则化
        x = self.backbone.classifier(x)
        return x


def get_model(model_type, num_classes):
    logger.info(f"创建模型: {model_type}, 类别数: {num_classes}")

    if model_type == 'efficientnet_b0':
        model = CustomEfficientNet(num_classes)
    elif model_type == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_type == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=PATIENCE):
    best_val_acc = 0.0
    early_stopping_counter = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info('-' * 50)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)

        logger.info(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.float() / len(val_loader.dataset)

        logger.info(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_model_wts = model.state_dict().copy()
            early_stopping_counter = 0
            logger.info(f'新的最佳验证准确率: {best_val_acc:.4f}')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logger.info(f'早停触发，验证准确率不再提升')
                break

    model.load_state_dict(best_model_wts)
    return model, best_val_acc


def main():
    logger.info("=" * 60)
    logger.info("🚀 开始训练角色分类模型 (抗过拟合版本)")
    logger.info("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 增强的数据增强策略
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomAffine(0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = SimpleImageDataset(DATA_DIR, transform=train_transform)

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    with open(os.path.join(MODEL_DIR, 'class_to_idx.json'), 'w', encoding='utf-8') as f:
        json.dump(full_dataset.class_to_idx, f, ensure_ascii=False, indent=2)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    for model_type in MODEL_TYPES:
        logger.info(f"\n训练模型: {model_type}")
        logger.info("-" * 50)

        model = get_model(model_type, len(full_dataset.class_to_idx))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        # 添加L2正则化（weight_decay）
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.3)

        model, best_acc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(MODEL_DIR, f'{model_type}_best.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_to_idx': full_dataset.class_to_idx,
            'best_acc': best_acc,
            'model_type': model_type
        }, model_path)

        logger.info(f"模型保存到: {model_path}")
        logger.info(f"最佳验证准确率: {best_acc:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("🎉 训练完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

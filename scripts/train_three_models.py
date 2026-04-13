#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练三种角色分类模型：mobilenet_v2、efficientnet_b0、resnet18
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

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("train_models")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_models")

# 设置环境变量，避免MPS问题
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 训练参数
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 0

# 模型类型
MODEL_TYPES = ['mobilenet_v2', 'efficientnet_b0', 'resnet18']

# 数据目录
DATA_DIR = './data/downloaded_images'
MODEL_DIR = './models'


class SimpleImageDataset(torch.utils.data.Dataset):
    """简单的图像数据集类"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # 加载数据
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
        
        logger.info(f"加载数据集: {len(self.samples)} 个样本, {len(self.class_to_idx)} 个类别")
        logger.info(f"类别映射: {self.class_to_idx}")
    
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
            logger.error(f"加载图像失败 {img_path}: {e}")
            # 返回一个随机图像作为后备
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
            if self.transform:
                image = self.transform(image)
            return image, label


def get_model(model_type, num_classes):
    """获取模型"""
    logger.info(f"创建模型: {model_type}, 类别数: {num_classes}")
    
    if model_type == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes)
        )
    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes)
        )
    elif model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
        feature_dim = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
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
        
        if (batch_idx + 1) % 5 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """验证模型"""
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
    """训练单个模型"""
    logger.info(f"=" * 50)
    logger.info(f"开始训练模型: {model_type}")
    logger.info(f"=" * 50)
    
    # 创建模型
    model = get_model(model_type, num_classes)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # 训练模型
    best_acc = 0.0
    best_model_state = None
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            logger.info(f"  保存最佳模型，验证准确率: {best_acc:.2f}%")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    logger.info(f"模型训练完成，最佳验证准确率: {best_acc:.2f}%")
    return model, best_acc


def save_model(model, model_type, class_to_idx, accuracy):
    """保存模型"""
    # 创建模型目录
    model_dir = os.path.join(MODEL_DIR, model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(model_dir, 'model_best.pth')
    
    checkpoint = {
        'model_type': model_type,
        'class_to_idx': class_to_idx,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存模型权重
    torch.save(checkpoint, model_path)
    
    # 同时保存完整的模型（包含结构）
    full_model_path = os.path.join(model_dir, 'model_full.pth')
    torch.save(model, full_model_path)
    
    logger.info(f"模型已保存到: {model_path}")
    logger.info(f"完整模型已保存到: {full_model_path}")
    
    return model_path


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始训练三种角色分类模型")
    logger.info("=" * 60)
    
    # 设备选择
    device = torch.device('cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据转换
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
    
    # 加载数据集
    logger.info(f"加载训练数据: {DATA_DIR}")
    full_dataset = SimpleImageDataset(DATA_DIR, transform=train_transform)
    
    num_classes = len(full_dataset.class_to_idx)
    logger.info(f"类别数: {num_classes}")
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 应用不同的数据转换
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    # 训练三种模型
    results = {}
    for model_type in MODEL_TYPES:
        try:
            logger.info("")
            logger.info(f"开始训练模型: {model_type}")
            logger.info("")
            
            # 创建新的模型实例
            model, best_acc = train_model(
                model_type,
                train_loader,
                val_loader,
                num_classes,
                device
            )
            
            # 保存模型
            model_path = save_model(
                model,
                model_type,
                full_dataset.class_to_idx,
                best_acc
            )
            
            results[model_type] = {
                'accuracy': best_acc,
                'path': model_path
            }
            
            logger.info(f"模型 {model_type} 训练完成，准确率: {best_acc:.2f}%")
            logger.info("")
            
        except Exception as e:
            logger.error(f"模型 {model_type} 训练失败: {e}")
            import traceback
            logger.error(f"异常堆栈: {traceback.format_exc()}")
            results[model_type] = {
                'accuracy': 0.0,
                'error': str(e)
            }
    
    # 打印结果汇总
    logger.info("=" * 60)
    logger.info("训练结果汇总")
    logger.info("=" * 60)
    for model_type, result in results.items():
        if 'error' in result:
            logger.info(f"{model_type}: 训练失败 - {result['error']}")
        else:
            logger.info(f"{model_type}: 准确率 {result['accuracy']:.2f}%")
    
    logger.info("=" * 60)
    logger.info("所有模型训练完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
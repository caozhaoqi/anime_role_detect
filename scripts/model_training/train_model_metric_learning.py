#!/usr/bin/env python3
"""
使用度量学习（ArcFace）训练模型
实现基于相似度的角色识别
"""
import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import RandomApply, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop, ColorJitter, GaussianBlur

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.loss.arcface_loss import ArcFaceLoss, CosFaceLoss

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_model_metric_learning')

class MetricLearningModel(nn.Module):
    """
    用于度量学习的模型
    """
    
    def __init__(self, num_classes, feature_dim=512, loss_type='arcface'):
        """
        初始化度量学习模型
        
        Args:
            num_classes: 类别数
            feature_dim: 特征维度
            loss_type: 损失函数类型，可选 'arcface' 或 'cosface'
        """
        super(MetricLearningModel, self).__init__()
        
        # 使用EfficientNet-B0作为骨干网络
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # 获取原始分类器的输入特征数
        in_features = self.backbone.classifier[1].in_features
        
        # 替换分类器为特征提取器
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        
        # 初始化损失函数
        if loss_type == 'arcface':
            self.loss_fn = ArcFaceLoss(feature_dim, num_classes)
        elif loss_type == 'cosface':
            self.loss_fn = CosFaceLoss(feature_dim, num_classes)
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")
        
        self.feature_dim = feature_dim
    
    def forward(self, x, labels=None):
        """
        前向传播
        
        Args:
            x: 输入图像
            labels: 标签，如果为None则只返回特征
            
        Returns:
            如果labels不为None，返回损失
            否则返回特征
        """
        features = self.backbone(x)
        
        if labels is not None:
            loss = self.loss_fn(features, labels)
            return loss
        else:
            # 归一化特征
            features = nn.functional.normalize(features, dim=1)
            return features

def get_data_loaders(train_dir, val_dir, batch_size, num_workers=4):
    """
    获取数据加载器
    
    Args:
        train_dir: 训练数据目录
        val_dir: 验证数据目录
        batch_size: 批量大小
        num_workers: 工作线程数
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        class_to_idx: 类别到索引的映射
    """
    # 训练数据增强
    train_transform = transforms.Compose([
        RandomResizedCrop(224, scale=(0.8, 1.0)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.2),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        RandomApply([GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证数据变换
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.class_to_idx

def train_epoch(model, train_loader, optimizer, scheduler, device):
    """
    训练一个轮次
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        
    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        loss = model(images, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新统计信息
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        # 打印进度
        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    # 更新学习率
    scheduler.step()
    
    avg_loss = total_loss / total_samples
    return avg_loss

def validate(model, val_loader, device):
    """
    验证模型
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            loss = model(images, labels)
            
            # 计算特征和预测
            features = model(images)
            
            # 计算与权重的相似度
            similarities = nn.functional.linear(features, model.loss_fn.weight)
            similarities = nn.functional.softmax(similarities * model.loss_fn.s, dim=1)
            
            # 获取预测
            _, predicted = torch.max(similarities, 1)
            
            # 更新统计信息
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    
    return avg_loss, accuracy

def save_model(model, class_to_idx, output_dir, epoch, loss, accuracy):
    """
    保存模型
    
    Args:
        model: 模型
        class_to_idx: 类别到索引的映射
        output_dir: 输出目录
        epoch: 轮次
        loss: 损失
        accuracy: 准确率
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型权重
    model_path = os.path.join(output_dir, f'metric_learning_model_epoch_{epoch}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'feature_dim': model.feature_dim,
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }, model_path)
    
    logger.info(f"模型已保存到: {model_path}")
    
    # 保存最佳模型
    best_model_path = os.path.join(output_dir, 'metric_learning_model_best.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'feature_dim': model.feature_dim,
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }, best_model_path)
    
    logger.info(f"最佳模型已保存到: {best_model_path}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='使用度量学习训练模型')
    
    parser.add_argument('--train-dir', type=str, required=True,
                       help='训练数据目录')
    parser.add_argument('--val-dir', type=str, required=True,
                       help='验证数据目录')
    parser.add_argument('--output-dir', type=str, default='models/metric_learning',
                       help='输出目录')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批量大小')
    parser.add_argument('--num-epochs', type=int, default=30,
                       help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='工作线程数')
    parser.add_argument('--loss-type', type=str, default='arcface',
                       choices=['arcface', 'cosface'],
                       help='损失函数类型')
    parser.add_argument('--feature-dim', type=int, default=512,
                       help='特征维度')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.train_dir):
        logger.error(f"训练目录不存在: {args.train_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.val_dir):
        logger.error(f"验证目录不存在: {args.val_dir}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 获取数据加载器
    logger.info("初始化数据集...")
    train_loader, val_loader, class_to_idx = get_data_loaders(
        args.train_dir, args.val_dir, args.batch_size, args.num_workers
    )
    
    num_classes = len(class_to_idx)
    logger.info(f"数据集初始化完成，包含 {num_classes} 个类别")
    
    # 创建模型
    logger.info("创建模型...")
    model = MetricLearningModel(
        num_classes=num_classes,
        feature_dim=args.feature_dim,
        loss_type=args.loss_type
    )
    model.to(device)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs
    )
    
    # 训练模型
    logger.info("开始训练模型...")
    best_accuracy = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"开始第 {epoch}/{args.num_epochs} 轮训练")
        
        # 训练
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        train_time = time.time() - start_time
        
        # 验证
        val_loss, val_accuracy = validate(model, val_loader, device)
        
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Val Accuracy: {val_accuracy:.4f}")
        logger.info(f"Time: {train_time:.2f}s")
        
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, class_to_idx, args.output_dir, epoch, val_loss, val_accuracy)
            logger.info(f"新的最佳模型，准确率: {best_accuracy:.4f}")
    
    logger.info(f"训练完成，最佳准确率: {best_accuracy:.4f}")

if __name__ == '__main__':
    main()

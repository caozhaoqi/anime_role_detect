#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的模型训练脚本

使用改进的训练策略减少过拟合，提升模型性能
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_optimized')


class CharacterDataset(Dataset):
    """角色数据集类"""
    
    def __init__(self, root_dir, transform=None):
        """初始化数据集
        
        Args:
            root_dir: 数据目录
            transform: 数据变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # 构建类别映射
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            
            # 遍历图像
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(idx)
        
        logger.info(f"数据集初始化完成，包含 {len(classes)} 个类别，{len(self.images)} 张图像")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class ImprovedCharacterClassifier(nn.Module):
    """改进的角色分类器模型，增加正则化"""
    
    def __init__(self, num_classes, dropout_rate=0.3):
        """初始化模型
        
        Args:
            num_classes: 类别数量
            dropout_rate: Dropout比率
        """
        super().__init__()
        # 使用EfficientNet-B0作为基础模型
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # 获取特征维度
        num_features = self.backbone.classifier[1].in_features
        
        # 添加Dropout层
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 替换分类头
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with tqdm(train_loader, desc=f'训练 Epoch {epoch}', unit='batch') as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix(loss=loss.item(), acc=100. * correct / total)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        with tqdm(val_loader, desc='验证', unit='batch') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix(loss=loss.item(), acc=100. * correct / total)
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model(args):
    """训练模型
    
    Args:
        args: 命令行参数
    """
    # 检测设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 增强的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 加载数据集
    train_dataset = CharacterDataset(args.train_dir, transform=train_transform)
    val_dataset = CharacterDataset(args.val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # 防止最后一个batch太小
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 初始化模型
    model = ImprovedCharacterClassifier(
        num_classes=len(train_dataset.class_to_idx),
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # 损失函数（添加标签平滑）
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # 优化器（使用AdamW）
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学习率调度器（使用OneCycleLR）
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 前10%的epoch用于warmup
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    
    # 训练循环
    best_val_acc = 0.0
    patience = args.early_stopping_patience
    patience_counter = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(args.num_epochs):
        logger.info(f"开始第 {epoch+1}/{args.num_epochs} 轮训练")
        logger.info(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练阶段
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        
        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录损失和准确率
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(f'Epoch {epoch+1}/{args.num_epochs}')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(args.output_dir, f'character_classifier_optimized_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'class_to_idx': train_dataset.class_to_idx
            }, model_path)
            logger.info(f'最佳模型已保存: {model_path}, 验证准确率: {best_val_acc:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f'验证准确率未提升，耐心计数: {patience_counter}/{patience}')
        
        # 早停
        if patience_counter >= patience:
            logger.info(f'验证准确率连续 {patience} 轮未提升，提前停止训练')
            break
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'character_classifier_optimized_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx
    }, final_model_path)
    logger.info(f'最终模型已保存: {final_model_path}')
    
    # 保存训练历史
    history_path = os.path.join(args.output_dir, 'training_history.txt')
    with open(history_path, 'w') as f:
        f.write('Epoch,Train Loss,Train Acc,Val Loss,Val Acc\n')
        for epoch in range(len(train_losses)):
            f.write(f'{epoch+1},{train_losses[epoch]:.4f},{train_accs[epoch]:.4f},{val_losses[epoch]:.4f},{val_accs[epoch]:.4f}\n')
    logger.info(f'训练历史已保存: {history_path}')
    
    return best_val_acc


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='优化的角色分类模型训练')
    
    # 数据参数
    parser.add_argument('--train-dir', type=str, default='data/split_dataset/train', help='训练集目录')
    parser.add_argument('--val-dir', type=str, default='data/split_dataset/val', help='验证集目录')
    parser.add_argument('--output-dir', type=str, default='models', help='模型输出目录')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--num-epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='权重衰减')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout比率')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='标签平滑系数')
    parser.add_argument('--num-workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--early-stopping-patience', type=int, default=15, help='早停耐心值')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info('开始训练优化模型...')
    logger.info(f'训练集目录: {args.train_dir}')
    logger.info(f'验证集目录: {args.val_dir}')
    logger.info(f'批量大小: {args.batch_size}')
    logger.info(f'训练轮数: {args.num_epochs}')
    logger.info(f'学习率: {args.learning_rate}')
    logger.info(f'权重衰减: {args.weight_decay}')
    logger.info(f'Dropout比率: {args.dropout_rate}')
    logger.info(f'标签平滑: {args.label_smoothing}')
    
    # 开始训练
    best_acc = train_model(args)
    
    logger.info(f'训练完成！最佳验证准确率: {best_acc:.2f}%')


if __name__ == "__main__":
    main()

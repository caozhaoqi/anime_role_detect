#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量训练脚本 - 支持在已有模型基础上继续训练新数据
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import json
import numpy as np
import random

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 使用项目的全局日志系统
from src.core.logging.global_logger import get_logger, log_training, log_error

logger = get_logger('train_incremental')


class CharacterDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_characters=None, existing_class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = existing_class_to_idx.copy() if existing_class_to_idx else {}
        
        all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        if target_characters:
            classes = [c for c in all_classes if any(tc in c for tc in target_characters)]
        else:
            classes = all_classes
        
        # 为新类别分配索引
        start_idx = len(self.class_to_idx)
        for cls in classes:
            if cls not in self.class_to_idx:
                self.class_to_idx[cls] = start_idx
                start_idx += 1
        
        # 加载所有图像
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])
            logger.info(f"角色 {cls}: {len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])} 张图像")
        
        logger.info(f"数据集初始化完成，包含 {len(self.class_to_idx)} 个类别，{len(self.images)} 张图像")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_model(model_type, num_classes, dropout_rate=0.3, checkpoint=None):
    if model_type == 'mobilenet_v2':
        logger.info(f"加载模型: MobileNetV2 (dropout={dropout_rate})")
        model = models.mobilenet_v2(pretrained=not checkpoint)
        
        # 调整分类器
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
    elif model_type == 'efficientnet_b0':
        logger.info(f"加载模型: EfficientNet-B0 (dropout={dropout_rate})")
        model = models.efficientnet_b0(pretrained=not checkpoint)
        
        # 调整分类器
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
    elif model_type == 'efficientnet_b3':
        logger.info(f"加载模型: EfficientNet-B3 (dropout={dropout_rate})")
        model = models.efficientnet_b3(pretrained=not checkpoint)
        
        # 调整分类器
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
    elif model_type == 'resnet50':
        logger.info(f"加载模型: ResNet50 (dropout={dropout_rate})")
        model = models.resnet50(pretrained=not checkpoint)
        
        # 调整分类器
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model


def load_existing_model(checkpoint_path, model_type, num_classes, dropout_rate=0.3):
    """加载已有模型并调整分类器以适应新的类别数"""
    logger.info(f"加载已有模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # 创建新模型
    model = get_model(model_type, num_classes, dropout_rate)
    
    # 检查模型类型是否相同
    if 'model_type' in checkpoint and checkpoint['model_type'].startswith(model_type):
        # 加载除分类器外的参数
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        # 过滤掉分类器层的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('classifier') and not k.startswith('fc')}
        
        # 更新模型参数
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        logger.info("模型加载完成，分类器已重置以适应新的类别数")
    else:
        logger.info("模型类型不同，使用新的预训练权重")
    
    return model, checkpoint


def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=0.0008, weight_decay=0.0003, 
                output_dir='models/incremental', class_to_idx=None, use_mixup=True, label_smoothing=0.08, 
                checkpoint=None, model_type='mobilenet_v2'):
    logger.info(f"开始增量训练，设备: {device}")
    logger.info(f"训练轮数: {num_epochs}, 学习率: {lr}, 权重衰减: {weight_decay}")
    logger.info(f"数据增强: Mixup={use_mixup}, 标签平滑={label_smoothing}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 不加载优化器状态，因为类别数发生了变化
    # 如果有 checkpoint，只继承最佳验证准确率
    if checkpoint and 'val_acc' in checkpoint:
        best_val_acc = checkpoint['val_acc']
        logger.info(f"继承之前的最佳验证准确率: {best_val_acc:.2f}%")
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01
    )
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    
    # 如果有 checkpoint，使用之前的最佳准确率
    if checkpoint and 'val_acc' in checkpoint:
        best_val_acc = checkpoint['val_acc']
        logger.info(f"继承之前的最佳验证准确率: {best_val_acc:.2f}%")
    
    patience = 30
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            if use_mixup and random.random() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)
                optimizer.zero_grad()
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})
        
        scheduler.step()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}: '
                   f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                   f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                   f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 保存模型，无论验证准确率如何
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'class_to_idx': class_to_idx
        }, os.path.join(output_dir, 'model_best.pth'))
        logger.info(f'保存模型，验证准确率: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            logger.info(f'早停: {patience} 个epoch没有改进')
            break
    
    results = {
        'model_type': f'{model_type}_incremental',
        'num_epochs': num_epochs,
        'best_val_accuracy': best_val_acc,
        'final_train_accuracy': train_accuracies[-1],
        'final_val_accuracy': val_accuracies[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'use_mixup': use_mixup,
        'label_smoothing': label_smoothing
    }
    
    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f'增量训练完成，最佳验证准确率: {best_val_acc:.2f}%')
    return results


def main():
    parser = argparse.ArgumentParser(description='增量训练脚本')
    parser.add_argument('--data-dir', type=str, default='../../data/train', help='数据目录')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2', 
                       choices=['mobilenet_v2', 'efficientnet_b0', 'efficientnet_b3', 'resnet50'],
                       help='模型类型')
    parser.add_argument('--checkpoint', type=str, default=None, help='已有模型的检查点路径')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0008, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0003, help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--label-smoothing', type=float, default=0.08, help='标签平滑系数')
    parser.add_argument('--use-mixup', action='store_true', help='使用Mixup数据增强')
    parser.add_argument('--output-dir', type=str, default='models/incremental', help='输出目录')
    
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载已有模型的类别映射（如果有）
    existing_class_to_idx = None
    checkpoint = None
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        if 'class_to_idx' in checkpoint:
            existing_class_to_idx = checkpoint['class_to_idx']
            logger.info(f"加载已有类别映射: {existing_class_to_idx}")
    
    logger.info('加载数据集...')
    full_dataset = CharacterDataset(args.data_dir, transform=train_transform, existing_class_to_idx=existing_class_to_idx)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = val_transform
    
    logger.info(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    num_classes = len(full_dataset.class_to_idx)
    
    if args.checkpoint:
        model, _ = load_existing_model(args.checkpoint, args.model_type, num_classes, args.dropout)
    else:
        model = get_model(args.model_type, num_classes, args.dropout)
    
    model = model.to(device)
    
    logger.info(f'模型类别数: {num_classes}')
    logger.info(f'类别映射: {full_dataset.class_to_idx}')
    
    results = train_model(
        model, train_loader, val_loader, device,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        class_to_idx=full_dataset.class_to_idx,
        use_mixup=args.use_mixup,
        label_smoothing=args.label_smoothing,
        checkpoint=checkpoint,
        model_type=args.model_type
    )
    
    logger.info('增量训练完成！')
    logger.info(f'最佳验证准确率: {results["best_val_accuracy"]:.2f}%')


if __name__ == '__main__':
    main()

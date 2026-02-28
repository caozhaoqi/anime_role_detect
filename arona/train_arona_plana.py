#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蔚蓝档案阿罗娜和普拉娜专用训练脚本
针对这两个角色进行模型优化训练
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import logging
from tqdm import tqdm
import json
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_arona_plana')


class CharacterDataset(Dataset):
    """角色数据集类"""
    
    def __init__(self, root_dir, transform=None, target_characters=None):
        """初始化数据集
        
        Args:
            root_dir: 数据目录
            transform: 数据变换
            target_characters: 目标角色列表，如果指定则只加载这些角色
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # 构建类别映射
        all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # 如果指定了目标角色，则只加载这些角色
        if target_characters:
            classes = [c for c in all_classes if any(tc in c for tc in target_characters)]
            # 同时保留其他角色，但减少样本数量
            other_classes = [c for c in all_classes if c not in classes]
            # 限制其他角色的样本数量，让目标角色占主导
            self.other_class_limit = 20  # 其他角色最多20张
        else:
            classes = all_classes
            other_classes = []
            self.other_class_limit = None
        
        idx = 0
        # 首先添加目标角色
        for cls in classes:
            self.class_to_idx[cls] = idx
            cls_dir = os.path.join(root_dir, cls)
            img_count = 0
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(idx)
                    img_count += 1
            logger.info(f"目标角色 {cls}: {img_count} 张图像")
            idx += 1
        
        # 然后添加其他角色（限制数量）
        for cls in other_classes:
            self.class_to_idx[cls] = idx
            cls_dir = os.path.join(root_dir, cls)
            img_count = 0
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    if self.other_class_limit and img_count >= self.other_class_limit:
                        break
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(idx)
                    img_count += 1
            idx += 1
        
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


def get_model(model_type, num_classes):
    """获取模型"""
    if model_type == 'mobilenet_v2':
        logger.info("加载模型: MobileNetV2")
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'efficientnet_b0':
        logger.info("加载模型: EfficientNet-B0")
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'resnet18':
        logger.info("加载模型: ResNet18")
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model


def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=0.0001, output_dir='models/arona_plana', class_to_idx=None):
    """训练模型"""
    logger.info(f"开始训练，设备: {device}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练记录
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    
    # 早停设置
    patience = 10
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 验证阶段
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
        
        # 更新学习率
        scheduler.step()
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}: '
                   f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                   f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': class_to_idx
            }, os.path.join(output_dir, 'model_best.pth'))
            logger.info(f'保存最佳模型，验证准确率: {val_acc:.2f}%')
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # 早停检查
        if no_improve_count >= patience:
            logger.info(f'早停: {patience} 个epoch没有改进')
            break
    
    # 保存训练结果
    results = {
        'model_type': 'mobilenet_v2',
        'num_epochs': num_epochs,
        'best_val_accuracy': best_val_acc,
        'final_train_accuracy': train_accuracies[-1],
        'final_val_accuracy': val_accuracies[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    
    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f'训练完成，最佳验证准确率: {best_val_acc:.2f}%')
    return results


def main():
    parser = argparse.ArgumentParser(description='蔚蓝档案阿罗娜和普拉娜专用训练脚本')
    parser.add_argument('--data-dir', type=str, default='data/train', help='数据目录')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2', 
                       choices=['mobilenet_v2', 'efficientnet_b0', 'resnet18'],
                       help='模型类型')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--output-dir', type=str, default='models/arona_plana', help='输出目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 目标角色
    target_characters = ['蔚蓝档案_阿罗娜', '蔚蓝档案_普拉娜']
    
    # 创建数据集
    logger.info('加载数据集...')
    full_dataset = CharacterDataset(args.data_dir, transform=train_transform, target_characters=target_characters)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 为验证集设置不同的变换
    val_dataset.dataset.transform = val_transform
    
    logger.info(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}')
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 创建模型
    num_classes = len(full_dataset.class_to_idx)
    model = get_model(args.model_type, num_classes)
    model = model.to(device)
    
    logger.info(f'模型类别数: {num_classes}')
    
    # 训练模型
    results = train_model(
        model, train_loader, val_loader, device,
        num_epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        class_to_idx=full_dataset.class_to_idx
    )
    
    logger.info('训练完成！')
    logger.info(f'最佳验证准确率: {results["best_val_accuracy"]:.2f}%')


if __name__ == '__main__':
    main()

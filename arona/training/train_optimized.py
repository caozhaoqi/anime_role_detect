#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化训练脚本 - 增加训练轮数，使用更多角色，进一步提升准确率
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import logging
from tqdm import tqdm
import json
import numpy as np
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_optimized')


class CharacterDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_characters=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        if target_characters:
            classes = [c for c in all_classes if any(tc in c for tc in target_characters)]
        else:
            classes = all_classes
        
        idx = 0
        for cls in classes:
            self.class_to_idx[cls] = idx
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(idx)
            logger.info(f"角色 {cls}: {len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])} 张图像")
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


def get_model(model_type, num_classes, dropout_rate=0.3):
    if model_type == 'mobilenet_v2':
        logger.info(f"加载模型: MobileNetV2 (dropout={dropout_rate})")
        model = models.mobilenet_v2(pretrained=True)
        
        for param in model.features.parameters():
            param.requires_grad = False
        
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
        model = models.efficientnet_b0(pretrained=True)
        
        for param in model.features.parameters():
            param.requires_grad = False
        
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
        model = models.resnet50(pretrained=True)
        
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False
        
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


def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.0008, weight_decay=0.0003, 
                output_dir='models/optimized', class_to_idx=None, use_mixup=True, label_smoothing=0.08):
    logger.info(f"开始训练，设备: {device}")
    logger.info(f"训练轮数: {num_epochs}, 学习率: {lr}, 权重衰减: {weight_decay}")
    logger.info(f"数据增强: Mixup={use_mixup}, 标签平滑={label_smoothing}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01
    )
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    
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
        
        if no_improve_count >= patience:
            logger.info(f'早停: {patience} 个epoch没有改进')
            break
    
    results = {
        'model_type': 'mobilenet_v2_optimized',
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
    
    logger.info(f'训练完成，最佳验证准确率: {best_val_acc:.2f}%')
    return results


def main():
    parser = argparse.ArgumentParser(description='优化训练脚本')
    parser.add_argument('--data-dir', type=str, default='../../data/train', help='数据目录')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2', 
                       choices=['mobilenet_v2', 'efficientnet_b0', 'resnet50'],
                       help='模型类型')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0008, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0003, help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--label-smoothing', type=float, default=0.08, help='标签平滑系数')
    parser.add_argument('--use-mixup', action='store_true', help='使用Mixup数据增强')
    parser.add_argument('--output-dir', type=str, default='../../models/optimized', help='输出目录')
    
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
    
    target_characters = None
    
    logger.info('加载数据集...')
    full_dataset = CharacterDataset(args.data_dir, transform=train_transform, target_characters=target_characters)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = val_transform
    
    logger.info(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    num_classes = len(full_dataset.class_to_idx)
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
        label_smoothing=args.label_smoothing
    )
    
    logger.info('训练完成！')
    logger.info(f'最佳验证准确率: {results["best_val_accuracy"]:.2f}%')


if __name__ == '__main__':
    main()

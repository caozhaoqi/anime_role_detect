#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带有属性标签的角色分类模型训练脚本
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import logging
from tqdm import tqdm
import json
import time
from models import get_model_with_attributes

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_with_attributes')


class CharacterAttributeDataset(Dataset):
    """带有属性标签的角色数据集类"""
    
    def __init__(self, root_dir, annotations_file, transform=None):
        """初始化数据集
        
        Args:
            root_dir: 数据目录
            annotations_file: 标注文件路径
            transform: 数据变换
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 加载标注
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 构建类别映射
        self.class_to_idx = {}
        idx = 0
        for ann in self.annotations:
            character = ann['character']
            if character not in self.class_to_idx:
                self.class_to_idx[character] = idx
                idx += 1
        
        logger.info(f"数据集初始化完成，包含 {len(self.class_to_idx)} 个类别，{len(self.annotations)} 张图像")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.root_dir, ann['image_path'])
        image = Image.open(img_path).convert('RGB')
        
        # 类别标签
        character = ann['character']
        label = self.class_to_idx[character]
        
        # 属性标签
        attribute_labels = ann['attribute_labels']
        attribute_labels = torch.tensor(attribute_labels, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, attribute_labels


def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=0.0001, output_dir='models/arona_plana_with_attributes', class_to_idx=None):
    """训练模型"""
    logger.info(f"开始训练，设备: {device}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 损失函数和优化器
    class_criterion = nn.CrossEntropyLoss()
    attribute_criterion = nn.MSELoss()  # 使用均方误差损失
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
        train_class_loss = 0.0
        train_attribute_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels, attribute_labels in pbar:
            images, labels, attribute_labels = images.to(device), labels.to(device), attribute_labels.to(device)
            
            optimizer.zero_grad()
            class_output, attribute_output = model(images)
            
            # 计算损失
            c_loss = class_criterion(class_output, labels)
            a_loss = attribute_criterion(attribute_output, attribute_labels)
            total_loss = c_loss + 0.5 * a_loss  # 属性损失权重
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_class_loss += c_loss.item()
            train_attribute_loss += a_loss.item()
            
            # 计算准确率
            _, predicted = torch.max(class_output, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
        
        train_loss = train_loss / len(train_loader)
        train_class_loss = train_class_loss / len(train_loader)
        train_attribute_loss = train_attribute_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_class_loss = 0.0
        val_attribute_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels, attribute_labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images, labels, attribute_labels = images.to(device), labels.to(device), attribute_labels.to(device)
                class_output, attribute_output = model(images)
                
                # 计算损失
                c_loss = class_criterion(class_output, labels)
                a_loss = attribute_criterion(attribute_output, attribute_labels)
                total_loss = c_loss + 0.5 * a_loss
                
                val_loss += total_loss.item()
                val_class_loss += c_loss.item()
                val_attribute_loss += a_loss.item()
                
                # 计算准确率
                _, predicted = torch.max(class_output, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_class_loss = val_class_loss / len(val_loader)
        val_attribute_loss = val_attribute_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 更新学习率
        scheduler.step()
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}: '  
                   f'Train Loss: {train_loss:.4f} (Class: {train_class_loss:.4f}, Attr: {train_attribute_loss:.4f}), '  
                   f'Train Acc: {train_acc:.2f}%, '  
                   f'Val Loss: {val_loss:.4f} (Class: {val_class_loss:.4f}, Attr: {val_attribute_loss:.4f}), '  
                   f'Val Acc: {val_acc:.2f}%')
        
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
        'model_type': 'mobilenet_v2_with_attributes',
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
    parser = argparse.ArgumentParser(description='带有属性标签的角色分类模型训练脚本')
    parser.add_argument('--data-dir', type=str, default='../data/downloaded_images', help='数据目录')
    parser.add_argument('--annotations-file', type=str, default='attribute_annotations.json', help='标注文件路径')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2', 
                       choices=['mobilenet_v2', 'efficientnet_b0', 'resnet18'],
                       help='模型类型')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--output-dir', type=str, default='models/arona_plana_with_attributes', help='输出目录')
    parser.add_argument('--config', type=str, default=None, help='属性配置文件路径')
    
    args = parser.parse_args()
    
    # 如果未指定配置文件，尝试使用默认路径
    if args.config is None:
        possible_configs = [
            '../config/character_attributes.json',
            '../../config/character_attributes.json',
            os.path.join(os.path.dirname(__file__), '..', 'config', 'character_attributes.json')
        ]
        for config_path in possible_configs:
            if os.path.exists(config_path):
                args.config = config_path
                break
    
    if args.config:
        logger.info(f"使用属性配置文件: {args.config}")
    
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
    
    # 创建数据集
    logger.info('加载数据集...')
    full_dataset = CharacterAttributeDataset(args.data_dir, args.annotations_file, transform=train_transform)
    
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
    num_attributes = 6  # 6个属性
    model = get_model_with_attributes(args.model_type, num_classes, num_attributes)
    model = model.to(device)
    
    logger.info(f'模型类别数: {num_classes}, 属性数: {num_attributes}')
    
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
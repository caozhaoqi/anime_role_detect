#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的模型训练脚本
"""

import os
import sys
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

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.core.classification.models import get_model, get_model_with_attributes

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_model')


class CharacterDataset(Dataset):
    """角色数据集类"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        classes = sorted(os.listdir(self.data_dir))
        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for file in os.listdir(class_dir):
                if file.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.images.append(os.path.join(class_name, file))
                    self.labels.append(idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CharacterAttributeDataset(Dataset):
    """带有属性标签的角色数据集类"""
    def __init__(self, data_dir, annotations_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.annotations = []
        self.class_to_idx = {}
        
        # 加载标注
        self._load_annotations(annotations_file)
    
    def _load_annotations(self, annotations_file):
        """加载标注"""
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # 构建类别映射
        classes = set()
        for item in annotations:
            classes.add(item['character'])
        
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(classes))}
        
        # 加载标注
        for item in annotations:
            image_path = os.path.join(self.data_dir, item['image_path'])
            if os.path.exists(image_path):
                self.annotations.append({
                    'image_path': image_path,
                    'character': item['character'],
                    'attribute_labels': item['attribute_labels']
                })
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        image = Image.open(item['image_path']).convert('RGB')
        class_label = self.class_to_idx[item['character']]
        attribute_labels = torch.tensor(item['attribute_labels'], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_label, attribute_labels


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, output_dir, dataset):
    """训练模型"""
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.float() / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.float() / len(val_loader.dataset)
        
        logger.info(f'Epoch {epoch+1}/{epochs}')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, 'model_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': dataset.class_to_idx,
                'model_type': args.model_type,
                'epoch': epoch+1,
                'val_acc': val_acc.item()
            }, model_path)
            logger.info(f'保存最佳模型到: {model_path}')
    
    return best_val_acc


def train_model_with_attributes(model, train_loader, val_loader, class_criterion, attribute_criterion, optimizer, device, epochs, output_dir, dataset):
    """训练带有属性预测的模型"""
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_class_loss = 0.0
        running_attribute_loss = 0.0
        running_corrects = 0
        
        for images, class_labels, attribute_labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
            images = images.to(device)
            class_labels = class_labels.to(device)
            attribute_labels = attribute_labels.to(device)
            
            optimizer.zero_grad()
            class_outputs, attribute_outputs = model(images)
            
            class_loss = class_criterion(class_outputs, class_labels)
            attribute_loss = attribute_criterion(attribute_outputs, attribute_labels)
            total_loss = class_loss + attribute_loss
            
            total_loss.backward()
            optimizer.step()
            
            running_class_loss += class_loss.item() * images.size(0)
            running_attribute_loss += attribute_loss.item() * images.size(0)
            _, preds = torch.max(class_outputs, 1)
            running_corrects += torch.sum(preds == class_labels.data)
        
        train_class_loss = running_class_loss / len(train_loader.dataset)
        train_attribute_loss = running_attribute_loss / len(train_loader.dataset)
        train_acc = running_corrects.float() / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_class_loss = 0.0
        val_attribute_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for images, class_labels, attribute_labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
                images = images.to(device)
                class_labels = class_labels.to(device)
                attribute_labels = attribute_labels.to(device)
                
                class_outputs, attribute_outputs = model(images)
                
                class_loss = class_criterion(class_outputs, class_labels)
                attribute_loss = attribute_criterion(attribute_outputs, attribute_labels)
                
                val_class_loss += class_loss.item() * images.size(0)
                val_attribute_loss += attribute_loss.item() * images.size(0)
                _, preds = torch.max(class_outputs, 1)
                val_corrects += torch.sum(preds == class_labels.data)
        
        val_class_loss = val_class_loss / len(val_loader.dataset)
        val_attribute_loss = val_attribute_loss / len(val_loader.dataset)
        val_acc = val_corrects.float() / len(val_loader.dataset)
        
        logger.info(f'Epoch {epoch+1}/{epochs}')
        logger.info(f'Train Class Loss: {train_class_loss:.4f}, Train Attribute Loss: {train_attribute_loss:.4f}, Train Acc: {train_acc:.4f}')
        logger.info(f'Val Class Loss: {val_class_loss:.4f}, Val Attribute Loss: {val_attribute_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, 'model_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': dataset.class_to_idx,
                'model_type': args.model_type,
                'epoch': epoch+1,
                'val_acc': val_acc.item()
            }, model_path)
            logger.info(f'保存最佳模型到: {model_path}')
    
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='统一的模型训练脚本')
    parser.add_argument('--data-dir', type=str, default='data/downloaded_images', help='数据目录')
    parser.add_argument('--annotations-file', type=str, default=None, help='属性标注文件路径')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2', 
                       choices=['mobilenet_v2', 'efficientnet_b0', 'resnet18'],
                       help='模型类型')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--output-dir', type=str, default='models/character_classifier', help='输出目录')
    parser.add_argument('--val-split', type=float, default=0.2, help='验证集比例')
    
    global args
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    if args.annotations_file:
        # 使用带有属性标签的数据集
        logger.info('加载带有属性标签的数据集...')
        dataset = CharacterAttributeDataset(args.data_dir, args.annotations_file, transform=transform)
        num_classes = len(dataset.class_to_idx)
        num_attributes = len(dataset.annotations[0]['attribute_labels'])
        logger.info(f'类别数: {num_classes}, 属性数: {num_attributes}')
        
        # 创建模型
        model = get_model_with_attributes(args.model_type, num_classes, num_attributes)
        model = model.to(device)
        
        # 损失函数和优化器
        class_criterion = nn.CrossEntropyLoss()
        attribute_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        # 使用普通数据集
        logger.info('加载普通数据集...')
        dataset = CharacterDataset(args.data_dir, transform=transform)
        num_classes = len(dataset.class_to_idx)
        logger.info(f'类别数: {num_classes}')
        
        # 创建模型
        model = get_model(args.model_type, num_classes)
        model = model.to(device)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 分割数据集
    train_size = int((1 - args.val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    logger.info(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}')
    
    # 训练模型
    start_time = time.time()
    if args.annotations_file:
        best_val_acc = train_model_with_attributes(
            model, train_loader, val_loader, 
            class_criterion, attribute_criterion, 
            optimizer, device, args.epochs, args.output_dir, dataset
        )
    else:
        best_val_acc = train_model(
            model, train_loader, val_loader, 
            criterion, optimizer, 
            device, args.epochs, args.output_dir, dataset
        )
    
    end_time = time.time()
    logger.info(f'训练完成！耗时: {end_time - start_time:.2f}秒')
    logger.info(f'最佳验证准确率: {best_val_acc:.4f}')


if __name__ == '__main__':
    main()
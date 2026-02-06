#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在线学习系统脚本

实现增量学习、动态分类器扩展和灾难性遗忘减少技术。
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import logging
from tqdm import tqdm
import json
import faiss
import numpy as np
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('online_learning_system')

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

class OnlineLearningSystem:
    """在线学习系统类"""
    
    def __init__(self, model_type, num_classes, feature_dim=1280, device='cpu'):
        """初始化在线学习系统
        
        Args:
            model_type: 模型类型
            num_classes: 初始类别数量
            feature_dim: 特征维度
            device: 设备
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        
        # 加载模型
        self.model = self._get_model(model_type, num_classes)
        self.model.to(device)
        
        # 创建特征提取器（去掉分类头）
        if model_type == 'efficientnet_b0':
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1], nn.AdaptiveAvgPool2d(1))
        elif model_type == 'mobilenet_v2':
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1], nn.AdaptiveAvgPool2d(1))
        elif model_type == 'resnet18':
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        else:
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        
        self.feature_extractor.to(device)
        
        # 创建FAISS索引
        self.index = faiss.IndexFlatL2(feature_dim)
        self.class_centers = {}  # 类别中心点
        self.class_samples = {}   # 每个类别的样本数
        
        # 优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        # 内存重放缓冲区
        self.replay_buffer = []
        self.buffer_size = 500
        
        logger.info(f"在线学习系统初始化完成，模型: {model_type}, 类别数: {num_classes}")
    
    def _get_model(self, model_type, num_classes):
        """获取模型
        
        Args:
            model_type: 模型类型
            num_classes: 类别数量
        
        Returns:
            模型
        """
        if model_type == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_type == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_type == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return model
    
    def extract_features(self, images):
        """提取特征
        
        Args:
            images: 图像批次
        
        Returns:
            特征向量
        """
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(images)
            features = features.view(features.size(0), -1)
        return features
    
    def update_classifier(self, new_classes):
        """动态扩展分类器
        
        Args:
            new_classes: 新类别列表
        """
        old_num_classes = self.num_classes
        new_num_classes = old_num_classes + len(new_classes)
        
        # 保存旧权重
        old_weights = self.model.classifier[1].weight.data if self.model_type in ['efficientnet_b0', 'mobilenet_v2'] else self.model.fc.weight.data
        old_bias = self.model.classifier[1].bias.data if self.model_type in ['efficientnet_b0', 'mobilenet_v2'] else self.model.fc.bias.data
        
        # 创建新的分类器
        if self.model_type == 'efficientnet_b0':
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, new_num_classes)
        elif self.model_type == 'mobilenet_v2':
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, new_num_classes)
        elif self.model_type == 'resnet18':
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, new_num_classes)
        
        # 复制旧权重到新分类器
        with torch.no_grad():
            if self.model_type in ['efficientnet_b0', 'mobilenet_v2']:
                self.model.classifier[1].weight.data[:old_num_classes] = old_weights
                self.model.classifier[1].bias.data[:old_num_classes] = old_bias
            else:
                self.model.fc.weight.data[:old_num_classes] = old_weights
                self.model.fc.bias.data[:old_num_classes] = old_bias
        
        self.num_classes = new_num_classes
        logger.info(f"分类器已扩展，新类别数: {new_num_classes}")
    
    def add_to_replay_buffer(self, images, labels):
        """添加到内存重放缓冲区
        
        Args:
            images: 图像批次
            labels: 标签批次
        """
        for i in range(images.size(0)):
            self.replay_buffer.append((images[i].cpu(), labels[i].cpu()))
            
        # 保持缓冲区大小
        if len(self.replay_buffer) > self.buffer_size:
            # 随机删除旧样本
            indices = np.random.choice(len(self.replay_buffer), size=self.buffer_size, replace=False)
            self.replay_buffer = [self.replay_buffer[i] for i in indices]
    
    def train_batch(self, images, labels, use_replay=True):
        """训练单个批次
        
        Args:
            images: 图像批次
            labels: 标签批次
            use_replay: 是否使用内存重放
        
        Returns:
            损失值
        """
        self.model.train()
        
        # 合并新样本和重放缓冲区样本
        if use_replay and len(self.replay_buffer) > 0:
            # 从缓冲区中采样
            replay_size = min(len(self.replay_buffer), images.size(0))
            replay_indices = np.random.choice(len(self.replay_buffer), size=replay_size, replace=False)
            replay_images = torch.stack([self.replay_buffer[i][0] for i in replay_indices])
            replay_labels = torch.tensor([self.replay_buffer[i][1] for i in replay_indices])
            
            # 合并批次
            combined_images = torch.cat([images, replay_images.to(self.device)], dim=0)
            combined_labels = torch.cat([labels, replay_labels.to(self.device)], dim=0)
        else:
            combined_images = images
            combined_labels = labels
        
        # 前向传播
        outputs = self.model(combined_images)
        loss = self.criterion(outputs, combined_labels)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新内存重放缓冲区
        self.add_to_replay_buffer(images, labels)
        
        return loss.item()
    
    def update_class_centers(self, features, labels):
        """更新类别中心点
        
        Args:
            features: 特征向量
            labels: 标签
        """
        features = features.cpu().numpy()
        labels = labels.cpu().numpy()
        
        for i in range(len(labels)):
            label = labels[i]
            feature = features[i]
            
            if label not in self.class_centers:
                self.class_centers[label] = feature
                self.class_samples[label] = 1
            else:
                # 增量更新中心点
                self.class_centers[label] = (self.class_centers[label] * self.class_samples[label] + feature) / (self.class_samples[label] + 1)
                self.class_samples[label] += 1
    
    def incremental_learn(self, data_loader, num_epochs=1):
        """增量学习
        
        Args:
            data_loader: 数据加载器
            num_epochs: 训练轮数
        """
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(data_loader, desc=f"增量学习 Epoch {epoch+1}")
            
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 提取特征
                features = self.extract_features(images)
                
                # 更新类别中心点
                self.update_class_centers(features, labels)
                
                # 训练批次
                loss = self.train_batch(images, labels)
                total_loss += loss
                
                # 计算准确率
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    'loss': loss,
                    'acc': (correct / total) * 100
                })
            
            epoch_loss = total_loss / len(data_loader)
            epoch_accuracy = (correct / total) * 100
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    
    def save(self, save_path):
        """保存模型
        
        Args:
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型状态
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_classes': self.num_classes,
            'model_type': self.model_type,
            'feature_dim': self.feature_dim,
            'class_centers': {k: v.tolist() for k, v in self.class_centers.items()},
            'class_samples': self.class_samples
        }
        
        torch.save(checkpoint, save_path)
        
        # 保存FAISS索引
        index_path = os.path.join(os.path.dirname(save_path), 'faiss_index.bin')
        faiss.write_index(self.index, index_path)
        
        logger.info(f"模型已保存到: {save_path}")
        logger.info(f"FAISS索引已保存到: {index_path}")
    
    def load(self, load_path):
        """加载模型
        
        Args:
            load_path: 加载路径
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_classes = checkpoint['num_classes']
        self.class_centers = {k: np.array(v) for k, v in checkpoint['class_centers'].items()}
        self.class_samples = checkpoint['class_samples']
        
        # 加载FAISS索引
        index_path = os.path.join(os.path.dirname(load_path), 'faiss_index.bin')
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        logger.info(f"模型已加载从: {load_path}")

def main():
    parser = argparse.ArgumentParser(description='在线学习系统脚本')
    
    parser.add_argument('--data-dir', type=str, default='data/train',
                       help='训练数据目录')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2',
                       choices=['efficientnet_b0', 'mobilenet_v2', 'resnet18'],
                       help='模型类型')
    parser.add_argument('--output-dir', type=str, default='models/online_learning',
                       help='输出目录')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批量大小')
    parser.add_argument('--num-epochs', type=int, default=5,
                       help='训练轮数')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载器工作线程数')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    dataset = CharacterDataset(root_dir=args.data_dir, transform=transform)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    
    num_classes = len(dataset.class_to_idx)
    
    # 创建在线学习系统
    online_system = OnlineLearningSystem(
        model_type=args.model_type,
        num_classes=num_classes,
        device=device
    )
    
    # 开始增量学习
    logger.info(f"开始增量学习，数据量: {len(dataset)}, 类别数: {num_classes}")
    online_system.incremental_learn(data_loader, num_epochs=args.num_epochs)
    
    # 保存模型
    output_path = os.path.join(args.output_dir, f'{args.model_type}_online.pth')
    online_system.save(output_path)
    
    # 评估模型
    logger.info("评估在线学习后的模型性能...")
    online_system.model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = online_system.model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = (correct / total) * 100
    logger.info(f"在线学习后模型准确率: {accuracy:.2f}%")
    logger.info("在线学习完成")

if __name__ == '__main__':
    main()

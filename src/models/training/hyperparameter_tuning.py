#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数调优脚本

使用网格搜索或随机搜索来优化模型性能
"""

import os
import argparse
import logging
import json
import random
from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hyperparameter_tuning')


class HyperparameterTuner:
    def __init__(self, train_dir, val_dir, output_dir='tuning_results'):
        """
        初始化超参数调优器
        
        Args:
            train_dir: 训练集目录
            val_dir: 验证集目录
            output_dir: 输出目录
        """
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 检测设备
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载数据集以获取类别数量
        train_dataset = datasets.ImageFolder(train_dir, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]))
        self.num_classes = len(train_dataset.classes)
        logger.info(f"类别数量: {self.num_classes}")
        
    def create_model(self, dropout_rate=0.3):
        """
        创建模型
        
        Args:
            dropout_rate: Dropout比率
            
        Returns:
            模型
        """
        class CustomEfficientNet(nn.Module):
            def __init__(self, num_classes, dropout_rate):
                super().__init__()
                self.backbone = models.efficientnet_b0(pretrained=False)
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                return self.backbone(x)
        
        model = CustomEfficientNet(self.num_classes, dropout_rate)
        return model
    
    def get_data_loaders(self, batch_size=32):
        """
        获取数据加载器
        
        Args:
            batch_size: 批量大小
            
        Returns:
            训练和验证数据加载器
        """
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
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
        
        train_dataset = datasets.ImageFolder(self.train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(self.val_dir, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, criterion, optimizer, device):
        """
        训练一个epoch
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            criterion: 损失函数
            optimizer: 优化器
            device: 设备
            
        Returns:
            平均损失和准确率
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc='训练', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, model, val_loader, criterion, device):
        """
        验证模型
        
        Args:
            model: 模型
            val_loader: 验证数据加载器
            criterion: 损失函数
            device: 设备
            
        Returns:
            平均损失和准确率
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='验证', leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train_with_hyperparameters(self, hyperparams, num_epochs=20):
        """
        使用指定超参数训练模型
        
        Args:
            hyperparams: 超参数字典
            num_epochs: 训练轮数
            
        Returns:
            训练结果
        """
        logger.info(f"开始训练，超参数: {hyperparams}")
        
        # 创建模型
        model = self.create_model(dropout_rate=hyperparams['dropout_rate'])
        model.to(self.device)
        
        # 获取数据加载器
        train_loader, val_loader = self.get_data_loaders(batch_size=hyperparams['batch_size'])
        
        # 定义损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=hyperparams['label_smoothing'])
        
        # 定义优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=hyperparams['learning_rate'],
            weight_decay=hyperparams['weight_decay']
        )
        
        # 定义学习率调度器
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=hyperparams['learning_rate'] * 1e-4
        )
        
        # 训练循环
        best_val_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, self.device)
            
            # 验证
            val_loss, val_acc = self.validate(model, val_loader, criterion, self.device)
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # 更新最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        result = {
            'hyperparams': hyperparams,
            'best_val_acc': best_val_acc,
            'history': history
        }
        
        return result
    
    def grid_search(self, param_grid, num_epochs=20):
        """
        网格搜索
        
        Args:
            param_grid: 参数网格
            num_epochs: 每组参数的训练轮数
            
        Returns:
            最佳超参数和结果
        """
        logger.info("开始网格搜索")
        
        # 生成所有参数组合
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(product(*values))
        
        logger.info(f"总共有 {len(combinations)} 种参数组合")
        
        results = []
        for i, combination in enumerate(combinations):
            hyperparams = dict(zip(keys, combination))
            logger.info(f"测试组合 {i+1}/{len(combinations)}")
            
            try:
                result = self.train_with_hyperparameters(hyperparams, num_epochs)
                results.append(result)
                
                # 保存结果
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_file = os.path.join(self.output_dir, f'result_{timestamp}_{i+1}.json')
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"结果已保存: {result_file}")
                
            except Exception as e:
                logger.error(f"训练失败: {str(e)}")
                continue
        
        # 找出最佳结果
        best_result = max(results, key=lambda x: x['best_val_acc'])
        logger.info(f"最佳超参数: {best_result['hyperparams']}")
        logger.info(f"最佳验证准确率: {best_result['best_val_acc']:.2f}%")
        
        # 保存最佳结果
        best_file = os.path.join(self.output_dir, 'best_result.json')
        with open(best_file, 'w') as f:
            json.dump(best_result, f, indent=2)
        
        return best_result
    
    def random_search(self, param_ranges, n_trials=10, num_epochs=20):
        """
        随机搜索
        
        Args:
            param_ranges: 参数范围
            n_trials: 试验次数
            num_epochs: 每组参数的训练轮数
            
        Returns:
            最佳超参数和结果
        """
        logger.info(f"开始随机搜索，共 {n_trials} 次试验")
        
        results = []
        for i in range(n_trials):
            # 随机选择参数
            hyperparams = {}
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range, list):
                    hyperparams[param_name] = random.choice(param_range)
                elif isinstance(param_range, tuple):
                    hyperparams[param_name] = random.uniform(param_range[0], param_range[1])
                else:
                    hyperparams[param_name] = param_range
            
            logger.info(f"试验 {i+1}/{n_trials}, 超参数: {hyperparams}")
            
            try:
                result = self.train_with_hyperparameters(hyperparams, num_epochs)
                results.append(result)
                
                # 保存结果
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_file = os.path.join(self.output_dir, f'result_{timestamp}_{i+1}.json')
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"结果已保存: {result_file}")
                
            except Exception as e:
                logger.error(f"训练失败: {str(e)}")
                continue
        
        # 找出最佳结果
        best_result = max(results, key=lambda x: x['best_val_acc'])
        logger.info(f"最佳超参数: {best_result['hyperparams']}")
        logger.info(f"最佳验证准确率: {best_result['best_val_acc']:.2f}%")
        
        # 保存最佳结果
        best_file = os.path.join(self.output_dir, 'best_result.json')
        with open(best_file, 'w') as f:
            json.dump(best_result, f, indent=2)
        
        return best_result


def main():
    parser = argparse.ArgumentParser(description='超参数调优脚本')
    
    parser.add_argument('--train-dir', type=str, 
                       default='data/split_dataset/train',
                       help='训练集目录')
    parser.add_argument('--val-dir', type=str, 
                       default='data/split_dataset/val',
                       help='验证集目录')
    parser.add_argument('--output-dir', type=str, 
                       default='tuning_results',
                       help='输出目录')
    parser.add_argument('--method', type=str, 
                       choices=['grid', 'random'],
                       default='random',
                       help='搜索方法')
    parser.add_argument('--num-epochs', type=int, 
                       default=20,
                       help='每组参数的训练轮数')
    parser.add_argument('--n-trials', type=int, 
                       default=10,
                       help='随机搜索的试验次数')
    
    args = parser.parse_args()
    
    # 初始化调优器
    tuner = HyperparameterTuner(args.train_dir, args.val_dir, args.output_dir)
    
    # 定义参数网格/范围
    if args.method == 'grid':
        param_grid = {
            'learning_rate': [1e-4, 3e-4, 5e-4],
            'batch_size': [16, 32, 64],
            'weight_decay': [1e-3, 1e-4, 1e-5],
            'dropout_rate': [0.2, 0.3, 0.4],
            'label_smoothing': [0.0, 0.1, 0.2]
        }
        best_result = tuner.grid_search(param_grid, args.num_epochs)
    else:  # random
        param_ranges = {
            'learning_rate': (1e-5, 1e-3),
            'batch_size': [16, 32, 64],
            'weight_decay': (1e-5, 1e-2),
            'dropout_rate': (0.1, 0.5),
            'label_smoothing': [0.0, 0.1, 0.2]
        }
        best_result = tuner.random_search(param_ranges, args.n_trials, args.num_epochs)
    
    logger.info("超参数调优完成！")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用扩充数据集训练模型脚本

使用扩充后的数据集训练不同架构的模型，提高模型性能。
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
import matplotlib.pyplot as plt
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_with_augmented_data')

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

def get_model(model_type, num_classes):
    """获取模型
    
    Args:
        model_type: 模型类型
        num_classes: 类别数量
    
    Returns:
        模型
    """
    if model_type == 'efficientnet_b0':
        logger.info("加载模型: EfficientNet-B0")
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'mobilenet_v2':
        logger.info("加载模型: MobileNetV2")
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'shufflenet_v2':
        logger.info("加载模型: ShuffleNetV2")
        model = models.shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'squeezenet':
        logger.info("加载模型: SqueezeNet")
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    elif model_type == 'resnet18':
        logger.info("加载模型: ResNet18")
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model

def calculate_model_size(model):
    """计算模型大小
    
    Args:
        model: 模型
    
    Returns:
        模型大小（MB）, 参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    model_size = total_params * 4 / (1024 * 1024)  # 每个参数4字节
    return model_size, total_params

def train_model(args):
    """训练模型
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    dataset = CharacterDataset(root_dir=args.data_dir, transform=train_transform)
    
    # 分割数据集为训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 为验证集应用不同的变换
    val_dataset.dataset.transform = val_transform
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    num_classes = len(dataset.class_to_idx)
    logger.info(f"类别数量: {num_classes}")
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存类别映射
    with open(os.path.join(output_dir, 'class_to_idx.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset.class_to_idx, f, ensure_ascii=False, indent=2)
    
    # 加载模型
    model = get_model(args.model_type, num_classes)
    model.to(device)
    
    # 计算模型大小
    model_size, num_params = calculate_model_size(model)
    logger.info(f"模型大小: {model_size:.2f} MB, 参数数量: {num_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # 训练指标
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # 最佳模型指标
    best_val_accuracy = 0.0
    
    # 训练开始时间
    start_time = time.time()
    
    # 训练循环
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch [{epoch+1}/{args.num_epochs}]")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}")
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算训练指标
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': (train_correct / train_total) * 100
            })
        
        # 计算训练 epoch 指标
        train_epoch_loss = train_loss / train_total
        train_epoch_accuracy = (train_correct / train_total) * 100
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        logger.info(f"训练 Loss: {train_epoch_loss:.4f}, Accuracy: {train_epoch_accuracy:.2f}%")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"验证 Epoch {epoch+1}")
            
            for images, labels in progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                
                # 模型推理
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # 计算验证指标
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 更新进度条
                current_acc = (val_correct / val_total) * 100 if val_total > 0 else 0
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': current_acc
                })
        
        # 计算验证 epoch 指标
        if val_total > 0:
            val_epoch_loss = val_loss / val_total
            val_epoch_accuracy = (val_correct / val_total) * 100
        else:
            val_epoch_loss = 0
            val_epoch_accuracy = 0
        
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        logger.info(f"验证 Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.2f}%")
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            best_model_path = os.path.join(output_dir, 'model_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_accuracy,
                'model_type': args.model_type,
                'class_to_idx': dataset.class_to_idx
            }, best_model_path)
            logger.info(f"保存最佳模型到: {best_model_path}")
    
    # 训练结束时间
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"训练总时间: {training_time:.2f} 秒")
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, 'model_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': args.model_type,
        'class_to_idx': dataset.class_to_idx
    }, final_model_path)
    logger.info(f"保存最终模型到: {final_model_path}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 6))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='训练准确率')
    plt.plot(val_accuracies, label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # 保存训练曲线
    plt.tight_layout()
    training_curve_path = os.path.join(output_dir, 'training_curve.png')
    plt.savefig(training_curve_path)
    logger.info(f"训练曲线已保存到: {training_curve_path}")
    
    # 保存训练结果
    training_results = {
        'model_type': args.model_type,
        'model_size_mb': model_size,
        'num_params': num_params,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'training_time_seconds': training_time,
        'best_val_accuracy': best_val_accuracy,
        'final_train_accuracy': train_accuracies[-1],
        'final_val_accuracy': val_accuracies[-1],
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
    
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(training_results, f, ensure_ascii=False, indent=2)
    logger.info(f"训练结果已保存到: {results_path}")
    
    logger.info("模型训练完成")
    logger.info(f"最佳验证准确率: {best_val_accuracy:.2f}%")
    
    return training_results

def main():
    parser = argparse.ArgumentParser(description='使用扩充数据集训练模型脚本')
    
    parser.add_argument('--data-dir', type=str, default='data/train',
                       help='训练数据目录')
    parser.add_argument('--output-dir', type=str, default='models/augmented_training',
                       help='输出目录')
    parser.add_argument('--model-type', type=str, default='efficientnet_b0',
                       choices=['efficientnet_b0', 'mobilenet_v2', 'shufflenet_v2', 'squeezenet', 'resnet18'],
                       help='模型类型')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批量大小')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载器工作线程数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始训练
    train_model(args)

if __name__ == '__main__':
    main()

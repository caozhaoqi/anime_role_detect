#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于检测模型的图片生成模型训练脚本
使用扩散模型架构，结合检测模型的特征来生成角色图片
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
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_generator')


class CharacterDataset(Dataset):
    """角色数据集类"""
    
    def __init__(self, root_dir, transform=None, target_characters=None):
        """初始化数据集
        
        Args:
            root_dir: 数据目录
            transform: 数据变换
            target_characters: 目标角色列表
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


class FeatureExtractor(nn.Module):
    """特征提取器，基于现有的检测模型"""
    
    def __init__(self, model_type='mobilenet_v2', num_classes=2):
        super().__init__()
        if model_type == 'mobilenet_v2':
            self.base_model = models.mobilenet_v2(pretrained=True)
            # 移除分类头，保留特征提取部分
            self.feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        elif model_type == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(pretrained=True)
            self.feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        elif model_type == 'resnet18':
            self.base_model = models.resnet18(pretrained=True)
            self.feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def forward(self, x):
        return self.base_model(x)


class DiffusionGenerator(nn.Module):
    """扩散模型生成器"""
    
    def __init__(self, feature_dim, image_size=(224, 224), num_channels=3):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        
        # 特征投影层
        self.feature_proj = nn.Linear(feature_dim, 256)
        
        # 扩散模型的UNet架构
        self.unet = nn.Sequential(
            # 下采样
            nn.Conv2d(num_channels * 2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 瓶颈层，融合特征
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # 上采样
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出0-1范围的图像
        )
    
    def forward(self, x, noise, features):
        # 融合噪声和特征
        batch_size = x.size(0)
        
        # 前向传播
        return self.unet(torch.cat([x, noise], dim=1))


def diffusion_loss(generator, feature_extractor, images, labels, device):
    """扩散模型损失函数"""
    batch_size = images.size(0)
    
    # 生成随机噪声
    noise = torch.randn_like(images).to(device)
    
    # 提取特征
    with torch.no_grad():
        features = feature_extractor(images)
    
    # 生成图像
    generated = generator(images, noise, features)
    
    # 计算损失
    loss = nn.MSELoss()(generated, images)
    
    # 特征匹配损失
    with torch.no_grad():
        gen_features = feature_extractor(generated)
    feature_loss = nn.MSELoss()(gen_features, features)
    
    return loss + 0.1 * feature_loss


def train_generator(generator, feature_extractor, train_loader, device, num_epochs=50, lr=0.0001, output_dir='models/generator'):
    """训练生成模型"""
    logger.info(f"开始训练生成模型，设备: {device}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 优化器
    optimizer = optim.Adam(generator.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练记录
    losses = []
    best_loss = float('inf')
    
    # 早停设置
    patience = 10
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        generator.train()
        feature_extractor.eval()  # 特征提取器保持冻结
        
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images = images.to(device)
            
            optimizer.zero_grad()
            loss = diffusion_loss(generator, feature_extractor, images, labels, device)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        scheduler.step()
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}: Loss: {avg_loss:.4f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'feature_extractor_state_dict': feature_extractor.state_dict(),
                'loss': best_loss
            }, os.path.join(output_dir, 'generator_best.pth'))
            logger.info(f'保存最佳模型，损失: {best_loss:.4f}')
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # 早停检查
        if no_improve_count >= patience:
            logger.info(f'早停: {patience} 个epoch没有改进')
            break
    
    # 保存训练结果
    results = {
        'num_epochs': num_epochs,
        'best_loss': best_loss,
        'final_loss': losses[-1],
        'losses': losses
    }
    
    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f'训练完成，最佳损失: {best_loss:.4f}')
    return results


def main():
    parser = argparse.ArgumentParser(description='基于检测模型的图片生成模型训练脚本')
    parser.add_argument('--data-dir', type=str, default='data/train', help='数据目录')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2', 
                       choices=['mobilenet_v2', 'efficientnet_b0', 'resnet18'],
                       help='特征提取器模型类型')
    parser.add_argument('--batch-size', type=int, default=16, help='批量大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--output-dir', type=str, default='models/generator', help='输出目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 目标角色
    target_characters = ['蔚蓝档案_阿罗娜', '蔚蓝档案_普拉娜']
    
    # 创建数据集
    logger.info('加载数据集...')
    dataset = CharacterDataset(args.data_dir, transform=transform, target_characters=target_characters)
    
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    logger.info(f'训练集大小: {len(dataset)}')
    
    # 创建模型
    feature_extractor = FeatureExtractor(args.model_type)
    feature_extractor = feature_extractor.to(device)
    
    generator = DiffusionGenerator(feature_extractor.feature_dim)
    generator = generator.to(device)
    
    # 冻结特征提取器
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    # 训练模型
    results = train_generator(
        generator, feature_extractor, train_loader, device,
        num_epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir
    )
    
    logger.info('训练完成！')
    logger.info(f'最佳损失: {results["best_loss"]:.4f}')


if __name__ == '__main__':
    main()

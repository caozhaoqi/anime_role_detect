#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用条件生成模型创建角色图片
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import logging
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('generate_conditional')


class ConditionalGenerator(nn.Module):
    """条件生成器 - 根据类别标签生成图像"""
    
    def __init__(self, num_classes, latent_dim=100, image_size=(224, 224), num_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.num_channels = num_channels
        
        # 类别嵌入层
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # 生成器网络
        self.generator = nn.Sequential(
            # 输入: latent_dim (噪声) + latent_dim (类别嵌入)
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            
            # 转换为图像尺寸
            nn.Linear(2048, image_size[0] * image_size[1] * num_channels),
            nn.Tanh()  # 输出-1到1的范围
        )
    
    def forward(self, noise, labels):
        # 嵌入类别标签
        label_emb = self.label_embedding(labels)
        
        # 拼接噪声和类别嵌入
        combined = torch.cat([noise, label_emb], dim=1)
        
        # 生成图像
        generated = self.generator(combined)
        
        # 重塑为图像格式
        generated = generated.view(-1, self.num_channels, self.image_size[0], self.image_size[1])
        
        return generated


class ConditionalDiscriminator(nn.Module):
    """条件判别器 - 判断图像是否真实且符合给定类别"""
    
    def __init__(self, num_classes, image_size=(224, 224), num_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_channels = num_channels
        
        # 类别嵌入层
        self.label_embedding = nn.Embedding(num_classes, image_size[0] * image_size[1])
        
        # 判别器网络
        self.discriminator = nn.Sequential(
            # 输入: 图像 + 类别嵌入
            nn.Conv2d(num_channels + 1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()  # 输出0-1的概率
        )
    
    def forward(self, images, labels):
        # 嵌入类别标签并重塑
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.view(-1, 1, self.image_size[0], self.image_size[1])
        
        # 拼接图像和类别嵌入
        combined = torch.cat([images, label_emb], dim=1)
        
        # 判别
        validity = self.discriminator(combined)
        
        return validity


class FeatureExtractor(nn.Module):
    """特征提取器，用于分类任务"""
    
    def __init__(self, model_type='mobilenet_v2', num_classes=2):
        super().__init__()
        if model_type == 'mobilenet_v2':
            self.base_model = models.mobilenet_v2(pretrained=True)
            self.feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(self.feature_dim, num_classes)
        elif model_type == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(pretrained=True)
            self.feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(self.feature_dim, num_classes)
        elif model_type == 'resnet18':
            self.base_model = models.resnet18(pretrained=True)
            self.feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(self.feature_dim, num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def forward(self, x):
        return self.base_model(x)


class ConditionalGAN(nn.Module):
    """条件GAN模型 - 结合分类和生成功能"""
    
    def __init__(self, num_classes, latent_dim=100, image_size=(224, 224), num_channels=3, model_type='mobilenet_v2'):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # 初始化各个组件
        self.generator = ConditionalGenerator(num_classes, latent_dim, image_size, num_channels)
        self.discriminator = ConditionalDiscriminator(num_classes, image_size, num_channels)
        self.feature_extractor = FeatureExtractor(model_type, num_classes)
    
    def forward(self, noise, labels, mode='generate'):
        """前向传播
        
        Args:
            noise: 输入噪声
            labels: 类别标签
            mode: 'generate' 生成图像, 'discriminate' 判别图像, 'classify' 分类图像
        """
        if mode == 'generate':
            return self.generator(noise, labels)
        elif mode == 'discriminate':
            return self.discriminator(noise, labels)
        elif mode == 'classify':
            return self.feature_extractor(noise)
        else:
            raise ValueError(f"不支持的mode: {mode}")


def load_model(model_path, num_classes, latent_dim=100, model_type='mobilenet_v2', device='mps'):
    """加载训练好的条件GAN模型
    
    Args:
        model_path: 模型路径
        num_classes: 类别数量
        latent_dim: 潜在空间维度
        model_type: 模型类型
        device: 设备
    """
    # 创建模型
    model = ConditionalGAN(num_classes, latent_dim, model_type=model_type)
    model = model.to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    model.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
    
    logger.info(f"模型加载完成: {model_path}")
    return model


def generate_conditional_images(model, num_images_per_class=5, output_dir='data/conditional_generated', device='mps'):
    """使用条件生成模型创建角色图片
    
    Args:
        model: 条件GAN模型
        num_images_per_class: 每个类别的生成图片数量
        output_dir: 输出目录
        device: 设备
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.generator.eval()
    
    logger.info(f"开始生成图片，每个类别 {num_images_per_class} 张")
    
    for class_idx in range(model.num_classes):
        class_dir = os.path.join(output_dir, f'class_{class_idx}')
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(num_images_per_class):
            try:
                # 生成随机噪声
                noise = torch.randn(1, model.latent_dim).to(device)
                
                # 创建类别标签
                labels = torch.LongTensor([class_idx]).to(device)
                
                # 生成图像
                with torch.no_grad():
                    generated = model(noise, labels, mode='generate')
                
                # 转换为PIL图像
                generated = generated.squeeze().cpu().permute(1, 2, 0).numpy()
                # 从-1到1转换到0到255
                generated = ((generated + 1) * 127.5).astype(np.uint8)
                image = Image.fromarray(generated)
                
                # 保存图像
                output_path = os.path.join(class_dir, f'generated_{i+1}.png')
                image.save(output_path)
                logger.info(f"生成图片: {output_path}")
                
            except Exception as e:
                logger.error(f"生成图片时出错: {e}")
                continue
    
    logger.info("图片生成完成")


def main():
    parser = argparse.ArgumentParser(description='使用条件生成模型创建角色图片')
    parser.add_argument('--model-path', type=str, default='models/conditional_gan/model_final.pth', help='模型路径')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2', 
                       choices=['mobilenet_v2', 'efficientnet_b0', 'resnet18'],
                       help='特征提取器模型类型')
    parser.add_argument('--num-classes', type=int, default=2, help='类别数量')
    parser.add_argument('--latent-dim', type=int, default=100, help='潜在空间维度')
    parser.add_argument('--num-images-per-class', type=int, default=5, help='每个类别的生成图片数量')
    parser.add_argument('--output-dir', type=str, default='data/conditional_generated', help='输出目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 加载模型
    model = load_model(args.model_path, args.num_classes, args.latent_dim, args.model_type, device)
    
    # 生成图片
    generate_conditional_images(model, args.num_images_per_class, args.output_dir, device)


if __name__ == '__main__':
    main()

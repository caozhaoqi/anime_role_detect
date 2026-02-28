#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的生成模型创建角色图片
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
logger = logging.getLogger('generate_with_model')


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
        # 前向传播
        return self.unet(torch.cat([x, noise], dim=1))


def generate_images(generator, feature_extractor, num_images=5, output_dir='data/generated_model', device='mps'):
    """使用生成模型创建角色图片
    
    Args:
        generator: 生成模型
        feature_extractor: 特征提取器
        num_images: 生成的图片数量
        output_dir: 输出目录
        device: 设备
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    generator.eval()
    feature_extractor.eval()
    
    logger.info(f"开始生成图片，数量: {num_images}")
    
    for i in range(num_images):
        try:
            # 生成随机噪声作为输入
            noise = torch.randn(1, 3, 224, 224).to(device)
            
            # 生成随机特征（模拟从检测模型提取的特征）
            random_features = torch.randn(1, feature_extractor.feature_dim).to(device)
            
            # 生成图像
            with torch.no_grad():
                generated = generator(noise, noise, random_features)
            
            # 转换为PIL图像
            generated = generated.squeeze().cpu().permute(1, 2, 0).numpy()
            generated = (generated * 255).astype(np.uint8)
            image = Image.fromarray(generated)
            
            # 保存图像
            output_path = os.path.join(output_dir, f'generated_{i+1}.png')
            image.save(output_path)
            logger.info(f"生成图片: {output_path}")
            
        except Exception as e:
            logger.error(f"生成图片时出错: {e}")
            continue
    
    logger.info("图片生成完成")


def load_model(model_path, model_type='mobilenet_v2', device='mps'):
    """加载训练好的模型
    
    Args:
        model_path: 模型路径
        model_type: 模型类型
        device: 设备
    """
    # 创建特征提取器
    feature_extractor = FeatureExtractor(model_type)
    feature_extractor = feature_extractor.to(device)
    
    # 创建生成器
    generator = DiffusionGenerator(feature_extractor.feature_dim)
    generator = generator.to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
    
    logger.info(f"模型加载完成: {model_path}")
    return generator, feature_extractor


def main():
    parser = argparse.ArgumentParser(description='使用训练好的生成模型创建角色图片')
    parser.add_argument('--model-path', type=str, default='models/generator/generator_best.pth', help='模型路径')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2', 
                       choices=['mobilenet_v2', 'efficientnet_b0', 'resnet18'],
                       help='特征提取器模型类型')
    parser.add_argument('--num-images', type=int, default=5, help='生成的图片数量')
    parser.add_argument('--output-dir', type=str, default='data/generated_model', help='输出目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 加载模型
    generator, feature_extractor = load_model(args.model_path, args.model_type, device)
    
    # 生成图片
    generate_images(generator, feature_extractor, args.num_images, args.output_dir, device)


if __name__ == '__main__':
    main()

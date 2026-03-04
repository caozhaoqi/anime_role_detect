#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估融合模型性能
包括分类准确率评估和生成质量评估
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import logging
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('evaluate_model')


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


class ConditionalGenerator(nn.Module):
    """条件生成器"""
    
    def __init__(self, num_classes, latent_dim=100, image_size=(224, 224), num_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.num_channels = num_channels
        
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        self.generator = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, image_size[0] * image_size[1] * num_channels),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels)
        combined = torch.cat([noise, label_emb], dim=1)
        generated = self.generator(combined)
        generated = generated.view(-1, self.num_channels, self.image_size[0], self.image_size[1])
        return generated


class ConditionalDiscriminator(nn.Module):
    """条件判别器"""
    
    def __init__(self, num_classes, image_size=(224, 224), num_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_channels = num_channels
        
        self.label_embedding = nn.Embedding(num_classes, image_size[0] * image_size[1])
        
        self.discriminator = nn.Sequential(
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
            nn.Sigmoid()
        )
    
    def forward(self, images, labels):
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.view(-1, 1, self.image_size[0], self.image_size[1])
        combined = torch.cat([images, label_emb], dim=1)
        validity = self.discriminator(combined)
        return validity


class FeatureExtractor(nn.Module):
    """特征提取器"""
    
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
    """条件GAN模型"""
    
    def __init__(self, num_classes, latent_dim=100, image_size=(224, 224), num_channels=3, model_type='mobilenet_v2'):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        self.generator = ConditionalGenerator(num_classes, latent_dim, image_size, num_channels)
        self.discriminator = ConditionalDiscriminator(num_classes, image_size, num_channels)
        self.feature_extractor = FeatureExtractor(model_type, num_classes)
    
    def forward(self, noise, labels, mode='generate'):
        if mode == 'generate':
            return self.generator(noise, labels)
        elif mode == 'discriminate':
            return self.discriminator(noise, labels)
        elif mode == 'classify':
            return self.feature_extractor(noise)
        else:
            raise ValueError(f"不支持的mode: {mode}")


def evaluate_classification(model, test_loader, device, class_names):
    """评估分类性能"""
    logger.info("开始评估分类性能...")
    
    model.feature_extractor.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='分类评估'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images, labels, mode='classify')
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_predictions)
    logger.info(f"分类准确率: {accuracy * 100:.2f}%")
    
    # 生成分类报告
    report = classification_report(all_labels, all_predictions, 
                               target_names=class_names, 
                               output_dict=True)
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }


def evaluate_generation_quality(model, num_samples=10, device='mps'):
    """评估生成质量"""
    logger.info("开始评估生成质量...")
    
    model.generator.eval()
    
    # 生成样本
    generated_samples = []
    for class_idx in range(model.num_classes):
        for _ in range(num_samples):
            noise = torch.randn(1, model.latent_dim).to(device)
            labels = torch.LongTensor([class_idx]).to(device)
            
            with torch.no_grad():
                generated = model(noise, labels, mode='generate')
            
            generated_samples.append(generated.squeeze().cpu().numpy())
    
    # 计算生成图像的统计特性
    generated_samples = np.array(generated_samples)
    mean_intensity = np.mean(generated_samples)
    std_intensity = np.std(generated_samples)
    
    logger.info(f"生成图像统计特性:")
    logger.info(f"  平均强度: {mean_intensity:.4f}")
    logger.info(f"  标准差: {std_intensity:.4f}")
    
    return {
        'num_samples': len(generated_samples),
        'mean_intensity': float(mean_intensity),
        'std_intensity': float(std_intensity)
    }


def evaluate_discriminator(model, test_loader, device):
    """评估判别器性能"""
    logger.info("开始评估判别器性能...")
    
    model.discriminator.eval()
    real_scores = []
    fake_scores = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='判别器评估'):
            images = images.to(device)
            labels = labels.to(device)
            
            # 真实图像的判别分数
            real_validity = model(images, labels, mode='discriminate')
            real_scores.extend(real_validity.cpu().numpy())
            
            # 生成虚假图像
            noise = torch.randn(images.size(0), model.latent_dim).to(device)
            fake_images = model(noise, labels, mode='generate')
            fake_validity = model(fake_images, labels, mode='discriminate')
            fake_scores.extend(fake_validity.cpu().numpy())
    
    real_scores = np.array(real_scores)
    fake_scores = np.array(fake_scores)
    
    logger.info(f"判别器性能:")
    logger.info(f"  真实图像平均分数: {np.mean(real_scores):.4f}")
    logger.info(f"  虚假图像平均分数: {np.mean(fake_scores):.4f}")
    logger.info(f"  判别器区分度: {np.mean(real_scores) - np.mean(fake_scores):.4f}")
    
    return {
        'real_mean_score': float(np.mean(real_scores)),
        'fake_mean_score': float(np.mean(fake_scores)),
        'discrimination_power': float(np.mean(real_scores) - np.mean(fake_scores))
    }


def main():
    parser = argparse.ArgumentParser(description='评估融合模型性能')
    parser.add_argument('--model-path', type=str, default='models/conditional_gan/model_epoch_10.pth', help='模型路径')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2', 
                       choices=['mobilenet_v2', 'efficientnet_b0', 'resnet18'],
                       help='特征提取器模型类型')
    parser.add_argument('--data-dir', type=str, default='data/train', help='数据目录')
    parser.add_argument('--batch-size', type=int, default=8, help='批量大小')
    parser.add_argument('--num-classes', type=int, default=2, help='类别数量')
    parser.add_argument('--latent-dim', type=int, default=100, help='潜在空间维度')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 目标角色
    target_characters = ['蔚蓝档案_阿罗娜', '蔚蓝档案_普拉娜']
    
    # 创建数据集
    logger.info('加载数据集...')
    dataset = CharacterDataset(args.data_dir, transform=transform, target_characters=target_characters)
    
    # 创建数据加载器
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    logger.info(f'测试集大小: {len(dataset)}')
    
    # 加载模型
    logger.info('加载模型...')
    model = ConditionalGAN(args.num_classes, args.latent_dim, model_type=args.model_type)
    model = model.to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    model.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
    
    # 评估分类性能
    class_names = list(dataset.class_to_idx.keys())
    classification_results = evaluate_classification(model, test_loader, device, class_names)
    
    # 评估生成质量
    generation_results = evaluate_generation_quality(model, num_samples=10, device=device)
    
    # 评估判别器性能
    discriminator_results = evaluate_discriminator(model, test_loader, device)
    
    # 汇总结果
    results = {
        'model_path': args.model_path,
        'classification': classification_results,
        'generation': generation_results,
        'discriminator': discriminator_results,
        'class_names': class_names
    }
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f'评估结果已保存到: {output_path}')
    
    # 打印摘要
    logger.info('\n' + '='*50)
    logger.info('评估摘要')
    logger.info('='*50)
    logger.info(f"分类准确率: {classification_results['accuracy'] * 100:.2f}%")
    logger.info(f"生成图像数量: {generation_results['num_samples']}")
    logger.info(f"判别器区分度: {discriminator_results['discrimination_power']:.4f}")
    logger.info('='*50)


if __name__ == '__main__':
    main()

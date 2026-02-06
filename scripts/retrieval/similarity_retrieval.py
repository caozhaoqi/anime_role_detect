#!/usr/bin/env python3
"""
基于相似度的角色检索
使用度量学习模型提取特征并计算相似度
"""
import os
import sys
import argparse
import logging
import json
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.loss.arcface_loss import ArcFaceLoss

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('similarity_retrieval')

class MetricLearningModel(nn.Module):
    """
    用于度量学习的模型
    """
    
    def __init__(self, num_classes, feature_dim=512):
        """
        初始化度量学习模型
        
        Args:
            num_classes: 类别数
            feature_dim: 特征维度
        """
        super(MetricLearningModel, self).__init__()
        
        # 使用EfficientNet-B0作为骨干网络
        self.backbone = models.efficientnet_b0(pretrained=False)
        
        # 获取原始分类器的输入特征数
        in_features = self.backbone.classifier[1].in_features
        
        # 替换分类器为特征提取器
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像
            
        Returns:
            features: 归一化的特征向量
        """
        features = self.backbone(x)
        # 归一化特征
        features = nn.functional.normalize(features, dim=1)
        return features

def load_model(model_path):
    """
    加载度量学习模型
    
    Args:
        model_path: 模型权重路径
        
    Returns:
        model: 加载好的模型
        class_to_idx: 类别到索引的映射
        idx_to_class: 索引到类别的映射
    """
    logger.info(f"加载模型: {model_path}")
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 获取类别信息
    class_to_idx = checkpoint.get('class_to_idx', {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    # 获取特征维度
    feature_dim = checkpoint.get('feature_dim', 512)
    
    # 创建模型
    model = MetricLearningModel(num_classes=num_classes, feature_dim=feature_dim)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"模型加载成功，包含 {num_classes} 个类别，特征维度: {feature_dim}")
    
    return model, class_to_idx, idx_to_class

def extract_features(model, image_path, device):
    """
    提取图像特征
    
    Args:
        model: 模型
        image_path: 图像路径
        device: 设备
        
    Returns:
        features: 特征向量
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 提取特征
    with torch.no_grad():
        features = model(image_tensor)
    
    return features.squeeze().cpu().numpy()

def compute_similarity(feature1, feature2):
    """
    计算两个特征向量的相似度
    
    Args:
        feature1: 第一个特征向量
        feature2: 第二个特征向量
        
    Returns:
        similarity: 余弦相似度
    """
    similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    return similarity

def build_feature_database(model, data_dir, device):
    """
    构建特征数据库
    
    Args:
        model: 模型
        data_dir: 数据目录
        device: 设备
        
    Returns:
        feature_db: 特征数据库，格式为 {class_name: [feature1, feature2, ...]}
    """
    logger.info(f"构建特征数据库，数据目录: {data_dir}")
    
    feature_db = {}
    
    # 遍历数据目录
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.isdir(class_dir):
            continue
        
        # 为每个类别提取特征
        class_features = []
        
        for image_name in os.listdir(class_dir):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            image_path = os.path.join(class_dir, image_name)
            
            try:
                # 提取特征
                feature = extract_features(model, image_path, device)
                class_features.append(feature)
                
                if len(class_features) >= 5:  # 每个类别最多提取5个特征
                    break
                    
            except Exception as e:
                logger.error(f"处理图像 {image_path} 失败: {e}")
                continue
        
        if class_features:
            feature_db[class_name] = class_features
            logger.info(f"类别 {class_name} 已添加 {len(class_features)} 个特征")
    
    logger.info(f"特征数据库构建完成，包含 {len(feature_db)} 个类别")
    return feature_db

def search_similar(model, image_path, feature_db, device, top_k=5):
    """
    搜索相似角色
    
    Args:
        model: 模型
        image_path: 查询图像路径
        feature_db: 特征数据库
        device: 设备
        top_k: 返回前k个最相似的结果
        
    Returns:
        results: 相似角色列表，每个元素为 (class_name, similarity)
    """
    logger.info(f"搜索相似角色，查询图像: {image_path}")
    
    # 提取查询图像的特征
    query_feature = extract_features(model, image_path, device)
    
    # 计算与每个类别的相似度
    similarities = []
    
    for class_name, class_features in feature_db.items():
        # 计算与该类别所有特征的平均相似度
        class_similarities = []
        
        for feature in class_features:
            similarity = compute_similarity(query_feature, feature)
            class_similarities.append(similarity)
        
        # 计算平均相似度
        avg_similarity = np.mean(class_similarities)
        similarities.append((class_name, avg_similarity))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 返回前k个结果
    results = similarities[:top_k]
    
    logger.info(f"搜索完成，找到 {len(results)} 个相似角色")
    
    return results

def preprocess_image(image_path):
    """
    预处理图像
    
    Args:
        image_path: 图像路径
        
    Returns:
        image_tensor: 预处理后的图像张量
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='基于相似度的角色检索')
    
    parser.add_argument('--model', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--image', type=str, required=True,
                       help='查询图像路径')
    parser.add_argument('--data-dir', type=str, default='data/train',
                       help='训练数据目录，用于构建特征数据库')
    parser.add_argument('--top-k', type=int, default=5,
                       help='返回前k个最相似的结果')
    
    args = parser.parse_args()
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    model, class_to_idx, idx_to_class = load_model(args.model)
    model.to(device)
    
    # 构建特征数据库
    feature_db = build_feature_database(model, args.data_dir, device)
    
    # 搜索相似角色
    results = search_similar(model, args.image, feature_db, device, args.top_k)
    
    # 打印结果
    print("\n" + "="*60)
    print("相似角色检索结果")
    print("="*60)
    print(f"查询图像: {args.image}")
    print("\nTop {args.top_k} 相似角色:")
    
    for i, (class_name, similarity) in enumerate(results, 1):
        print(f"{i}. {class_name} - 相似度: {similarity:.4f}")
    
    print("="*60)

if __name__ == '__main__':
    main()

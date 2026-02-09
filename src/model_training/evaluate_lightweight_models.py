#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级模型评估脚本

评估不同轻量级模型的性能、推理速度和模型大小，以选择最优的模型架构。
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import logging
import time
import numpy as np
from tqdm import tqdm
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('lightweight_evaluation')

class CharacterDataset(torch.utils.data.Dataset):
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
    elif model_type == 'efficientnet_lite':
        logger.info("加载模型: EfficientNet-Lite")
        try:
            import timm
            model = timm.create_model('efficientnet_lite0', pretrained=True, num_classes=num_classes)
        except Exception as e:
            logger.warning(f"无法加载EfficientNet-Lite: {e}，使用MobileNetV2作为替代")
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
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

def evaluate_model_performance(model, data_loader, device):
    """评估模型性能
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
    
    Returns:
        准确率, 平均推理时间, FPS
    """
    model.eval()
    correct = 0
    total = 0
    total_inference_time = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="评估模型性能"):
            images = images.to(device)
            labels = labels.to(device)
            
            # 记录推理开始时间
            start_time = time.time()
            
            # 模型推理
            outputs = model(images)
            
            # 记录推理结束时间
            end_time = time.time()
            inference_time = end_time - start_time
            total_inference_time += inference_time
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = (correct / total) * 100
    avg_inference_time = (total_inference_time / total) * 1000  # 转换为毫秒
    fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
    
    return accuracy, avg_inference_time, fps

def main():
    parser = argparse.ArgumentParser(description='轻量级模型评估脚本')
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='评估数据目录')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批量大小')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载器工作线程数')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
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
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    num_classes = len(dataset.class_to_idx)
    logger.info(f"类别数量: {num_classes}")
    
    # 要评估的模型列表
    model_types = [
        'efficientnet_b0',  # 基准模型
        'mobilenet_v2',
        'shufflenet_v2',
        'squeezenet',
        'resnet18',
        'efficientnet_lite'
    ]
    
    # 评估结果
    results = []
    
    for model_type in model_types:
        logger.info(f"\n评估模型: {model_type}")
        
        # 获取模型
        model = get_model(model_type, num_classes)
        model.to(device)
        
        # 计算模型大小
        model_size, num_params = calculate_model_size(model)
        logger.info(f"模型大小: {model_size:.2f} MB, 参数数量: {num_params:,}")
        
        # 评估模型性能
        accuracy, avg_inference_time, fps = evaluate_model_performance(model, data_loader, device)
        logger.info(f"准确率: {accuracy:.2f}%, 平均推理时间: {avg_inference_time:.2f} ms, FPS: {fps:.2f}")
        
        # 保存结果
        results.append({
            'model_type': model_type,
            'model_size_mb': model_size,
            'num_params': num_params,
            'accuracy': accuracy,
            'avg_inference_time_ms': avg_inference_time,
            'fps': fps
        })
    
    # 保存评估结果
    results_path = os.path.join(args.output_dir, 'lightweight_models_evaluation.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存到: {results_path}")
    
    # 打印评估摘要
    logger.info("\n" + "="*80)
    logger.info("轻量级模型评估摘要")
    logger.info("="*80)
    
    # 按准确率排序
    results_sorted_by_accuracy = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    # 按FPS排序
    results_sorted_by_fps = sorted(results, key=lambda x: x['fps'], reverse=True)
    # 按模型大小排序
    results_sorted_by_size = sorted(results, key=lambda x: x['model_size_mb'])
    
    logger.info("\n按准确率排序:")
    for i, result in enumerate(results_sorted_by_accuracy[:3], 1):
        logger.info(f"{i}. {result['model_type']}: 准确率 {result['accuracy']:.2f}%, FPS {result['fps']:.2f}, 模型大小 {result['model_size_mb']:.2f} MB")
    
    logger.info("\n按推理速度排序:")
    for i, result in enumerate(results_sorted_by_fps[:3], 1):
        logger.info(f"{i}. {result['model_type']}: FPS {result['fps']:.2f}, 准确率 {result['accuracy']:.2f}%, 模型大小 {result['model_size_mb']:.2f} MB")
    
    logger.info("\n按模型大小排序:")
    for i, result in enumerate(results_sorted_by_size[:3], 1):
        logger.info(f"{i}. {result['model_type']}: 模型大小 {result['model_size_mb']:.2f} MB, 准确率 {result['accuracy']:.2f}%, FPS {result['fps']:.2f}")
    
    # 选择最佳平衡模型
    # 计算每个模型的综合评分（准确率权重0.6，FPS权重0.4）
    for result in results:
        # 归一化指标
        max_accuracy = max(r['accuracy'] for r in results)
        max_fps = max(r['fps'] for r in results)
        normalized_accuracy = result['accuracy'] / max_accuracy
        normalized_fps = result['fps'] / max_fps
        # 计算综合评分
        score = 0.6 * normalized_accuracy + 0.4 * normalized_fps
        result['score'] = score
    
    # 按综合评分排序
    results_sorted_by_score = sorted(results, key=lambda x: x['score'], reverse=True)
    best_model = results_sorted_by_score[0]
    
    logger.info("\n" + "="*80)
    logger.info("最佳平衡模型")
    logger.info("="*80)
    logger.info(f"模型类型: {best_model['model_type']}")
    logger.info(f"准确率: {best_model['accuracy']:.2f}%")
    logger.info(f"推理速度: {best_model['fps']:.2f} FPS ({best_model['avg_inference_time_ms']:.2f} ms/张)")
    logger.info(f"模型大小: {best_model['model_size_mb']:.2f} MB")
    logger.info(f"参数数量: {best_model['num_params']:,}")
    logger.info(f"综合评分: {best_model['score']:.4f}")
    
    logger.info("\n轻量级模型评估完成")

if __name__ == '__main__':
    main()

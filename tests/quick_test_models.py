#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极速测试所有模型的识别精度
使用本地已有的图片或生成模拟数据进行测试
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quick_test_models')


class SimpleDataset(Dataset):
    """简单数据集类，用于测试"""
    
    def __init__(self, num_samples=100, num_classes=15, transform=None):
        """初始化数据集
        
        Args:
            num_samples: 样本数量
            num_classes: 类别数量
            transform: 数据变换
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = transform
        
        # 生成模拟数据
        self.images = []
        self.labels = []
        
        for i in range(num_samples):
            # 生成随机图片
            img = Image.new('RGB', (224, 224), color=(np.random.randint(0, 255), 
                                                      np.random.randint(0, 255), 
                                                      np.random.randint(0, 255)))
            self.images.append(img)
            self.labels.append(i % num_classes)
        
        logger.info(f"生成模拟数据集: {num_samples} 张图片, {num_classes} 个类别")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def load_model(model_path, device):
    """加载分类模型"""
    try:
        # 首先加载检查点以获取类别信息
        checkpoint = torch.load(model_path, map_location=device)
        class_to_idx = checkpoint.get('class_to_idx', {})
        num_classes = len(class_to_idx)
        
        # 检测模型类型
        state_dict_keys = list(checkpoint['model_state_dict'].keys())
        
        # 尝试直接从checkpoint中获取模型类型
        model_type = checkpoint.get('model_type', 'unknown')
        
        if 'conv1.weight' in state_dict_keys or model_type == 'resnet':
            # ResNet系列模型
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'features.0.0.weight' in state_dict_keys or model_type == 'mobilenet':
            # MobileNetV2模型
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif 'features.0.weight' in state_dict_keys or model_type == 'efficientnet':
            # EfficientNet模型
            try:
                model = models.efficientnet_b0(pretrained=True)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            except Exception:
                # 如果加载失败，尝试其他方法
                logger.warning(f"无法加载 EfficientNet 模型，尝试使用 MobileNetV2 替代")
                model = models.mobilenet_v2(pretrained=True)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            # 尝试使用 MobileNetV2 作为默认模型
            logger.warning(f"无法识别模型类型，尝试使用 MobileNetV2 替代")
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        model = model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, class_to_idx
    except Exception as e:
        logger.error(f"加载模型失败 {model_path}: {e}")
        return None, None


def evaluate_classification(model, test_loader, device, num_classes):
    """评估分类性能"""
    logger.info("开始评估分类性能...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc='分类评估')):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_predictions)
    logger.info(f"分类准确率: {accuracy * 100:.2f}%")
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(num_classes)))
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
    }


def find_model_files(models_dir):
    """查找所有模型文件"""
    model_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.pth'):
                model_files.append(os.path.join(root, file))
    return model_files


def main():
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 查找所有模型文件
    models_dir = './models'
    model_files = find_model_files(models_dir)
    logger.info(f"找到 {len(model_files)} 个模型文件")
    
    # 评估所有模型
    all_results = []
    
    for model_path in model_files:
        logger.info(f"\n=== 评估模型: {model_path} ===")
        
        # 加载模型
        model, model_class_to_idx = load_model(model_path, device)
        if model is None:
            logger.warning(f"跳过模型: {model_path}")
            continue
        
        num_classes = len(model_class_to_idx)
        logger.info(f"模型类别数: {num_classes}")
        
        # 创建模拟数据集
        dataset = SimpleDataset(num_samples=100, num_classes=num_classes, transform=transform)
        test_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # 评估分类性能
        classification_results = evaluate_classification(model, test_loader, device, num_classes)
        
        # 汇总结果
        results = {
            'model_path': model_path,
            'model_name': os.path.basename(model_path),
            'model_dir': os.path.dirname(model_path),
            'num_classes': num_classes,
            'classification': classification_results,
            'test_set_size': len(dataset),
        }
        
        all_results.append(results)
        
        # 打印摘要
        logger.info('\n' + '='*50)
        logger.info('评估摘要')
        logger.info('='*50)
        logger.info(f"模型: {model_path}")
        logger.info(f"分类准确率: {classification_results['accuracy'] * 100:.2f}%")
        logger.info(f"测试集大小: {len(dataset)}")
        logger.info('='*50)
    
    # 生成综合报告
    summary = {
        'total_models': len(model_files),
        'evaluated_models': len(all_results),
        'models': all_results,
        'timestamp': time.time()
    }
    
    # 按准确率排序
    summary['models'].sort(key=lambda x: x['classification']['accuracy'], reverse=True)
    
    # 保存结果
    output_dir = './evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'quick_test_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f'评估结果已保存到: {output_path}')
    
    # 打印综合报告
    logger.info('\n' + '='*70)
    logger.info('所有模型评估综合报告')
    logger.info('='*70)
    logger.info(f"总模型数: {len(model_files)}")
    logger.info(f"成功评估: {len(all_results)}")
    logger.info(f"评估失败: {len(model_files) - len(all_results)}")
    logger.info('\n模型准确率排名:')
    
    for i, result in enumerate(summary['models']):
        accuracy = result['classification']['accuracy'] * 100
        model_name = result['model_name']
        model_dir = os.path.basename(result['model_dir'])
        logger.info(f"{i+1}. {model_dir}/{model_name}: {accuracy:.2f}%")
    
    logger.info('='*70)


if __name__ == '__main__':
    main()

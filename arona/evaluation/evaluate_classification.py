#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估分类模型性能
专门用于评估arona_plana分类模型
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
logger = logging.getLogger('evaluate_classification')


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


def load_model(model_path, device):
    """加载分类模型"""
    # 首先加载检查点以获取类别信息
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint.get('class_to_idx', {})
    num_classes = len(class_to_idx)
    
    # 检测模型类型
    state_dict_keys = list(checkpoint['model_state_dict'].keys())
    if 'conv1.weight' in state_dict_keys:
        # ResNet系列模型
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'features.0.0.weight' in state_dict_keys:
        # MobileNetV2模型
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif 'features.0.weight' in state_dict_keys:
        # EfficientNet模型
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("无法识别模型类型")
    
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, class_to_idx


def evaluate_classification(model, test_loader, device, class_names):
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
    
    # 获取实际出现的类别
    unique_labels = sorted(list(set(all_labels)))
    actual_class_names = [class_names[label] for label in unique_labels]
    
    # 生成分类报告
    report = classification_report(all_labels, all_predictions, 
                               labels=unique_labels,
                               target_names=actual_class_names, 
                               output_dict=True)
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions, labels=unique_labels)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'actual_classes': actual_class_names
    }


def main():
    parser = argparse.ArgumentParser(description='评估分类模型性能')
    parser.add_argument('--model-path', type=str, default='models/arona_plana/model_best.pth', help='模型路径')
    parser.add_argument('--data-dir', type=str, default='data/train', help='数据目录')
    parser.add_argument('--batch-size', type=int, default=8, help='批量大小')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='输出目录')
    
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
    
    # 创建数据集（加载所有类别，包括目标角色和其他角色）
    logger.info('加载数据集...')
    dataset = CharacterDataset(args.data_dir, transform=transform)
    
    # 创建数据加载器
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f'测试集大小: {len(dataset)}')
    
    # 加载模型
    logger.info('加载模型...')
    model, model_class_to_idx = load_model(args.model_path, device)
    
    # 过滤测试数据，只包含模型训练时存在的类别
    logger.info('过滤测试数据...')
    filtered_images = []
    filtered_labels = []
    filtered_class_to_idx = {}
    
    # 构建从原始类别到模型类别的映射
    for cls_name, cls_idx in dataset.class_to_idx.items():
        if cls_name in model_class_to_idx:
            filtered_class_to_idx[cls_name] = model_class_to_idx[cls_name]
    
    # 过滤图像和标签
    for img_path, label in zip(dataset.images, dataset.labels):
        cls_name = list(dataset.class_to_idx.keys())[list(dataset.class_to_idx.values()).index(label)]
        if cls_name in model_class_to_idx:
            filtered_images.append(img_path)
            filtered_labels.append(model_class_to_idx[cls_name])
    
    # 创建过滤后的数据集
    class FilteredDataset(Dataset):
        def __init__(self, images, labels, transform):
            self.images = images
            self.labels = labels
            self.transform = transform
        def __len__(self):
            return len(self.images)
        def __getitem__(self, idx):
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
    
    filtered_dataset = FilteredDataset(filtered_images, filtered_labels, transform)
    filtered_loader = DataLoader(filtered_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    logger.info(f'过滤后的测试集大小: {len(filtered_dataset)}')
    
    # 评估分类性能
    class_names = list(filtered_class_to_idx.keys())
    classification_results = evaluate_classification(model, filtered_loader, device, class_names)
    
    # 汇总结果
    results = {
        'model_path': args.model_path,
        'classification': classification_results,
        'class_names': class_names,
        'actual_classes': classification_results.get('actual_classes', []),
        'test_set_size': len(dataset),
        'filtered_test_set_size': len(filtered_dataset)
    }
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'classification_evaluation_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f'评估结果已保存到: {output_path}')
    
    # 打印摘要
    logger.info('\n' + '='*50)
    logger.info('评估摘要')
    logger.info('='*50)
    logger.info(f"分类准确率: {classification_results['accuracy'] * 100:.2f}%")
    logger.info(f"原始测试集大小: {len(dataset)}")
    logger.info(f"过滤后测试集大小: {len(filtered_dataset)}")
    logger.info(f"实际评估类别: {classification_results.get('actual_classes', [])}")
    logger.info('\n详细分类报告:')
    for class_name, metrics in classification_results['classification_report'].items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            logger.info(f"  {class_name}:")
            logger.info(f"    精确率: {metrics['precision'] * 100:.2f}%")
            logger.info(f"    召回率: {metrics['recall'] * 100:.2f}%")
            logger.info(f"    F1分数: {metrics['f1-score'] * 100:.2f}%")
    logger.info('='*50)


if __name__ == '__main__':
    main()

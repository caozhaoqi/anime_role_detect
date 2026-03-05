#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有模型的识别精度并输出综合报告
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
logger = logging.getLogger('test_all_models')


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
        all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        logger.info(f"发现 {len(all_classes)} 个类别目录")
        
        idx = 0
        for cls in all_classes:
            # 提取角色名（去掉前缀）
            if cls.startswith('blda_spider_img_keyword_'):
                role_name = cls[len('blda_spider_img_keyword_'):]
            else:
                role_name = cls
            
            self.class_to_idx[role_name] = idx
            cls_dir = os.path.join(root_dir, cls)
            
            # 统计每个类别的图像数量
            img_count = 0
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(idx)
                    img_count += 1
            
            logger.info(f"类别 {role_name} (目录: {cls}) 包含 {img_count} 张图像")
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


class FilteredDataset(Dataset):
    """过滤后的数据集"""
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
    
    # 加载数据集
    data_dir = 'data/train'
    logger.info('加载数据集...')
    dataset = CharacterDataset(data_dir, transform=transform)
    
    # 查找所有模型文件
    models_dir = '../models' if os.path.exists('../models') else './models'
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
        
        # 过滤测试数据，只包含模型训练时存在的类别
        logger.info('过滤测试数据...')
        filtered_images = []
        filtered_labels = []
        filtered_class_to_idx = {}
        
        # 构建从原始类别到模型类别的映射
        # 尝试直接匹配和部分匹配
        for cls_name, cls_idx in dataset.class_to_idx.items():
            # 直接匹配
            if cls_name in model_class_to_idx:
                filtered_class_to_idx[cls_name] = model_class_to_idx[cls_name]
            else:
                # 尝试部分匹配（角色名在模型类别名中）
                for model_cls, model_idx in model_class_to_idx.items():
                    if cls_name in model_cls:
                        filtered_class_to_idx[cls_name] = model_idx
                        logger.info(f"匹配类别: {cls_name} -> {model_cls}")
                        break
        
        if not filtered_class_to_idx:
            logger.warning(f"模型 {model_path} 没有与测试数据匹配的类别")
            continue
        
        # 过滤图像和标签
        logger.info(f"数据集图像数量: {len(dataset.images)}")
        logger.info(f"数据集标签数量: {len(dataset.labels)}")
        logger.info(f"过滤后的类别映射: {filtered_class_to_idx}")
        
        # 创建从标签到类别名的映射
        idx_to_cls = {v: k for k, v in dataset.class_to_idx.items()}
        logger.info(f"标签到类别名的映射: {idx_to_cls}")
        
        for img_path, label in zip(dataset.images, dataset.labels):
            if label in idx_to_cls:
                cls_name = idx_to_cls[label]
                logger.info(f"处理图像: {img_path}, 标签: {label}, 类别名: {cls_name}")
                if cls_name in filtered_class_to_idx:
                    filtered_images.append(img_path)
                    filtered_labels.append(filtered_class_to_idx[cls_name])
                    logger.info(f"添加到过滤数据集: {img_path}")
            else:
                logger.warning(f"标签 {label} 不在 idx_to_cls 中")
        
        if not filtered_images:
            logger.warning(f"模型 {model_path} 没有可测试的图像")
            continue
        
        # 创建过滤后的数据集
        filtered_dataset = FilteredDataset(filtered_images, filtered_labels, transform)
        filtered_loader = DataLoader(filtered_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        logger.info(f'过滤后的测试集大小: {len(filtered_dataset)}')
        
        # 评估分类性能
        # 构建从模型类别索引到类名的映射
        idx_to_class = {v: k for k, v in filtered_class_to_idx.items()}
        # 确保类名列表与模型类别索引对应
        max_idx = max(idx_to_class.keys()) if idx_to_class else 0
        class_names = [''] * (max_idx + 1)
        for idx, name in idx_to_class.items():
            class_names[idx] = name
        classification_results = evaluate_classification(model, filtered_loader, device, class_names)
        
        # 汇总结果
        results = {
            'model_path': model_path,
            'model_name': os.path.basename(model_path),
            'model_dir': os.path.dirname(model_path),
            'classification': classification_results,
            'class_names': class_names,
            'actual_classes': classification_results.get('actual_classes', []),
            'test_set_size': len(dataset),
            'filtered_test_set_size': len(filtered_dataset)
        }
        
        all_results.append(results)
        
        # 打印摘要
        logger.info('\n' + '='*50)
        logger.info('评估摘要')
        logger.info('='*50)
        logger.info(f"模型: {model_path}")
        logger.info(f"分类准确率: {classification_results['accuracy'] * 100:.2f}%")
        logger.info(f"原始测试集大小: {len(dataset)}")
        logger.info(f"过滤后测试集大小: {len(filtered_dataset)}")
        logger.info(f"实际评估类别: {classification_results.get('actual_classes', [])}")
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
    output_dir = '../evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'all_models_evaluation_results.json')
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

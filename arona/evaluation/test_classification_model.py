#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试分类模型效果
使用训练好的模型对测试数据进行分类预测
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
logger = logging.getLogger('test_classification_model')


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
        self.idx_to_class = {}
        
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
            self.idx_to_class[idx] = cls
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(idx)
            idx += 1
        
        logger.info(f"数据集初始化完成，包含 {len(self.class_to_idx)} 个类别，{len(self.images)} 张图像")
        logger.info(f"类别映射: {self.class_to_idx}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def load_model(model_path, model_type='mobilenet_v2'):
    """加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        model_type: 模型类型
        
    Returns:
        model: 加载的模型
        class_to_idx: 类别到索引的映射
    """
    logger.info(f"加载模型: {model_path}")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 从checkpoint中获取类别数
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        num_classes = len(class_to_idx)
    else:
        # 尝试从模型状态字典中推断类别数
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 检查分类器的权重形状
        if 'classifier.5.weight' in state_dict:
            num_classes = state_dict['classifier.5.weight'].shape[0]
        elif 'fc.5.weight' in state_dict:
            num_classes = state_dict['fc.5.weight'].shape[0]
        elif 'classifier.1.weight' in state_dict:
            num_classes = state_dict['classifier.1.weight'].shape[0]
        elif 'fc.weight' in state_dict:
            num_classes = state_dict['fc.weight'].shape[0]
        else:
            num_classes = 2  # 默认值
        
        class_to_idx = {}
    
    logger.info(f"检测到类别数: {num_classes}")
    
    # 创建模型
    if model_type == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        # 调整分类器，与train_incremental.py中的结构一致
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.15),
            nn.Linear(512, num_classes)
        )
    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        # 调整分类器，与train_incremental.py中的结构一致
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.15),
            nn.Linear(512, num_classes)
        )
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=False)
        # 调整分类器，与train_incremental.py中的结构一致
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.15),
            nn.Linear(512, num_classes)
        )
    elif model_type == 'resnet18':
        model = models.resnet18(pretrained=False)
        # 调整分类器，与train_incremental.py中的结构一致
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.15),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info(f"模型加载完成，类别映射: {class_to_idx}")
    
    return model, class_to_idx


def evaluate_model(model, test_loader, device, class_names):
    """评估模型性能
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        class_names: 类别名称列表
        
    Returns:
        results: 评估结果字典
    """
    logger.info("开始评估模型性能...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='评估中'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_predictions)
    logger.info(f"分类准确率: {accuracy * 100:.2f}%")
    
    # 确保labels参数与类别索引一致
    labels = list(range(len(class_names)))
    
    # 生成分类报告
    report = classification_report(all_labels, all_predictions, 
                               target_names=class_names, 
                               labels=labels,
                               output_dict=True)
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions, labels=labels)
    
    logger.info("\n分类报告:")
    logger.info(classification_report(all_labels, all_predictions, target_names=class_names, labels=labels))
    
    logger.info("\n混淆矩阵:")
    logger.info(cm)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': [p.tolist() for p in all_probs]
    }


def test_single_image(model, image_path, transform, device, class_names):
    """测试单张图像
    
    Args:
        model: 模型
        image_path: 图像路径
        transform: 数据变换
        device: 设备
        class_names: 类别名称列表
    """
    logger.info(f"测试单张图像: {image_path}")
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 应用变换
    if transform:
        image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence = probs[0][predicted.item()].item()
    
    logger.info(f"预测结果: {predicted_class}")
    logger.info(f"置信度: {confidence * 100:.2f}%")
    logger.info(f"各类别概率:")
    for i, class_name in enumerate(class_names):
        logger.info(f"  {class_name}: {probs[0][i].item() * 100:.2f}%")
    
    return predicted_class, confidence, probs[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='测试分类模型效果')
    parser.add_argument('--model-path', type=str, default='models/arona_plana/model_best.pth', 
                       help='模型文件路径')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2',
                       choices=['mobilenet_v2', 'efficientnet_b0', 'resnet18', 'resnet50'],
                       help='模型类型')
    parser.add_argument('--data-dir', type=str, default='../data/downloaded_images', 
                       help='数据目录')
    parser.add_argument('--batch-size', type=int, default=8, help='批量大小')
    parser.add_argument('--test-image', type=str, default=None, 
                       help='单张测试图像路径（可选）')
    parser.add_argument('--output-dir', type=str, default='test_results', 
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载模型
    model, loaded_class_to_idx = load_model(args.model_path, args.model_type)
    model = model.to(device)
    
    # 创建数据集 - 加载所有类别
    logger.info('加载数据集...')
    dataset = CharacterDataset(args.data_dir, transform=transform)
    
    # 获取类别名称
    class_names = list(dataset.class_to_idx.keys())
    logger.info(f'类别名称: {class_names}')
    
    # 如果指定了单张测试图像
    if args.test_image:
        test_single_image(model, args.test_image, transform, device, class_names)
        return
    
    # 创建数据加载器
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f'测试集大小: {len(dataset)}')
    
    # 评估模型
    results = evaluate_model(model, test_loader, device, class_names)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'test_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        # 只保存可序列化的数据
        save_results = {
            'accuracy': results['accuracy'],
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'],
            'class_names': class_names
        }
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f'\n测试结果已保存到: {output_path}')
    
    # 打印摘要
    logger.info('\n' + '='*50)
    logger.info('测试摘要')
    logger.info('='*50)
    logger.info(f"分类准确率: {results['accuracy'] * 100:.2f}%")
    logger.info(f"测试图像数量: {len(dataset)}")
    logger.info(f"类别数量: {len(class_names)}")
    logger.info('='*50)


if __name__ == '__main__':
    main()

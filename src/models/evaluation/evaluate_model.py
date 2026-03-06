#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的模型评估脚本
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import logging
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.core.classification.models import get_model, get_model_with_attributes

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('evaluate_model')


class CharacterDataset(torch.utils.data.Dataset):
    """角色数据集类"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        classes = sorted(os.listdir(self.data_dir))
        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for file in os.listdir(class_dir):
                if file.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.images.append(os.path.join(class_name, file))
                    self.labels.append(idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CharacterAttributeDataset(torch.utils.data.Dataset):
    """带有属性标签的角色数据集类"""
    def __init__(self, data_dir, annotations_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.annotations = []
        self.class_to_idx = {}
        
        # 加载标注
        self._load_annotations(annotations_file)
    
    def _load_annotations(self, annotations_file):
        """加载标注"""
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # 构建类别映射
        classes = set()
        for item in annotations:
            classes.add(item['character'])
        
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(classes))}
        
        # 加载标注
        for item in annotations:
            image_path = os.path.join(self.data_dir, item['image_path'])
            if os.path.exists(image_path):
                self.annotations.append({
                    'image_path': image_path,
                    'character': item['character'],
                    'attribute_labels': item['attribute_labels']
                })
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        image = Image.open(item['image_path']).convert('RGB')
        class_label = self.class_to_idx[item['character']]
        attribute_labels = torch.tensor(item['attribute_labels'], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_label, attribute_labels


def load_model(model_path, model_type):
    """加载模型"""
    logger.info(f"加载模型: {model_path}")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 从checkpoint中获取类别数
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        num_classes = len(class_to_idx)
    else:
        num_classes = 5
    
    logger.info(f"检测到类别数: {num_classes}")
    
    # 创建模型
    if 'attribute_predictor' in checkpoint['model_state_dict']:
        # 带有属性预测的模型
        num_attributes = checkpoint['model_state_dict']['attribute_predictor.weight'].shape[0]
        model = get_model_with_attributes(model_type, num_classes, num_attributes)
    else:
        # 普通分类模型
        model = get_model(model_type, num_classes)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info(f"模型加载完成")
    
    return model, class_to_idx


def evaluate_model(model, test_loader, device, class_names):
    """评估模型"""
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='评估模型'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    return results


def evaluate_model_with_attributes(model, test_loader, device, class_names, attribute_names):
    """评估带有属性预测的模型"""
    model.eval()
    
    y_true = []
    y_pred = []
    attribute_true = []
    attribute_pred = []
    
    with torch.no_grad():
        for images, class_labels, attribute_labels in tqdm(test_loader, desc='评估模型'):
            images = images.to(device)
            class_labels = class_labels.to(device)
            attribute_labels = attribute_labels.to(device)
            
            class_outputs, attribute_outputs = model(images)
            _, class_preds = torch.max(class_outputs, 1)
            
            y_true.extend(class_labels.cpu().numpy())
            y_pred.extend(class_preds.cpu().numpy())
            attribute_true.extend(attribute_labels.cpu().numpy())
            attribute_pred.extend(attribute_outputs.cpu().numpy())
    
    # 计算分类评估指标
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算属性预测评估指标
    import numpy as np
    attribute_accuracy = []
    for i in range(len(attribute_names)):
        attr_true = np.array(attribute_true)[:, i]
        attr_pred = np.array(attribute_pred)[:, i]
        # 对于二分类属性，使用准确率
        if len(np.unique(attr_true)) == 2:
            attr_acc = accuracy_score(attr_true, np.round(attr_pred))
        else:
            # 对于多分类属性，使用MSE
            attr_acc = np.mean((attr_true - attr_pred) ** 2)
        attribute_accuracy.append(attr_acc)
    
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'attribute_accuracy': dict(zip(attribute_names, attribute_accuracy))
    }
    
    return results


def test_single_image(model, image_path, transform, device, class_names, attribute_names=None):
    """测试单张图像"""
    logger.info(f"测试图像: {image_path}")
    
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        if attribute_names:
            class_output, attribute_output = model(image_tensor)
        else:
            class_output = model(image_tensor)
    
    # 分类预测
    class_prob = torch.softmax(class_output, dim=1)
    class_idx = torch.argmax(class_prob, dim=1).item()
    class_confidence = class_prob[0, class_idx].item()
    predicted_class = class_names[class_idx]
    
    # 构建结果
    result = {
        "character": predicted_class,
        "confidence": class_confidence
    }
    
    # 属性预测
    if attribute_names:
        attribute_preds = attribute_output.squeeze().cpu().numpy()
        predicted_attributes = {}
        for i, attr_name in enumerate(attribute_names):
            predicted_attributes[attr_name] = attribute_preds[i]
        result["attributes"] = predicted_attributes
    
    return result


def main():
    parser = argparse.ArgumentParser(description='统一的模型评估脚本')
    parser.add_argument('--model-path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2', 
                       choices=['mobilenet_v2', 'efficientnet_b0', 'resnet18'],
                       help='模型类型')
    parser.add_argument('--data-dir', type=str, default='data/downloaded_images', help='数据目录')
    parser.add_argument('--annotations-file', type=str, default=None, help='属性标注文件路径')
    parser.add_argument('--batch-size', type=int, default=8, help='批量大小')
    parser.add_argument('--output-dir', type=str, default='test_results', help='输出目录')
    parser.add_argument('--test-image', type=str, default=None, help='测试单张图像')
    
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
    model, class_to_idx = load_model(args.model_path, args.model_type)
    model = model.to(device)
    
    # 获取类别名称
    class_names = list(class_to_idx.keys())
    logger.info(f'类别名称: {class_names}')
    
    # 属性名称
    attribute_names = ['hair_color', 'eye_color', 'has_halo', 'outfit', 'hair_style', 'accessories']
    
    # 如果指定了单张测试图像
    if args.test_image:
        result = test_single_image(model, args.test_image, transform, device, class_names, attribute_names if args.annotations_file else None)
        logger.info("\n" + "="*50)
        logger.info("预测结果:")
        logger.info("="*50)
        logger.info(f"角色: {result['character']}")
        logger.info(f"置信度: {result['confidence']:.4f}")
        if 'attributes' in result:
            logger.info("\n属性预测:")
            for attr, value in result['attributes'].items():
                logger.info(f"  {attr}: {value}")
        logger.info("="*50)
        return
    
    # 创建数据集
    if args.annotations_file:
        logger.info('加载带有属性标签的数据集...')
        dataset = CharacterAttributeDataset(args.data_dir, args.annotations_file, transform=transform)
    else:
        logger.info('加载普通数据集...')
        dataset = CharacterDataset(args.data_dir, transform=transform)
    
    # 创建数据加载器
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f'测试集大小: {len(dataset)}')
    
    # 评估模型
    if args.annotations_file:
        results = evaluate_model_with_attributes(model, test_loader, device, class_names, attribute_names)
    else:
        results = evaluate_model(model, test_loader, device, class_names)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'test_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    logger.info("\n" + "="*50)
    logger.info("评估结果:")
    logger.info("="*50)
    logger.info(f"准确率: {results['accuracy']:.4f}")
    
    if 'attribute_accuracy' in results:
        logger.info("\n属性预测准确率:")
        for attr, acc in results['attribute_accuracy'].items():
            logger.info(f"  {attr}: {acc:.4f}")
    
    logger.info("\n详细报告已保存到: {output_path}")
    logger.info("="*50)


if __name__ == '__main__':
    main()
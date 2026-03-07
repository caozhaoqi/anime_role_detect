#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带有属性标签的角色分类模型测试脚本
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import logging
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.models import get_model_with_attributes

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_with_attributes')


class CharacterAttributeDataset(torch.utils.data.Dataset):
    """带有属性标签的角色数据集类"""
    
    def __init__(self, root_dir, annotations_file, transform=None):
        """初始化数据集
        
        Args:
            root_dir: 数据目录
            annotations_file: 标注文件路径
            transform: 数据变换
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 加载标注
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 构建类别映射
        self.class_to_idx = {}
        idx = 0
        for ann in self.annotations:
            character = ann['character']
            if character not in self.class_to_idx:
                self.class_to_idx[character] = idx
                idx += 1
        
        logger.info(f"数据集初始化完成，包含 {len(self.class_to_idx)} 个类别，{len(self.annotations)} 张图像")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.root_dir, ann['image_path'])
        image = Image.open(img_path).convert('RGB')
        
        # 类别标签
        character = ann['character']
        label = self.class_to_idx[character]
        
        # 属性标签
        attribute_labels = ann['attribute_labels']
        attribute_labels = torch.tensor(attribute_labels, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, attribute_labels


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
        if 'classifier.weight' in state_dict:
            num_classes = state_dict['classifier.weight'].shape[0]
        else:
            num_classes = 5  # 默认值
        
        class_to_idx = {}
    
    logger.info(f"检测到类别数: {num_classes}")
    
    # 创建模型
    num_attributes = 6  # 6个属性
    model = get_model_with_attributes(model_type, num_classes, num_attributes)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info(f"模型加载完成，类别映射: {class_to_idx}")
    
    return model, class_to_idx


def evaluate_model(model, test_loader, device, class_names, attribute_names):
    """评估模型性能
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        class_names: 类别名称列表
        attribute_names: 属性名称列表
        
    Returns:
        results: 评估结果字典
    """
    logger.info("开始评估模型性能...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_attribute_predictions = []
    all_attribute_labels = []
    
    with torch.no_grad():
        for images, labels, attribute_labels in tqdm(test_loader, desc='评估中'):
            images, labels, attribute_labels = images.to(device), labels.to(device), attribute_labels.to(device)
            
            class_output, attribute_output = model(images)
            _, predicted = torch.max(class_output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attribute_predictions.extend(attribute_output.cpu().numpy())
            all_attribute_labels.extend(attribute_labels.cpu().numpy())
    
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
    
    # 计算属性预测准确率
    attribute_accuracy = []
    for i in range(len(attribute_names)):
        attr_labels = [lbl[i] for lbl in all_attribute_labels]
        attr_preds = [pred[i] for pred in all_attribute_predictions]
        # 计算属性预测的准确率（四舍五入到最近的整数）
        attr_acc = accuracy_score(attr_labels, [round(p) for p in attr_preds])
        attribute_accuracy.append(attr_acc)
        logger.info(f"{attribute_names[i]} 预测准确率: {attr_acc * 100:.2f}%")
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions,
        'labels': all_labels,
        'attribute_predictions': all_attribute_predictions,
        'attribute_labels': all_attribute_labels,
        'attribute_accuracy': attribute_accuracy
    }


def test_single_image(model, image_path, transform, device, class_names, attribute_names):
    """测试单张图像
    
    Args:
        model: 模型
        image_path: 图像路径
        transform: 数据变换
        device: 设备
        class_names: 类别名称列表
        attribute_names: 属性名称列表
    """
    logger.info(f"测试单张图像: {image_path}")
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 预处理
    if transform:
        image = transform(image).unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        class_output, attribute_output = model(image)
        
    # 分类预测
    class_prob = torch.softmax(class_output, dim=1)
    class_idx = torch.argmax(class_prob, dim=1).item()
    class_name = class_names[class_idx]
    class_confidence = class_prob[0, class_idx].item()
    
    # 属性预测
    attribute_preds = attribute_output.squeeze().cpu().numpy()
    attribute_results = {}
    for i, attr_name in enumerate(attribute_names):
        attribute_results[attr_name] = round(attribute_preds[i])
    
    logger.info(f"预测结果: {class_name} (置信度: {class_confidence:.4f})")
    logger.info(f"属性预测: {attribute_results}")


def main():
    parser = argparse.ArgumentParser(description='带有属性标签的角色分类模型测试脚本')
    parser.add_argument('--model-path', type=str, default='models/arona_plana_with_attributes/model_best.pth', help='模型文件路径')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2', 
                       choices=['mobilenet_v2', 'efficientnet_b0', 'resnet18'],
                       help='模型类型')
    parser.add_argument('--data-dir', type=str, default='../data/downloaded_images', help='数据目录')
    parser.add_argument('--annotations-file', type=str, default='../config/attribute_annotations.json', help='标注文件路径')
    parser.add_argument('--batch-size', type=int, default=8, help='批量大小')
    parser.add_argument('--output-dir', type=str, default='test_results_with_attributes', help='输出目录')
    parser.add_argument('--test-image', type=str, default=None, help='测试单张图像')
    parser.add_argument('--config', type=str, default=None, help='属性配置文件路径')
    
    args = parser.parse_args()
    
    # 如果未指定配置文件，尝试使用默认路径
    if args.config is None:
        possible_configs = [
            '../config/character_attributes.json',
            '../../config/character_attributes.json',
            os.path.join(os.path.dirname(__file__), '..', 'config', 'character_attributes.json')
        ]
        for config_path in possible_configs:
            if os.path.exists(config_path):
                args.config = config_path
                break
    
    if args.config:
        logger.info(f"使用属性配置文件: {args.config}")
    
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
    
    # 加载属性配置文件
    attribute_names = ['hair_color', 'eye_color', 'has_halo', 'outfit', 'hair_style', 'accessories']
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'attribute_order' in config:
                    attribute_names = config['attribute_order']
                    logger.info(f"从配置文件加载属性名称: {attribute_names}")
        except Exception as e:
            logger.warning(f"无法加载配置文件: {e}")
    
    # 创建数据集
    logger.info('加载数据集...')
    dataset = CharacterAttributeDataset(args.data_dir, args.annotations_file, transform=transform)
    
    # 获取类别名称
    class_names = list(dataset.class_to_idx.keys())
    logger.info(f'类别名称: {class_names}')
    
    # 如果指定了单张测试图像
    if args.test_image:
        test_single_image(model, args.test_image, transform, device, class_names, attribute_names)
        return
    
    # 创建数据加载器
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f'测试集大小: {len(dataset)}')
    
    # 评估模型
    results = evaluate_model(model, test_loader, device, class_names, attribute_names)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'test_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        # 只保存可序列化的数据
        save_results = {
            'accuracy': results['accuracy'],
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'],
            'attribute_accuracy': results['attribute_accuracy'],
            'attribute_names': attribute_names,
            'class_names': class_names
        }
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n测试结果已保存到: {output_path}")
    
    # 打印测试摘要
    logger.info("\n==================================================")
    logger.info("测试摘要")
    logger.info("==================================================")
    logger.info(f"分类准确率: {results['accuracy'] * 100:.2f}%")
    logger.info(f"测试图像数量: {len(dataset)}")
    logger.info(f"类别数量: {len(class_names)}")
    logger.info(f"属性数量: {len(attribute_names)}")
    logger.info("==================================================")


if __name__ == '__main__':
    main()
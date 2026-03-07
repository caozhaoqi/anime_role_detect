#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带有属性标签的角色分类推理脚本
"""

import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import logging
import json
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.models import get_model_with_attributes

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('predict_with_attributes')


def load_model(model_path, model_type='mobilenet_v2'):
    """加载训练好的模型"""
    from models import get_model_with_attributes
    
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
    num_attributes = 6
    model = get_model_with_attributes(model_type, num_classes, num_attributes)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info(f"模型加载完成")
    
    return model, class_to_idx


def load_attribute_config(config_path):
    """加载属性配置文件"""
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def predict_with_attributes(model, image_path, transform, device, class_to_idx, attribute_config=None):
    """预测图片的角色和属性
    
    Args:
        model: 模型
        image_path: 图片路径
        transform: 数据变换
        device: 设备
        class_to_idx: 类别到索引的映射
        attribute_config: 属性配置
    
    Returns:
        dict: 预测结果
    """
    logger.info(f"预测图片: {image_path}")
    
    # 加载并预处理图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        class_output, attribute_output = model(image_tensor)
    
    # 分类预测
    class_prob = torch.softmax(class_output, dim=1)
    class_idx = torch.argmax(class_prob, dim=1).item()
    class_confidence = class_prob[0, class_idx].item()
    
    # 索引到类别名的映射
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    predicted_class = idx_to_class.get(class_idx, "unknown")
    
    # 属性预测
    attribute_preds = attribute_output.squeeze().cpu().numpy()
    
    # 属性名称和映射
    attribute_order = ['hair_color', 'eye_color', 'has_halo', 'outfit', 'hair_style', 'accessories']
    attribute_mappings = {}
    
    if attribute_config:
        attribute_order = attribute_config.get('attribute_order', attribute_order)
        attribute_mappings = attribute_config.get('attribute_mappings', {})
    
    # 将预测的索引转换为属性名
    predicted_attributes = {}
    for i, attr_name in enumerate(attribute_order):
        pred_idx = round(attribute_preds[i])
        
        # 获取属性值的映射
        mapping = attribute_mappings.get(attr_name, {})
        # 反转映射：索引到属性名
        idx_to_attr = {v: k for k, v in mapping.items()}
        predicted_attributes[attr_name] = idx_to_attr.get(pred_idx, f"unknown_{pred_idx}")
    
    # 构建结果
    result = {
        "character": predicted_class,
        "confidence": class_confidence,
        "attributes": predicted_attributes,
        "attribute_confidences": attribute_preds.tolist()
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='带有属性标签的角色分类推理')
    parser.add_argument('--model-path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2', 
                       choices=['mobilenet_v2', 'efficientnet_b0', 'resnet18'],
                       help='模型类型')
    parser.add_argument('--image', type=str, required=True, help='要分类的图片路径')
    parser.add_argument('--config', type=str, default='../config/character_attributes.json', help='属性配置文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出结果文件路径')
    
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
    
    # 加载属性配置
    attribute_config = None
    if os.path.exists(args.config):
        attribute_config = load_attribute_config(args.config)
        logger.info(f"已加载属性配置: {args.config}")
    else:
        logger.warning(f"属性配置文件不存在: {args.config}")
    
    # 预测
    result = predict_with_attributes(
        model, args.image, transform, device, 
        class_to_idx, attribute_config
    )
    
    # 输出结果
    logger.info("\n" + "="*50)
    logger.info("预测结果:")
    logger.info("="*50)
    logger.info(f"角色: {result['character']}")
    logger.info(f"置信度: {result['confidence']:.4f}")
    logger.info("\n属性预测:")
    for attr, value in result['attributes'].items():
        logger.info(f"  {attr}: {value}")
    logger.info("="*50)
    
    # 保存结果到文件
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"\n结果已保存到: {args.output}")
    
    return result


if __name__ == '__main__':
    main()
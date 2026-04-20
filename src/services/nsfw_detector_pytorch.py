#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSFW检测服务（PyTorch实现）
负责检测图像是否包含敏感内容
基于 https://gitee.com/caozhaoqi/nsfw_model_img 项目实现
使用PyTorch实现NSFW检测
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from src.core.logging.global_logger import get_logger

logger = get_logger("nsfw_detector_pytorch")

# 全局模型缓存
_model = None

# 标签顺序
LABELS = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

# 预处理转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    """
    加载NSFW检测模型
    
    Returns:
        torch.nn.Module: 加载的模型
    """
    global _model
    
    if _model is not None:
        return _model
    
    try:
        # 加载预训练的MobileNet V2模型
        model = models.mobilenet_v2(pretrained=True)
        
        # 修改分类器以适应5个类别
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, len(LABELS))
        
        # 尝试加载权重（如果存在）
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'nsfw_model', 'nsfw_model.pth')
        model_path = os.path.normpath(model_path)
        
        if os.path.exists(model_path):
            logger.info(f"加载模型权重: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            logger.warning(f"模型权重文件不存在: {model_path}")
            logger.warning("使用预训练的MobileNet V2权重")
        
        # 设置为评估模式
        model.eval()
        
        logger.info("成功加载NSFW模型")
        _model = model
        return model
    except Exception as e:
        logger.error(f"加载NSFW模型失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        return None

def preprocess_image(image_source):
    """
    预处理图像
    
    Args:
        image_source: 图像路径或内存缓冲区
    
    Returns:
        torch.Tensor: 预处理后的图像
    """
    # 加载图像
    img = Image.open(image_source).convert('RGB')
    
    # 应用转换
    img_tensor = transform(img)
    
    # 添加批次维度
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def detect_nsfw_with_pytorch(image_source):
    """
    使用PyTorch进行NSFW检测
    
    Args:
        image_source: 图像路径或内存缓冲区(BytesIO)
    
    Returns:
        dict: NSFW检测结果
    """
    try:
        logger.info("使用PyTorch进行NSFW检测")
        
        # 加载模型
        model = load_model()
        if model is None:
            logger.error("模型加载失败")
            return None
        
        # 预处理图像
        try:
            img_tensor = preprocess_image(image_source)
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            return None
        
        # 进行预测
        try:
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                scores = probabilities[0].numpy()
            
            # 构建结果字典
            details = {}
            for i, label in enumerate(LABELS):
                details[label] = float(scores[i])
            
            # 确定最高概率的类别
            max_score = float(max(scores))
            max_index = np.argmax(scores)
            predicted_label = LABELS[max_index]
            
            # 判断是否为NSFW
            nsfw_categories = ['porn', 'sexy', 'hentai']
            is_nsfw = predicted_label in nsfw_categories and max_score > 0.5
            
            # 计算皮肤比例（模拟值，实际由模型决定）
            skin_ratio = 0.0
            if predicted_label in ['porn', 'sexy']:
                skin_ratio = max_score
            
            logger.info(f"NSFW检测完成，类别: {predicted_label}, 置信度: {max_score:.4f}, is_nsfw: {is_nsfw}")
            logger.info(f"NSFW检测详细结果: {details}")
            
            return {
                'is_nsfw': is_nsfw,
                'skin_ratio': float(skin_ratio),
                'details': details
            }
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            import traceback
            logger.error(f"异常堆栈: {traceback.format_exc()}")
            return None
    except Exception as e:
        logger.error(f"PyTorch检测失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        return None

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型加载器

负责加载训练好的模型
"""

import os
import torch
from src.core.logging.global_logger import get_logger

logger = get_logger("image_processor")


def load_trained_model(model_name):
    """
    加载训练好的模型
    
    Args:
        model_name: 模型名称
    
    Returns:
        tuple: (模型, 类别映射) 或 None
    """
    try:
        # 处理默认模型
        if model_name == "default":
            model_name = "efficientnet_b0"
        
        # 模型文件路径
        model_dir = f"models/{model_name}"
        # 优先使用 model_full.pth 文件，因为它包含完整的权重数据
        model_path = os.path.join(model_dir, "model_full.pth")
        # 如果 model_full.pth 不存在，尝试使用 model_best.pth
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model_best.pth")
        class_map_path = os.path.join(model_dir, "class_to_idx.json")
        
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            return None
        
        # 加载模型
        try:
            # 尝试使用 weights_only=False 加载完整模型
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        except Exception as e:
            # 如果失败，尝试使用 weights_only=True 加载
            logger.warning(f"使用 weights_only=False 加载模型失败: {e}")
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        
        # 加载类别映射
        class_to_idx = {}
        
        # 1. 首先尝试从模型文件中加载class_to_idx
        try:
            if 'class_to_idx' in checkpoint:
                class_to_idx = checkpoint['class_to_idx']
                logger.info(f"从模型文件加载了类别映射: {len(class_to_idx)} 个类别")
        except Exception as e:
            logger.warning(f"从模型文件加载类别映射失败: {e}")
        
        # 2. 尝试从class_to_idx.json文件加载
        if not class_to_idx and os.path.exists(class_map_path):
            try:
                import json
                with open(class_map_path, 'r', encoding='utf-8') as f:
                    class_to_idx = json.load(f)
                logger.info(f"从class_to_idx.json文件加载了类别映射: {len(class_to_idx)} 个类别")
            except Exception as e:
                logger.warning(f"从class_to_idx.json文件加载类别映射失败: {e}")
        
        # 3. 如果都失败，使用默认类别映射
        if not class_to_idx:
            class_to_idx = {"unknown": 0, "plana": 1, "other": 2}
            logger.warning(f"类别映射文件不存在: {class_map_path}，使用默认类别映射")
        
        # 检查是否是完整模型文件（包含模型结构）
        if isinstance(checkpoint, torch.nn.Module):
            logger.info(f"加载完整模型: {model_name}")
            model = checkpoint
            model.eval()
        else:
            # 创建模型实例
            import torchvision.models as models
            import torch.nn as nn
            
            # 根据模型名称选择合适的模型结构
            if 'resnet18' in model_name:
                model = models.resnet18(pretrained=False)
                model.fc = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(model.fc.in_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(p=0.1),
                    nn.Linear(512, len(class_to_idx))
                )
            elif 'mobilenet_v2' in model_name:
                model = models.mobilenet_v2(pretrained=False)
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(model.classifier[1].in_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(p=0.1),
                    nn.Linear(512, len(class_to_idx))
                )
            else:  # 默认使用EfficientNetB0模型结构
                model = models.efficientnet_b0(pretrained=False)
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(model.classifier[1].in_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(p=0.1),
                    nn.Linear(512, len(class_to_idx))
                )
            
            # 加载权重
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                logger.warning(f"模型文件中没有找到权重数据: {model_path}")
                return None
            
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        
        logger.info(f"成功加载模型: {model_name}")
        return model, class_to_idx
        
    except Exception as e:
        logger.error(f"加载训练模型失败: {e}")
        return None

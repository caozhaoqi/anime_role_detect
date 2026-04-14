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
        # 模型文件路径
        model_dir = f"models/trained/{model_name}"
        model_path = os.path.join(model_dir, "model.pth")
        class_map_path = os.path.join(model_dir, "class_to_idx.json")
        
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            return None
        
        if not os.path.exists(class_map_path):
            logger.warning(f"类别映射文件不存在: {class_map_path}")
            return None
        
        # 加载模型
        model = torch.load(model_path)
        model.eval()
        
        # 加载类别映射
        import json
        with open(class_map_path, 'r', encoding='utf-8') as f:
            class_to_idx = json.load(f)
        
        logger.info(f"成功加载模型: {model_name}")
        return model, class_to_idx
        
    except Exception as e:
        logger.error(f"加载训练模型失败: {e}")
        return None

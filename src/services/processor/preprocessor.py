#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像预处理器

负责图像的预处理操作
"""

from PIL import Image
from src.core.logging.global_logger import get_logger
from src.services.cache_service import get_image_transform

logger = get_logger("image_processor")


def preprocess_image(image_source):
    """
    预处理图像
    
    Args:
        image_source: 图像路径或内存缓冲区(BytesIO)
    
    Returns:
        预处理后的图像张量
    """
    # 延迟导入PyTorch
    import torch
    
    try:
        # 获取图像变换
        transform = get_image_transform()
        
        # 加载图像并转换
        img = Image.open(image_source).convert('RGB')
        
        # 限制图像大小，避免内存占用过高
        max_size = 448  # 模型需要的最小尺寸
        width, height = img.size
        if width > max_size or height > max_size:
            # 计算缩放比例
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"图像已缩放: {width}x{height} -> {new_width}x{new_height}")
        
        img = transform(img)
        img = img.unsqueeze(0)  # 添加批次维度
        
        return img
    except Exception as e:
        logger.error(f"预处理图像失败: {e}")
        raise

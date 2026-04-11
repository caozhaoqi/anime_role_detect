#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSFW检测服务
负责检测图像是否包含敏感内容
基于 https://gitee.com/caozhaoqi/nsfw_model_img 项目实现
使用基于规则的NSFW检测
"""

import os
import sys
from src.core.logging.global_logger import get_logger

logger = get_logger("nsfw_detector")


def detect_nsfw(image_path):
    """
    NSFW检测
    
    Args:
        image_path: 图像路径
    
    Returns:
        dict: NSFW检测结果
    """
    logger.info(f"开始NSFW检测: {image_path}")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            logger.error(f"图像文件不存在: {image_path}")
            return {
                'is_nsfw': False,
                'skin_ratio': 0.0
            }
        
        # 使用基于规则的NSFW检测
        logger.info(f"使用基于规则的NSFW检测: {image_path}")
        try:
            from PIL import Image
            import numpy as np
            
            # 加载图像
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # 定义皮肤颜色范围（HSL）
            # 这些值是基于经验设置的，可能需要根据实际情况调整
            def rgb_to_hsl(r, g, b):
                r, g, b = r/255.0, g/255.0, b/255.0
                max_val = max(r, g, b)
                min_val = min(r, g, b)
                h, s, l = 0, 0, (max_val + min_val) / 2
                
                if max_val != min_val:
                    d = max_val - min_val
                    s = d / (2 - max_val - min_val) if l > 0.5 else d / (max_val + min_val)
                    if max_val == r:
                        h = (g - b) / d + (6 if g < b else 0)
                    elif max_val == g:
                        h = (b - r) / d + 2
                    else:
                        h = (r - g) / d + 4
                    h /= 6
                
                return h, s, l
            
            # 统计皮肤颜色像素数量
            skin_pixels = 0
            total_pixels = img_array.shape[0] * img_array.shape[1]
            
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    r, g, b = img_array[i, j]
                    h, s, l = rgb_to_hsl(r, g, b)
                    
                    # 皮肤颜色的HSL范围
                    if (0.0 <= h <= 0.1 or 0.9 <= h <= 1.0) and 0.1 <= s <= 0.3 and 0.4 <= l <= 0.7:
                        skin_pixels += 1
            
            # 计算皮肤颜色像素比例
            skin_ratio = skin_pixels / total_pixels
            
            # 判断是否为NSFW
            is_nsfw = skin_ratio > 0.3
            
            # 构建检测结果
            scores = {
                'neutral': 1.0 - skin_ratio,
                'porn': skin_ratio * 0.5,
                'sexy': skin_ratio * 0.3,
                'hentai': skin_ratio * 0.1,
                'drawings': (1.0 - skin_ratio) * 0.5
            }
            
            # 确定最高概率的类别
            max_score = 0
            predicted_label = 'neutral'
            for label, score in scores.items():
                if score > max_score:
                    max_score = score
                    predicted_label = label
            
            # 输出详细日志
            logger.info(f"NSFW检测完成，类别: {predicted_label}, 置信度: {max_score:.4f}, is_nsfw: {is_nsfw}")
            logger.info(f"NSFW检测详细结果: {scores}")
            
            return {
                'is_nsfw': is_nsfw,
                'skin_ratio': float(skin_ratio),
                'details': scores
            }
        except Exception as e:
            logger.error(f"基于规则的NSFW检测失败: {e}")
            import traceback
            logger.error(f"异常堆栈: {traceback.format_exc()}")
    except Exception as e:
        logger.error(f"NSFW检测失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
    
    # 默认返回值
    logger.info("NSFW检测返回默认值")
    return {
        'is_nsfw': False,
        'skin_ratio': 0.0
    }

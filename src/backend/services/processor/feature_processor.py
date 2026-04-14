#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征处理器

负责处理图像特征
"""

from src.core.logging.global_logger import get_logger
from src.backend.services.model_loader import get_keypoint_detector, get_tagger

logger = get_logger("image_processor")


def process_image_features(image_source, content_type, attributes):
    """
    处理图像特征
    
    Args:
        image_source: 图像来源
        content_type: 内容类型
        attributes: 图像属性
    
    Returns:
        tuple: (文本检测结果, 关键点检测结果, AI预测角色)
    """
    text_detections = []
    keypoints = []
    ai_predicted_role = "unknown"
    
    try:
        # 检测关键点
        keypoint_detector = get_keypoint_detector()
        if keypoint_detector:
            try:
                keypoints = keypoint_detector.detect(image_source)
                logger.info(f"关键点检测完成，检测到 {len(keypoints)} 个关键点")
            except Exception as e:
                logger.error(f"关键点检测失败: {e}")
        
        # 标签检测
        tagger = get_tagger()
        if tagger:
            try:
                tags = tagger.tag(image_source)
                attributes.extend(tags)
                logger.info(f"标签检测完成，检测到 {len(tags)} 个标签")
            except Exception as e:
                logger.error(f"标签检测失败: {e}")
        
        # 这里可以添加其他特征处理逻辑
        
    except Exception as e:
        logger.error(f"处理图像特征失败: {e}")
    
    return text_detections, keypoints, ai_predicted_role

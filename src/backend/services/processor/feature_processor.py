#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征处理器

负责处理图像特征
"""

from src.core.logging.global_logger import get_logger
from src.backend.services.model_loader import get_keypoint_detector, get_tagger, get_role_predictor

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
        # 标签检测
        tag_names = []
        # 暂时禁用标签检测，避免锁竞争
        logger.info("标签检测已禁用")
        
        # AI 角色预测
        if tag_names:
            try:
                role_predictor = get_role_predictor()
                if role_predictor:
                    try:
                        ai_predicted_role = role_predictor.predict_role(tag_names)
                        logger.info(f"AI 角色预测完成: {ai_predicted_role}")
                    except Exception as e:
                        logger.error(f"AI 角色预测失败: {e}")
            except Exception as e:
                logger.error(f"加载角色预测模块失败: {e}")
        
        # 这里可以添加其他特征处理逻辑
        
    except Exception as e:
        logger.error(f"处理图像特征失败: {e}")
    
    return text_detections, keypoints, ai_predicted_role

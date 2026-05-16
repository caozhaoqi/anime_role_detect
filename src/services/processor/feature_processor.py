#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征处理器

负责处理图像特征
"""

from src.core.logging.global_logger import get_logger
from src.services.processor.model_loader import get_keypoint_detector, get_tagger, get_role_predictor
from src.core.ocr.easyocr_detector import detect_text as ocr_detect_text

logger = get_logger("image_processor")


def convert_numpy_types(obj):
    """
    将numpy类型转换为普通Python类型
    
    Args:
        obj: 要转换的对象
    
    Returns:
        转换后的对象
    """
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def generate_image_summary(text_detections, tags, role_info, attributes, nsfw_status):
    """
    生成图片内容的一句话总结
    
    Args:
        text_detections: 文本检测结果
        tags: 图像标签
        role_info: 角色信息
        attributes: 属性信息
        nsfw_status: NSFW状态
    
    Returns:
        str: 图片内容的一句话总结
    """
    try:
        # 转换numpy类型
        text_detections = convert_numpy_types(text_detections)
        tags = convert_numpy_types(tags)
        role_info = convert_numpy_types(role_info)
        attributes = convert_numpy_types(attributes)
        nsfw_status = convert_numpy_types(nsfw_status)
        
        summary_parts = []
        
        # 添加角色信息
        if role_info and hasattr(role_info, 'role') and role_info.role:
            summary_parts.append(f"图片中是{role_info.role}")
        elif role_info and isinstance(role_info, dict) and role_info.get('role'):
            summary_parts.append(f"图片中是{role_info.get('role')}")
        else:
            summary_parts.append("图片中是一个动漫角色")
        
        # 添加文本信息
        if text_detections and len(text_detections) > 0:
            texts = [detection['text'] for detection in text_detections if detection.get('text')]
            if texts:
                if len(texts) == 1:
                    summary_parts.append(f"，包含文字'{texts[0]}'")
                else:
                    summary_parts.append(f"，包含{len(texts)}处文字")
        
        # 添加属性信息
        if attributes and len(attributes) > 0:
            attribute_names = [attr['tag'] for attr in attributes if attr.get('tag')]
            if attribute_names:
                if len(attribute_names) <= 3:
                    summary_parts.append(f"，具有{', '.join(attribute_names)}等特征")
                else:
                    summary_parts.append(f"，具有{len(attribute_names)}个特征")
        
        # 添加NSFW警告
        if nsfw_status and (hasattr(nsfw_status, 'is_nsfw') and nsfw_status.is_nsfw):
            summary_parts.append("，包含敏感内容")
        elif nsfw_status and isinstance(nsfw_status, dict) and nsfw_status.get('is_nsfw'):
            summary_parts.append("，包含敏感内容")
        
        # 组合成一句话
        if summary_parts:
            summary = ''.join(summary_parts) + "。"
            # 确保总结长度适中
            if len(summary) > 100:
                summary = summary[:97] + "..."
            return summary
        else:
            return "这是一张动漫角色图片。"
    except Exception as e:
        logger.error(f"生成图片总结失败: {e}")
        return "这是一张动漫角色图片。"


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
        try:
            tagger = get_tagger()
            if tagger:
                tag_results = tagger(image_source)
                # 降低置信度阈值，确保至少保留10个标签
                tag_names = [tag['tag'] for tag in tag_results if tag['confidence'] > 0.3]
                
                # 确保至少有10个标签
                if len(tag_names) < 10 and len(tag_results) > 10:
                    # 如果标签不足10个，进一步降低阈值
                    tag_names = [tag['tag'] for tag in tag_results[:15]]
                
                logger.info(f"标签检测完成，提取到 {len(tag_names)} 个标签")
            else:
                logger.warning("标签检测器未加载")
        except Exception as e:
            logger.error(f"标签检测失败: {e}")
        
        # OCR 文字检测
        try:
            text_detections = ocr_detect_text(image_source)
            logger.info(f"OCR 文字检测完成，检测到 {len(text_detections)} 个文本区域")
        except Exception as e:
            logger.error(f"OCR 文字检测失败: {e}")
        
        # AI 角色预测
        if tag_names:
            try:
                role_predictor = get_role_predictor()
                if role_predictor:
                    try:
                        # 传递图像来源和标签给角色预测器
                        # image_source 可能是图像路径或图像字节数据
                        ai_predicted_role = role_predictor(image_path=image_source, tags=tag_names)
                        logger.info(f"AI 角色预测完成: {ai_predicted_role}")
                    except Exception as e:
                        logger.error(f"AI 角色预测失败: {e}")
            except Exception as e:
                logger.error(f"加载角色预测模块失败: {e}")
        
        # 这里可以添加其他特征处理逻辑
        
    except Exception as e:
        logger.error(f"处理图像特征失败: {e}")
    
    return text_detections, keypoints, ai_predicted_role

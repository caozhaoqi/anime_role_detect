#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类器工具

提供分类器相关的辅助函数
"""

import os
from PIL import Image
from src.core.logging.global_logger import get_logger
from src.core.classification.classifiers.general_classifier import GeneralClassification
from src.backend.services.model_loader import get_role_predictor

logger = get_logger("general_classification")


# 全局分类器实例
_classifier_instance = None


def get_classifier(index_path="role_index", model=None):
    """
    获取分类器实例

    Args:
        index_path: 索引路径
        model: 模型实例

    Returns:
        GeneralClassification: 分类器实例
    """
    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = GeneralClassification(index_path)

    return _classifier_instance


def classify_image(image_path, use_model=False):
    """
    分类图像

    Args:
        image_path: 图像路径
        use_model: 是否使用模型

    Returns:
        list: 分类结果
    """
    try:
        if use_model:
            # 使用AI角色预测器
            role_predictor = get_role_predictor()
            if role_predictor:
                # 从图像中提取标签
                from src.backend.services.processor.feature_processor import process_image_features

                attributes = []
                text_detections, keypoints, ai_predicted_role = process_image_features(
                    image_path, "image/jpeg", attributes
                )
                return [(ai_predicted_role or "unknown", 1.0)]

        # 使用CLIP分类器
        classifier = get_classifier()
        return classifier.classify(image_path)
    except Exception as e:
        logger.error(f"分类图像失败: {e}")
        return []


def classify_pil_image(pil_image, use_model=False):
    """
    分类PIL图像

    Args:
        pil_image: PIL图像
        use_model: 是否使用模型

    Returns:
        list: 分类结果
    """
    try:
        if use_model:
            # 使用AI角色预测器
            role_predictor = get_role_predictor()
            if role_predictor:
                # 从图像中提取标签
                import io

                img_io = io.BytesIO()
                pil_image.save(img_io, format="PNG")
                img_io.seek(0)

                from src.backend.services.processor.feature_processor import process_image_features

                attributes = []
                text_detections, keypoints, ai_predicted_role = process_image_features(
                    img_io, "image/png", attributes
                )
                return [(ai_predicted_role or "unknown", 1.0)]

        # 使用CLIP分类器
        classifier = get_classifier()
        return classifier.classify(pil_image)
    except Exception as e:
        logger.error(f"分类PIL图像失败: {e}")
        return []


def classify_image_ensemble(
    image_path, clip_weight=0.7, model_weight=0.3, confidence_threshold=0.6
):
    """
    使用集成方法分类图像

    Args:
        image_path: 图像路径
        clip_weight: CLIP权重
        model_weight: 模型权重
        confidence_threshold: 置信度阈值

    Returns:
        list: 分类结果
    """
    try:
        # 获取CLIP分类结果
        clip_results = classify_image(image_path, use_model=False)

        # 获取模型分类结果
        model_results = classify_image(image_path, use_model=True)

        # 集成结果
        combined_results = {}

        # 处理CLIP结果
        for role, score in clip_results:
            combined_results[role] = score * clip_weight

        # 处理模型结果
        for role, score in model_results:
            if role in combined_results:
                combined_results[role] += score * model_weight
            else:
                combined_results[role] = score * model_weight

        # 排序并过滤
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        filtered_results = [
            (role, score) for role, score in sorted_results if score >= confidence_threshold
        ]

        return filtered_results
    except Exception as e:
        logger.error(f"集成分类失败: {e}")
        return []


def build_index_from_directory(data_dir):
    """
    从目录构建索引

    Args:
        data_dir: 数据目录
    """
    try:
        classifier = get_classifier()

        # 遍历目录
        for role_name in os.listdir(data_dir):
            role_dir = os.path.join(data_dir, role_name)
            if os.path.isdir(role_dir):
                # 收集图像路径
                image_paths = []
                for filename in os.listdir(role_dir):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                        image_path = os.path.join(role_dir, filename)
                        image_paths.append(image_path)

                if image_paths:
                    logger.info(f"为角色 {role_name} 添加 {len(image_paths)} 张图像")
                    classifier.add_role(role_name, image_paths)
    except Exception as e:
        logger.error(f"构建索引失败: {e}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键点工具

提供关键点相关的辅助函数
"""

import cv2
import numpy as np
from src.core.logging.global_logger import get_logger
from .mediapipe_detector import MediaPipeKeypointDetector

logger = get_logger("mediapipe_keypoint_detector")

# 全局检测器实例
_detector_instance = None


def detect_keypoints(image, model_complexity=1):
    """
    检测关键点

    Args:
        image: 图像路径或 numpy 数组
        model_complexity: 模型复杂度

    Returns:
        list: 关键点列表
    """
    global _detector_instance

    if _detector_instance is None:
        _detector_instance = MediaPipeKeypointDetector(model_complexity=model_complexity)

    return _detector_instance.detect(image)


def draw_keypoints(image, keypoints):
    """
    绘制关键点

    Args:
        image: 图像路径或 numpy 数组
        keypoints: 关键点列表

    Returns:
        numpy.ndarray: 绘制后的图像
    """
    global _detector_instance

    if _detector_instance is None:
        _detector_instance = MediaPipeKeypointDetector()

    return _detector_instance.draw(image, keypoints)


def get_keypoint_distance(keypoint1, keypoint2):
    """
    计算两个关键点之间的距离

    Args:
        keypoint1: 第一个关键点
        keypoint2: 第二个关键点

    Returns:
        float: 距离
    """
    try:
        x1, y1, z1 = keypoint1["x"], keypoint1["y"], keypoint1["z"]
        x2, y2, z2 = keypoint2["x"], keypoint2["y"], keypoint2["z"]

        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
        return distance
    except Exception as e:
        logger.error(f"计算关键点距离失败: {e}")
        return 0.0


def get_keypoint_by_id(keypoints, keypoint_id):
    """
    根据ID获取关键点

    Args:
        keypoints: 关键点列表
        keypoint_id: 关键点ID

    Returns:
        dict: 关键点信息
    """
    for keypoint in keypoints:
        if keypoint["id"] == keypoint_id:
            return keypoint
    return None


def calculate_angle(keypoints, point1_id, point2_id, point3_id):
    """
    计算三个关键点之间的角度

    Args:
        keypoints: 关键点列表
        point1_id: 第一个关键点ID
        point2_id: 第二个关键点ID（顶点）
        point3_id: 第三个关键点ID

    Returns:
        float: 角度（度）
    """
    global _detector_instance

    if _detector_instance is None:
        _detector_instance = MediaPipeKeypointDetector()

    return _detector_instance.calculate_angle(keypoints, point1_id, point2_id, point3_id)


def get_detector():
    """
    获取关键点检测器实例

    Returns:
        MediaPipeKeypointDetector: 关键点检测器实例
    """
    global _detector_instance

    if _detector_instance is None:
        _detector_instance = MediaPipeKeypointDetector()

    return _detector_instance


def release_detector():
    """
    释放检测器实例
    """
    global _detector_instance

    if _detector_instance:
        _detector_instance.close()
        _detector_instance = None
        logger.info("关键点检测器已释放")

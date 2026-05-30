#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe 关键点检测器模块

负责使用 MediaPipe 进行关键点检测

注意：此模块已重构，所有实现已移至 detectors 子模块
"""

# 为了保持向后兼容性，从新的子模块导入所有类和函数
from .detectors import (
    MediaPipeKeypointDetector,
    detect_keypoints,
    draw_keypoints,
    get_keypoint_distance,
    get_keypoint_by_id,
    calculate_angle,
    get_detector,
    release_detector,
)

__all__ = [
    "MediaPipeKeypointDetector",
    "detect_keypoints",
    "draw_keypoints",
    "get_keypoint_distance",
    "get_keypoint_by_id",
    "calculate_angle",
    "get_detector",
    "release_detector",
]

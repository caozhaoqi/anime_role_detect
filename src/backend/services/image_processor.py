#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像处理服务
负责处理图像相关的操作

注意：此模块已重构，所有实现已移至 processor 子模块
"""

# 为了保持向后兼容性，从新的子模块导入所有函数
from .processor import (
    preprocess_image,
    process_with_model_service,
    process_with_local_model,
    process_with_trained_model,
    process_with_traditional_model,
    process_image_features,
    load_trained_model,
    process_single_image,
    process_batch_images,
)

# 类名列表
class_names = [
    "unknown", "plana", "other"
]

__all__ = [
    'preprocess_image',
    'process_with_model_service',
    'process_with_local_model',
    'process_with_trained_model',
    'process_with_traditional_model',
    'process_image_features',
    'load_trained_model',
    'process_single_image',
    'process_batch_images',
    'class_names',
]

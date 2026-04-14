#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理模块

负责数据和图像的预处理操作

注意：此模块已重构，所有实现已移至 preprocessors 子模块
"""

# 为了保持向后兼容性，从新的子模块导入所有类和函数
from .preprocessors import (
    ImagePreprocessor,
    DataPreprocessor,
    preprocess_image,
    preprocess_data,
    get_preprocessing_transform,
    get_image_preprocessor,
    get_data_preprocessor,
)

__all__ = [
    'ImagePreprocessor',
    'DataPreprocessor',
    'preprocess_image',
    'preprocess_data',
    'get_preprocessing_transform',
    'get_image_preprocessor',
    'get_data_preprocessor',
]

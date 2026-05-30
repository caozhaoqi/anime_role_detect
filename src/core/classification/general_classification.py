#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用分类模块

提供通用的分类功能

注意：此模块已重构，所有实现已移至 classifiers 子模块
"""

# 为了保持向后兼容性，从新的子模块导入所有类和函数
from .classifiers import (
    GeneralClassification,
    get_classifier,
    classify_image,
    classify_pil_image,
    classify_image_ensemble,
    build_index_from_directory,
)

__all__ = [
    "GeneralClassification",
    "get_classifier",
    "classify_image",
    "classify_pil_image",
    "classify_image_ensemble",
    "build_index_from_directory",
]

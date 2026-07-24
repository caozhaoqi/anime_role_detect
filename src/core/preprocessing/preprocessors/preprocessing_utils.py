#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理工具

提供预处理相关的辅助函数
"""

from torchvision import transforms
from src.core.logging import get_enhanced_logger as get_logger
from .image_preprocessor import ImagePreprocessor
from .data_preprocessor import DataPreprocessor

logger = get_logger("preprocessing")

# 全局预处理器实例
_image_preprocessor = None
_data_preprocessor = None


def preprocess_image(image, size=(224, 224)):
    """
    预处理图像

    Args:
        image: 图像路径或PIL图像
        size: 图像大小

    Returns:
        torch.Tensor: 预处理后的图像张量
    """
    global _image_preprocessor

    if _image_preprocessor is None or _image_preprocessor.size != size:
        _image_preprocessor = ImagePreprocessor(size=size)

    return _image_preprocessor.preprocess(image)


def preprocess_data(data, data_type="image"):
    """
    预处理数据

    Args:
        data: 数据
        data_type: 数据类型

    Returns:
        预处理后的数据
    """
    global _data_preprocessor

    if _data_preprocessor is None:
        _data_preprocessor = DataPreprocessor()

    if data_type == "csv":
        return _data_preprocessor.preprocess_csv(data)
    elif data_type == "json":
        return _data_preprocessor.preprocess_json(data)
    else:
        logger.warning(f"不支持的数据类型: {data_type}")
        return data


def get_preprocessing_transform(
    size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
):
    """
    获取预处理变换

    Args:
        size: 图像大小
        mean: 均值
        std: 标准差

    Returns:
        transforms.Compose: 预处理变换
    """
    return transforms.Compose(
        [transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )


def get_image_preprocessor(size=(224, 224)):
    """
    获取图像预处理器

    Args:
        size: 图像大小

    Returns:
        ImagePreprocessor: 图像预处理器
    """
    global _image_preprocessor

    if _image_preprocessor is None or _image_preprocessor.size != size:
        _image_preprocessor = ImagePreprocessor(size=size)

    return _image_preprocessor


def get_data_preprocessor():
    """
    获取数据预处理器

    Returns:
        DataPreprocessor: 数据预处理器
    """
    global _data_preprocessor

    if _data_preprocessor is None:
        _data_preprocessor = DataPreprocessor()

    return _data_preprocessor

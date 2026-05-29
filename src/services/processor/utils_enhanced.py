#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的文件验证工具 - 修复文件验证缺陷

修复的问题：
1. 验证文件是否真的是图像
2. 检查文件大小限制
3. 检查文件格式
4. 防止恶意文件上传
"""

import os
import io
from typing import Optional, Tuple
from PIL import Image

from src.core.logging.global_logger import get_logger

logger = get_logger("processor_utils")

# 允许的图像格式
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}

# 文件大小限制（50MB）
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', '52428800'))


def validate_file(content: bytes) -> Tuple[bool, Optional[str]]:
    """
    验证文件内容（增强版）
    
    Args:
        content: 文件内容（字节）
        
    Returns:
        tuple: (是否有效，错误信息)
    """
    # 检查内容是否为空
    if not content:
        return False, "文件内容为空"
    
    if len(content) == 0:
        return False, "文件大小为 0"
    
    # 检查文件大小
    if len(content) > MAX_FILE_SIZE:
        size_mb = len(content) / (1024 * 1024)
        return False, f"文件过大：{size_mb:.2f}MB，最大允许{MAX_FILE_SIZE / (1024 * 1024):.0f}MB"
    
    # 检查文件头（魔数）验证是否为图像
    if not is_valid_image_content(content):
        return False, "文件不是有效的图像格式"
    
    return True, None


def is_valid_image_content(content: bytes) -> bool:
    """
    验证内容是否为有效的图像文件
    
    Args:
        content: 文件内容（字节）
        
    Returns:
        bool: 是否为有效图像
    """
    try:
        # 使用 PIL 验证
        image = Image.open(io.BytesIO(content))
        image.verify()  # 验证完整性
        
        # 重新打开以检查格式
        image = Image.open(io.BytesIO(content))
        format_name = image.format
        
        if format_name is None:
            return False
        
        # 检查格式是否在允许列表中
        format_to_ext = {
            'JPEG': ['.jpg', '.jpeg'],
            'PNG': ['.png'],
            'GIF': ['.gif'],
            'WEBP': ['.webp'],
            'BMP': ['.bmp'],
        }
        
        # 检查是否支持该格式
        if format_name not in format_to_ext:
            logger.warning(f"不支持的图像格式：{format_name}")
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"图像验证失败：{e}")
        return False


def validate_file_extension(filename: str) -> Tuple[bool, Optional[str]]:
    """
    验证文件扩展名
    
    Args:
        filename: 文件名
        
    Returns:
        tuple: (是否有效，错误信息)
    """
    if not filename:
        return False, "文件名为空"
    
    ext = os.path.splitext(filename)[1].lower()
    
    if not ext:
        return False, "文件没有扩展名"
    
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"不支持的文件格式：{ext}，允许的格式：{', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, None


def get_file_extension(filename: str) -> str:
    """
    获取文件扩展名（带验证）
    
    Args:
        filename: 文件名
        
    Returns:
        str: 文件扩展名
        
    Raises:
        ValueError: 如果扩展名无效
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件扩展名：{ext}")
    
    return ext


def safe_read_image(content: bytes) -> Optional[Image.Image]:
    """
    安全读取图像
    
    Args:
        content: 文件内容（字节）
        
    Returns:
        PIL.Image 对象或 None
    """
    try:
        # 验证内容
        is_valid, error = validate_file(content)
        if not is_valid:
            logger.error(f"图像验证失败：{error}")
            return None
        
        # 读取图像
        image = Image.open(io.BytesIO(content))
        image.load()  # 完全加载图像到内存
        
        # 转换为 RGB 模式（处理 RGBA、P 等模式）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        logger.error(f"读取图像失败：{e}")
        return None

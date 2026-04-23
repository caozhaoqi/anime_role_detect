#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理器工具函数

提供通用的文件操作和工具函数
"""

import os
import tempfile
from src.core.logging.global_logger import get_logger

logger = get_logger("processor_utils")


def with_temp_file(content, suffix, callback):
    """
    创建临时文件并在处理完成后清理
    
    Args:
        content: 文件内容
        suffix: 文件后缀
        callback: 处理函数，接收临时文件路径作为参数
    
    Returns:
        callback的返回值
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        return callback(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"清理临时文件失败: {e}")


def validate_file(content):
    """
    验证文件内容
    
    Args:
        content: 文件内容
    
    Returns:
        bool: 文件是否有效
    """
    if not content:
        return False
    if len(content) == 0:
        return False
    return True


def get_file_extension(filename):
    """
    获取文件扩展名
    
    Args:
        filename: 文件名
    
    Returns:
        str: 文件扩展名
    """
    return os.path.splitext(filename)[1]

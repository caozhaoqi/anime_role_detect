#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志模块 
"""

import logging
import os
from typing import Optional

from src.core.config import get_config

config = get_config()

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    log_level = config.get("logging.level", "INFO")
    log_format = config.get("logging.format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    
    # 设置日志级别
    logger.setLevel(getattr(logging, log_level))
    
    # 创建格式化器
    formatter = logging.Formatter(log_format)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # 创建文件处理器（如果指定了文件）
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    
    # 禁止向上传播
    logger.propagate = False
    
    return logger

# 全局日志记录器
root_logger = get_logger("anime_role_detect")

def log_info(message: str, *args):
    """记录信息日志"""
    root_logger.info(message, *args)

def log_warning(message: str, *args):
    """记录警告日志"""
    root_logger.warning(message, *args)

def log_error(message: str, *args, exc_info=False):
    """记录错误日志"""
    root_logger.error(message, *args, exc_info=exc_info)

def log_debug(message: str, *args):
    """记录调试日志"""
    root_logger.debug(message, *args)
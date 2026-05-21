#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志模块
提供统一的日志配置和使用接口
"""

import logging
import logging.config
import os
from datetime import datetime
from typing import Optional

class LogConfig:
    """日志配置类"""
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_DIR = os.getenv("LOG_DIR", "logs")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    JSON_LOG_FORMAT = (
        '{"time": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", '
        '"module": "%(module)s", "line": %(lineno)d, "message": "%(message)s"}'
    )
    
    @classmethod
    def ensure_log_dir(cls):
        """确保日志目录存在"""
        if not os.path.exists(cls.LOG_DIR):
            os.makedirs(cls.LOG_DIR)
    
    @classmethod
    def get_log_file_path(cls, name: str) -> str:
        """获取日志文件路径"""
        timestamp = datetime.now().strftime("%Y%m%d")
        return os.path.join(cls.LOG_DIR, f"{name}_{timestamp}.log")

def setup_logging(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    设置统一日志配置
    
    Args:
        name: 日志器名称
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        配置好的日志器实例
    """
    LogConfig.ensure_log_dir()
    
    # 确定日志级别
    level = log_level or LogConfig.LOG_LEVEL
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level_int = level_map.get(level, logging.INFO)
    
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(log_level_int)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(LogConfig.LOG_FORMAT)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_int)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（按日期轮转）
    file_handler = logging.FileHandler(LogConfig.get_log_file_path(name))
    file_handler.setLevel(log_level_int)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # JSON格式文件处理器（用于结构化分析）
    json_formatter = logging.Formatter(LogConfig.JSON_LOG_FORMAT)
    json_file_handler = logging.FileHandler(LogConfig.get_log_file_path(f"{name}_json"))
    json_file_handler.setLevel(log_level_int)
    json_file_handler.setFormatter(json_formatter)
    logger.addHandler(json_file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    获取日志器实例
    
    Args:
        name: 日志器名称，通常使用 __name__
    
    Returns:
        日志器实例
    """
    return setup_logging(name)

class LoggerMixin:
    """日志混入类，方便其他类使用日志"""
    
    @property
    def logger(self) -> logging.Logger:
        """获取日志器"""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger

# 常用日志便捷函数
def debug(message: str, *args, **kwargs):
    """记录DEBUG级别日志"""
    get_logger("ardc").debug(message, *args, **kwargs)

def info(message: str, *args, **kwargs):
    """记录INFO级别日志"""
    get_logger("ardc").info(message, *args, **kwargs)

def warning(message: str, *args, **kwargs):
    """记录WARNING级别日志"""
    get_logger("ardc").warning(message, *args, **kwargs)

def error(message: str, *args, **kwargs):
    """记录ERROR级别日志"""
    get_logger("ardc").error(message, *args, **kwargs)

def critical(message: str, *args, **kwargs):
    """记录CRITICAL级别日志"""
    get_logger("ardc").critical(message, *args, **kwargs)

def exception(message: str, *args, exc_info=True, **kwargs):
    """记录异常日志"""
    get_logger("ardc").error(message, *args, exc_info=exc_info, **kwargs)

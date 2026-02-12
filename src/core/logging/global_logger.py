#!/usr/bin/env python3
"""
全局日志系统模块
统一管理系统日志，包括系统运行状态、模型推理结果、模型训练结果和错误日志
"""
import os
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger

class GlobalLogger:
    """
    全局日志系统类
    统一管理所有类型的日志
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        初始化全局日志系统
        
        Args:
            log_dir: 日志根目录
        """
        self.log_dir = Path(log_dir)
        self._setup_directories()
        self._configure_logger()
        
    def _setup_directories(self):
        """
        创建日志目录结构
        """
        # 按类型分目录
        self.system_log_dir = self.log_dir / "system"
        self.inference_log_dir = self.log_dir / "inference"
        self.training_log_dir = self.log_dir / "training"
        self.error_log_dir = self.log_dir / "error"
        
        # 创建所有目录
        for dir_path in [
            self.system_log_dir,
            self.inference_log_dir,
            self.training_log_dir,
            self.error_log_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
    def _configure_logger(self):
        """
        配置loguru日志系统
        """
        # 清除默认的日志处理器
        logger.remove()
        
        # 日志格式
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
        
        # 系统日志配置
        system_log_file = str(self.system_log_dir / "system_{time:YYYY-MM-DD}.log")
        logger.add(
            system_log_file,
            rotation="100 MB",  # 按大小轮转
            retention="7 days",  # 保留7天
            compression="zip",  # 压缩旧日志
            level="INFO",
            format=log_format,
            filter=lambda record: "system" in record["extra"]
        )
        
        # 推理日志配置
        inference_log_file = str(self.inference_log_dir / "inference_{time:YYYY-MM-DD}.log")
        logger.add(
            inference_log_file,
            rotation="100 MB",
            retention="14 days",  # 保留14天
            compression="zip",
            level="INFO",
            format=log_format,
            filter=lambda record: "inference" in record["extra"]
        )
        
        # 训练日志配置
        training_log_file = str(self.training_log_dir / "training_{time:YYYY-MM-DD}.log")
        logger.add(
            training_log_file,
            rotation="200 MB",  # 训练日志可能更大
            retention="30 days",  # 保留30天
            compression="zip",
            level="INFO",
            format=log_format,
            filter=lambda record: "training" in record["extra"]
        )
        
        # 错误日志配置
        error_log_file = str(self.error_log_dir / "error_{time:YYYY-MM-DD}.log")
        logger.add(
            error_log_file,
            rotation="50 MB",
            retention="30 days",
            compression="zip",
            level="ERROR",
            format=log_format,
            filter=lambda record: "error" in record["extra"] or record["level"].no >= logger.level("ERROR").no
        )
        
        # 控制台输出配置
        logger.add(
            sys.stdout,
            level="INFO",
            format=log_format
        )
        
    def get_logger(self, name: str = "global"):
        """
        获取日志记录器
        
        Args:
            name: 记录器名称
            
        Returns:
            loguru.Logger: 日志记录器
        """
        return logger.bind(name=name)
    
    def log_system(self, message: str, level: str = "info", **kwargs):
        """
        记录系统日志
        
        Args:
            message: 日志消息
            level: 日志级别
            **kwargs: 额外参数
        """
        log_method = getattr(logger.bind(system=True), level.lower())
        log_method(message, **kwargs)
    
    def log_inference(self, message: str, level: str = "info", **kwargs):
        """
        记录推理日志
        
        Args:
            message: 日志消息
            level: 日志级别
            **kwargs: 额外参数
        """
        log_method = getattr(logger.bind(inference=True), level.lower())
        log_method(message, **kwargs)
    
    def log_training(self, message: str, level: str = "info", **kwargs):
        """
        记录训练日志
        
        Args:
            message: 日志消息
            level: 日志级别
            **kwargs: 额外参数
        """
        log_method = getattr(logger.bind(training=True), level.lower())
        log_method(message, **kwargs)
    
    def log_error(self, message: str, level: str = "error", **kwargs):
        """
        记录错误日志
        
        Args:
            message: 日志消息
            level: 日志级别
            **kwargs: 额外参数
        """
        log_method = getattr(logger.bind(error=True), level.lower())
        log_method(message, **kwargs)

# 创建全局日志实例
global_logger = GlobalLogger()

# 便捷函数
def get_logger(name: str = "global"):
    """
    获取日志记录器
    
    Args:
        name: 记录器名称
        
    Returns:
        loguru.Logger: 日志记录器
    """
    return global_logger.get_logger(name)

def log_system(message: str, level: str = "info", **kwargs):
    """
    记录系统日志
    
    Args:
        message: 日志消息
        level: 日志级别
        **kwargs: 额外参数
    """
    global_logger.log_system(message, level, **kwargs)

def log_inference(message: str, level: str = "info", **kwargs):
    """
    记录推理日志
    
    Args:
        message: 日志消息
        level: 日志级别
        **kwargs: 额外参数
    """
    global_logger.log_inference(message, level, **kwargs)

def log_training(message: str, level: str = "info", **kwargs):
    """
    记录训练日志
    
    Args:
        message: 日志消息
        level: 日志级别
        **kwargs: 额外参数
    """
    global_logger.log_training(message, level, **kwargs)

def log_error(message: str, level: str = "error", **kwargs):
    """
    记录错误日志
    
    Args:
        message: 日志消息
        level: 日志级别
        **kwargs: 额外参数
    """
    global_logger.log_error(message, level, **kwargs)

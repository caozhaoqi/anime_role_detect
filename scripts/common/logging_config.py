#!/usr/bin/env python3
"""
统一的日志配置模块
- 提供标准化的日志配置
- 支持日志轮转
- 统一日志格式
- 提供日志工具函数
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime

# 获取项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# 将项目根目录添加到Python路径
sys.path.insert(0, PROJECT_ROOT)

# 全局配置
LOG_CONFIG = {
    'LOG_DIR': os.path.join(PROJECT_ROOT, 'logs'),
    'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'LOG_DATE_FORMAT': '%Y-%m-%d %H:%M:%S',
    'LOG_MAX_BYTES': 10 * 1024 * 1024,  # 10MB
    'LOG_BACKUP_COUNT': 5,  # 保留5个备份文件
}

def ensure_log_dir():
    """确保日志目录存在"""
    if not os.path.exists(LOG_CONFIG['LOG_DIR']):
        os.makedirs(LOG_CONFIG['LOG_DIR'])
        print(f"创建日志目录: {LOG_CONFIG['LOG_DIR']}")

    # 创建子目录
    sub_dirs = ['data_collection', 'training', 'evaluation', 'api', 'system']
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(LOG_CONFIG['LOG_DIR'], sub_dir)
        if not os.path.exists(sub_dir_path):
            os.makedirs(sub_dir_path)
            print(f"创建日志子目录: {sub_dir_path}")

def get_logger(name, log_file=None, level=logging.INFO):
    """
    获取标准化的日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件名，默认为None（使用默认日志文件）
        level: 日志级别，默认为INFO
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 确保日志目录存在
    ensure_log_dir()
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # 创建格式化器
        formatter = logging.Formatter(
            LOG_CONFIG['LOG_FORMAT'],
            datefmt=LOG_CONFIG['LOG_DATE_FORMAT']
        )
        console_handler.setFormatter(formatter)
        
        # 添加控制台处理器
        logger.addHandler(console_handler)
        
        # 创建文件处理器（带轮转）
        if log_file:
            # 确定日志文件路径
            if '/' in log_file or '\\' in log_file:
                # 完整路径
                log_file_path = log_file
            else:
                # 根据脚本名称确定子目录
                sub_dir = 'system'
                if 'data_collection' in name:
                    sub_dir = 'data_collection'
                elif 'training' in name:
                    sub_dir = 'training'
                elif 'evaluation' in name:
                    sub_dir = 'evaluation'
                elif 'api' in name:
                    sub_dir = 'api'
                
                log_file_path = os.path.join(LOG_CONFIG['LOG_DIR'], sub_dir, log_file)
            
            # 创建带轮转的文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=LOG_CONFIG['LOG_MAX_BYTES'],
                backupCount=LOG_CONFIG['LOG_BACKUP_COUNT'],
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

def log_function_call(logger):
    """
    日志装饰器，记录函数调用
    
    Args:
        logger: 日志记录器
    
    Returns:
        function: 装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"调用函数: {func.__name__}，参数: {args}, {kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"函数 {func.__name__} 执行成功，返回值: {result}")
                return result
            except Exception as e:
                logger.error(f"函数 {func.__name__} 执行失败: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator

# 测试函数
if __name__ == "__main__":
    # 测试日志配置
    logger = get_logger('test_logger', 'test.log')
    logger.debug('这是一条调试信息')
    logger.info('这是一条信息')
    logger.warning('这是一条警告')
    logger.error('这是一条错误')
    
    print("日志配置测试完成")

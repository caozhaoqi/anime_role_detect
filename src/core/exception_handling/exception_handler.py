#!/usr/bin/env python3
"""
异常处理类，实现更细粒度的错误处理
"""
import logging
import traceback
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


# 自定义异常类
class CollectionError(Exception):
    """采集异常基类"""
    pass


class NetworkError(CollectionError):
    """网络异常"""
    pass


class ParseError(CollectionError):
    """解析异常"""
    pass


class SaveError(CollectionError):
    """保存异常"""
    pass


class ValidationError(CollectionError):
    """验证异常"""
    pass


class ExceptionHandler:
    """异常处理类"""
    
    def __init__(self, max_retries=3, backoff_factor=0.5):
        """
        初始化异常处理器
        
        Args:
            max_retries: 最大重试次数
            backoff_factor: 退避因子
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.error_stats = {
            'network_errors': 0,
            'parse_errors': 0,
            'save_errors': 0,
            'validation_errors': 0,
            'other_errors': 0
        }
    
    def classify_error(self, error):
        """
        分类错误
        
        Args:
            error: 错误对象
            
        Returns:
            str: 错误类型
        """
        if isinstance(error, NetworkError) or 'request' in str(type(error).__name__).lower():
            return 'network'
        elif isinstance(error, ParseError):
            return 'parse'
        elif isinstance(error, SaveError):
            return 'save'
        elif isinstance(error, ValidationError):
            return 'validation'
        else:
            return 'other'
    
    def handle_error(self, error, context=None):
        """
        处理错误
        
        Args:
            error: 错误对象
            context: 错误上下文
        """
        error_type = self.classify_error(error)
        
        # 更新错误统计
        error_key = f'{error_type}_errors'
        if error_key in self.error_stats:
            self.error_stats[error_key] += 1
        else:
            self.error_stats['other_errors'] += 1
        
        # 记录错误信息
        error_msg = f"{error_type.upper()} ERROR: {str(error)}"
        if context:
            error_msg += f" (Context: {context})"
        
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        
        # 根据错误类型采取不同的处理策略
        if error_type == 'network':
            logger.warning("网络错误，建议检查网络连接或重试")
        elif error_type == 'parse':
            logger.warning("解析错误，建议检查页面结构是否变化")
        elif error_type == 'save':
            logger.warning("保存错误，建议检查存储权限或磁盘空间")
        elif error_type == 'validation':
            logger.warning("验证错误，建议检查数据质量")
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs):
        """
        带退避策略的重试
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
        """
        import time
        
        last_error = None
        
        for retry in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                # 计算退避时间
                backoff_time = self.backoff_factor * (2 ** retry) + (time.time() % 1)
                
                error_type = self.classify_error(e)
                logger.warning(f"尝试 {retry + 1}/{self.max_retries} 失败: {error_type} error - {str(e)}")
                logger.warning(f"等待 {backoff_time:.2f} 秒后重试...")
                
                # 处理错误
                self.handle_error(e, {
                    'retry': retry,
                    'function': func.__name__
                })
                
                # 退避等待
                time.sleep(backoff_time)
        
        # 重试失败，抛出最后一个错误
        logger.error(f"达到最大重试次数 {self.max_retries}，操作失败")
        raise last_error
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Optional[Any]:
        """
        安全执行函数，捕获所有异常
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果，出错时返回None
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, {
                'function': func.__name__
            })
            return None
    
    def get_error_stats(self):
        """
        获取错误统计信息
        
        Returns:
            dict: 错误统计
        """
        return self.error_stats
    
    def reset_error_stats(self):
        """
        重置错误统计信息
        """
        self.error_stats = {
            'network_errors': 0,
            'parse_errors': 0,
            'save_errors': 0,
            'validation_errors': 0,
            'other_errors': 0
        }
        logger.info("错误统计已重置")


# 全局异常处理器实例
exception_handler = ExceptionHandler()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志模块
提供统一的日志配置和使用接口，支持结构化日志输出和请求追踪
"""

import logging
import logging.config
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler

from ardc.config import settings

class ContextFormatter(logging.Formatter):
    """上下文格式化器 - 为缺失的上下文字段提供默认值"""
    
    def format(self, record):
        # 确保所有上下文字段都有默认值
        if not hasattr(record, 'request_id'):
            record.request_id = "N/A"
        if not hasattr(record, 'user'):
            record.user = "anonymous"
        if not hasattr(record, 'client_ip'):
            record.client_ip = "unknown"
        if not hasattr(record, 'trace_id'):
            record.trace_id = "N/A"
        if not hasattr(record, 'span_id'):
            record.span_id = "N/A"
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器 - 输出 JSON 格式日志"""
    
    def format(self, record):
        # 获取基本字段
        log_entry = {
            "time": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
            "message": record.getMessage(),
            "request_id": getattr(record, 'request_id', "N/A"),
            "user": getattr(record, 'user', "anonymous"),
            "client_ip": getattr(record, 'client_ip', "unknown"),
            "trace_id": getattr(record, 'trace_id', "N/A"),
            "span_id": getattr(record, 'span_id', "N/A"),
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    设置统一日志配置
    
    Args:
        name: 日志器名称
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        配置好的日志器实例
    """
    # 使用统一配置
    log_settings = settings.log
    
    # 确保日志目录存在
    os.makedirs(log_settings.dir, exist_ok=True)
    
    # 确定日志级别
    level = log_level or log_settings.level
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
    text_formatter = ContextFormatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d "
        "- [Req:%(request_id)s] [Trace:%(trace_id)s] [User:%(user)s] [IP:%(client_ip)s] %(message)s"
    )
    
    structured_formatter = StructuredFormatter()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_int)
    console_handler.setFormatter(text_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（按大小轮转）
    file_path = os.path.join(log_settings.dir, f"{name}.log")
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=log_settings.max_file_size_mb * 1024 * 1024,
        backupCount=log_settings.backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level_int)
    file_handler.setFormatter(text_formatter)
    logger.addHandler(file_handler)
    
    # JSON格式文件处理器（用于结构化分析）
    json_file_path = os.path.join(log_settings.dir, f"{name}_json.log")
    json_file_handler = RotatingFileHandler(
        json_file_path,
        maxBytes=log_settings.max_file_size_mb * 1024 * 1024,
        backupCount=log_settings.backup_count,
        encoding="utf-8"
    )
    json_file_handler.setLevel(log_level_int)
    json_file_handler.setFormatter(structured_formatter)
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


class RequestContextFilter(logging.Filter):
    """请求上下文过滤器 - 注入请求ID、用户、客户端IP等信息，支持追踪"""
    
    # 默认值
    DEFAULT_REQUEST_ID = "N/A"
    DEFAULT_USER = "anonymous"
    DEFAULT_CLIENT_IP = "unknown"
    DEFAULT_TRACE_ID = "N/A"
    DEFAULT_SPAN_ID = "N/A"
    
    def __init__(self):
        super().__init__()
        self.request_id = self.DEFAULT_REQUEST_ID
        self.user = self.DEFAULT_USER
        self.client_ip = self.DEFAULT_CLIENT_IP
        self.trace_id = self.DEFAULT_TRACE_ID
        self.span_id = self.DEFAULT_SPAN_ID
    
    def set_context(self, request_id: str = None, user: str = None, client_ip: str = None, trace_id: str = None, span_id: str = None):
        """设置请求上下文"""
        self.request_id = request_id or self.DEFAULT_REQUEST_ID
        self.user = user or self.DEFAULT_USER
        self.client_ip = client_ip or self.DEFAULT_CLIENT_IP
        self.trace_id = trace_id or self.DEFAULT_TRACE_ID
        self.span_id = span_id or self.DEFAULT_SPAN_ID
    
    def filter(self, record):
        """注入上下文信息到日志记录"""
        record.request_id = self.request_id
        record.user = self.user
        record.client_ip = self.client_ip
        record.trace_id = self.trace_id
        record.span_id = self.span_id
        return True


class RequestLogger:
    """请求日志器 - 用于记录请求的详细信息"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_request(self, method: str, path: str, client_ip: str, user: str = "anonymous", status_code: int = 200, duration: float = 0.0, size: int = 0):
        """记录请求日志"""
        self.logger.info(
            f"Request: {method} {path} | Status: {status_code} | Duration: {duration:.2f}ms | Size: {size} bytes | User: {user} | IP: {client_ip}"
        )
    
    def log_error(self, method: str, path: str, client_ip: str, error: Exception, user: str = "anonymous"):
        """记录请求错误日志"""
        self.logger.error(
            f"Request Error: {method} {path} | User: {user} | IP: {client_ip} | Error: {str(error)}",
            exc_info=True
        )


# 创建全局上下文过滤器
_request_context_filter = RequestContextFilter()


def get_request_logger(name: str = "ardc.request") -> RequestLogger:
    """获取请求日志器"""
    logger = get_logger(name)
    # 添加请求上下文过滤器
    logger.addFilter(_request_context_filter)
    return RequestLogger(logger)


def set_request_context(request_id: str = None, user: str = None, client_ip: str = None, trace_id: str = None, span_id: str = None):
    """设置当前请求上下文（供中间件调用），支持追踪信息"""
    _request_context_filter.set_context(request_id, user, client_ip, trace_id, span_id)

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

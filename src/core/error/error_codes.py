#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误码定义
"""

class ErrorCode:
    """
    错误码定义
    """
    # 系统错误
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    
    # 业务错误
    INVALID_REQUEST = "INVALID_REQUEST"
    MISSING_PARAMS = "MISSING_PARAMS"
    INVALID_PARAMS = "INVALID_PARAMS"
    
    # 资源错误
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_EXISTS = "RESOURCE_EXISTS"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    
    # 认证错误
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    
    # 业务逻辑错误
    NSFW_DETECTION_FAILED = "NSFW_DETECTION_FAILED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    IMAGE_PROCESSING_FAILED = "IMAGE_PROCESSING_FAILED"
    CLASSIFICATION_FAILED = "CLASSIFICATION_FAILED"
    
    # 错误信息映射
    ERROR_MESSAGES = {
        # 系统错误
        INTERNAL_ERROR: "内部服务器错误",
        SERVICE_UNAVAILABLE: "服务不可用",
        TIMEOUT_ERROR: "请求超时",
        
        # 业务错误
        INVALID_REQUEST: "无效的请求",
        MISSING_PARAMS: "缺少必要参数",
        INVALID_PARAMS: "参数无效",
        
        # 资源错误
        RESOURCE_NOT_FOUND: "资源不存在",
        RESOURCE_EXISTS: "资源已存在",
        RESOURCE_LIMIT_EXCEEDED: "资源限制已超出",
        
        # 认证错误
        UNAUTHORIZED: "未授权",
        FORBIDDEN: "禁止访问",
        TOKEN_EXPIRED: "令牌已过期",
        
        # 业务逻辑错误
        NSFW_DETECTION_FAILED: "NSFW检测失败",
        MODEL_LOAD_FAILED: "模型加载失败",
        IMAGE_PROCESSING_FAILED: "图像处理失败",
        CLASSIFICATION_FAILED: "分类失败"
    }
    
    @classmethod
    def get_message(cls, error_code: str) -> str:
        """
        获取错误信息
        
        Args:
            error_code: 错误码
            
        Returns:
            错误信息
        """
        return cls.ERROR_MESSAGES.get(error_code, "未知错误")

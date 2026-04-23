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
    
    # 文件错误
    FILE_UPLOAD_FAILED = "FILE_UPLOAD_FAILED"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    
    # 反馈错误
    FEEDBACK_SUBMISSION_FAILED = "FEEDBACK_SUBMISSION_FAILED"
    INVALID_FEEDBACK_TYPE = "INVALID_FEEDBACK_TYPE"
    
    # 错误信息映射
    ERROR_MESSAGES = {
        # 系统错误
        INTERNAL_ERROR: "内部服务器错误，请稍后重试",
        SERVICE_UNAVAILABLE: "服务暂时不可用，请稍后重试",
        TIMEOUT_ERROR: "请求超时，请检查网络连接",
        
        # 业务错误
        INVALID_REQUEST: "无效的请求参数",
        MISSING_PARAMS: "缺少必要的请求参数",
        INVALID_PARAMS: "参数格式无效，请检查后重试",
        
        # 资源错误
        RESOURCE_NOT_FOUND: "请求的资源不存在",
        RESOURCE_EXISTS: "资源已存在",
        RESOURCE_LIMIT_EXCEEDED: "资源使用已超出限制",
        
        # 认证错误
        UNAUTHORIZED: "未授权访问",
        FORBIDDEN: "禁止访问该资源",
        TOKEN_EXPIRED: "认证令牌已过期",
        
        # 业务逻辑错误
        NSFW_DETECTION_FAILED: "内容检测失败",
        MODEL_LOAD_FAILED: "模型加载失败，请检查模型配置",
        IMAGE_PROCESSING_FAILED: "图像处理失败，请确保上传的是有效图像",
        CLASSIFICATION_FAILED: "角色分类失败",
        
        # 文件错误
        FILE_UPLOAD_FAILED: "文件上传失败",
        INVALID_FILE_TYPE: "无效的文件类型，请上传图像文件",
        FILE_TOO_LARGE: "文件大小超出限制",
        
        # 反馈错误
        FEEDBACK_SUBMISSION_FAILED: "反馈提交失败，请稍后重试",
        INVALID_FEEDBACK_TYPE: "无效的反馈类型"
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

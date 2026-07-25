#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误处理系统
整合统一异常处理（src/core/exceptions.py）
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from src.core.error.error_codes import ErrorCode
from src.core.logging import get_enhanced_logger as get_logger

try:
    from src.core.exceptions import (
        BaseError,
        NotFoundError,
        InvalidParamsError,
        MissingParamsError,
        UnauthorizedError,
        ForbiddenError,
        format_error_response,
    )
    HAS_UNIFIED_EXCEPTIONS = True
except ImportError:
    HAS_UNIFIED_EXCEPTIONS = False
    logger.warning("统一异常处理模块不可用")

logger = get_logger("error_handler")


class AppException(HTTPException):
    """
    应用异常类
    """

    def __init__(self, status_code: int, error_code: str, detail: str = None):
        """
        初始化异常

        Args:
            status_code: HTTP状态码
            error_code: 错误码
            detail: 错误详情
        """
        if detail is None:
            detail = ErrorCode.get_message(error_code)

        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code

    def to_dict(self) -> dict:
        """
        转换为字典

        Returns:
            异常信息字典
        """
        return {"error": {"code": self.error_code, "message": self.detail}}


async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理器

    Args:
        request: 请求对象
        exc: 异常对象

    Returns:
        JSON响应
    """
    # 记录异常
    logger.error(f"全局异常: {exc}")

    # 处理统一异常（优先）
    if HAS_UNIFIED_EXCEPTIONS and isinstance(exc, BaseError):
        status_code, error_dict = format_error_response(exc)
        return JSONResponse(status_code=status_code, content={"error": error_dict})

    # 处理应用异常
    if isinstance(exc, AppException):
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    # 处理HTTP异常
    elif isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "HTTP_ERROR", "message": exc.detail}},
        )

    # 处理其他异常
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": ErrorCode.INTERNAL_ERROR,
                    "message": ErrorCode.get_message(ErrorCode.INTERNAL_ERROR),
                }
            },
        )


def create_error_response(error_code: str, status_code: int = 400, detail: str = None):
    """
    创建错误响应

    Args:
        error_code: 错误码
        status_code: HTTP状态码
        detail: 错误详情

    Returns:
        JSON响应
    """
    if detail is None:
        detail = ErrorCode.get_message(error_code)

    return JSONResponse(
        status_code=status_code, content={"error": {"code": error_code, "message": detail}}
    )


def raise_app_error(error_code: str, status_code: int = 400, detail: str = None):
    """
    抛出应用异常

    Args:
        error_code: 错误码
        status_code: HTTP状态码
        detail: 错误详情
    """
    raise AppException(status_code=status_code, error_code=error_code, detail=detail)

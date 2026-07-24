"""
统一异常处理系统
错误码设计，提供标准化的异常处理

功能：
1. 标准化异常类
2. 错误码系统
3. 异常处理器装饰器
4. API 错误响应格式化
"""

import traceback
from typing import Any, Dict, Optional, Tuple

ERROR_CODES = {
    'SUCCESS': (0, '成功'),
    'UNKNOWN_ERROR': (-1, '未知错误'),
    'INVALID_PARAMS': (1000, '参数无效'),
    'MISSING_PARAMS': (1001, '缺少参数'),
    'INVALID_FORMAT': (1002, '格式错误'),
    'NOT_FOUND': (2000, '资源不存在'),
    'ALREADY_EXISTS': (2001, '资源已存在'),
    'ACCESS_DENIED': (3000, '访问被拒绝'),
    'UNAUTHORIZED': (3001, '未授权'),
    'FORBIDDEN': (3002, '禁止访问'),
    'RATE_LIMITED': (3003, '请求频率超限'),
    'DATABASE_ERROR': (4000, '数据库错误'),
    'CACHE_ERROR': (4001, '缓存错误'),
    'INTERNAL_ERROR': (5000, '内部错误'),
    'SERVICE_UNAVAILABLE': (5001, '服务不可用'),
    'TIMEOUT': (5002, '请求超时'),
    'DEPENDENCY_ERROR': (6000, '依赖错误'),
    'MODEL_ERROR': (7000, '模型错误'),
    'VALIDATION_ERROR': (8000, '验证错误'),
}


class BaseError(Exception):
    """
    基础异常类

    Args:
        code: 错误码
        message: 错误消息
        details: 错误详情
        cause: 原始异常
    """

    def __init__(
        self,
        code: int = -1,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        self.code = code
        self.message = message or ERROR_CODES.get(code, (code, '未知错误'))[1]
        self.details = details or {}
        self.cause = cause
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            错误信息字典
        """
        result = {
            'code': self.code,
            'message': self.message,
        }
        if self.details:
            result['details'] = self.details
        if self.cause:
            result['cause'] = str(self.cause)
        return result

    def __repr__(self):
        return f"<{self.__class__.__name__}: code={self.code}, message='{self.message}'>"


class InvalidParamsError(BaseError):
    """参数无效异常"""

    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(ERROR_CODES['INVALID_PARAMS'][0], message, details)


class MissingParamsError(BaseError):
    """缺少参数异常"""

    def __init__(self, params: Optional[list] = None, message: Optional[str] = None):
        details = {'missing_params': params} if params else None
        super().__init__(ERROR_CODES['MISSING_PARAMS'][0], message, details)


class InvalidFormatError(BaseError):
    """格式错误异常"""

    def __init__(self, field: Optional[str] = None, expected_format: Optional[str] = None):
        details = {'field': field, 'expected_format': expected_format} if field else None
        message = f"字段 {field} 格式错误，期望格式: {expected_format}" if field else None
        super().__init__(ERROR_CODES['INVALID_FORMAT'][0], message, details)


class NotFoundError(BaseError):
    """资源不存在异常"""

    def __init__(self, resource: Optional[str] = None, id: Optional[Any] = None):
        details = {'resource': resource, 'id': id} if resource else None
        message = f"{resource} 不存在，ID: {id}" if resource else None
        super().__init__(ERROR_CODES['NOT_FOUND'][0], message, details)


class AlreadyExistsError(BaseError):
    """资源已存在异常"""

    def __init__(self, resource: Optional[str] = None, id: Optional[Any] = None):
        details = {'resource': resource, 'id': id} if resource else None
        message = f"{resource} 已存在，ID: {id}" if resource else None
        super().__init__(ERROR_CODES['ALREADY_EXISTS'][0], message, details)


class AccessDeniedError(BaseError):
    """访问被拒绝异常"""

    def __init__(self, reason: Optional[str] = None):
        details = {'reason': reason} if reason else None
        super().__init__(ERROR_CODES['ACCESS_DENIED'][0], reason, details)


class UnauthorizedError(BaseError):
    """未授权异常"""

    def __init__(self, message: Optional[str] = None):
        super().__init__(ERROR_CODES['UNAUTHORIZED'][0], message)


class ForbiddenError(BaseError):
    """禁止访问异常"""

    def __init__(self, reason: Optional[str] = None):
        details = {'reason': reason} if reason else None
        super().__init__(ERROR_CODES['FORBIDDEN'][0], reason, details)


class RateLimitedError(BaseError):
    """请求频率超限异常"""

    def __init__(self, retry_after: Optional[int] = None):
        details = {'retry_after': retry_after} if retry_after else None
        message = f"请求频率超限，请 {retry_after} 秒后重试" if retry_after else None
        super().__init__(ERROR_CODES['RATE_LIMITED'][0], message, details)


class DatabaseError(BaseError):
    """数据库错误异常"""

    def __init__(self, message: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(ERROR_CODES['DATABASE_ERROR'][0], message, cause=cause)


class CacheError(BaseError):
    """缓存错误异常"""

    def __init__(self, message: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(ERROR_CODES['CACHE_ERROR'][0], message, cause=cause)


class InternalError(BaseError):
    """内部错误异常"""

    def __init__(self, message: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(ERROR_CODES['INTERNAL_ERROR'][0], message, cause=cause)


class ServiceUnavailableError(BaseError):
    """服务不可用异常"""

    def __init__(self, service: Optional[str] = None):
        details = {'service': service} if service else None
        message = f"服务 {service} 不可用" if service else None
        super().__init__(ERROR_CODES['SERVICE_UNAVAILABLE'][0], message, details)


class TimeoutError(BaseError):
    """请求超时异常"""

    def __init__(self, timeout: Optional[int] = None):
        details = {'timeout': timeout} if timeout else None
        message = f"请求超时，超时时间: {timeout} 秒" if timeout else None
        super().__init__(ERROR_CODES['TIMEOUT'][0], message, details)


class DependencyError(BaseError):
    """依赖错误异常"""

    def __init__(self, dependency: Optional[str] = None, message: Optional[str] = None):
        details = {'dependency': dependency} if dependency else None
        message = message or f"依赖 {dependency} 不可用" if dependency else None
        super().__init__(ERROR_CODES['DEPENDENCY_ERROR'][0], message, details)


class ModelError(BaseError):
    """模型错误异常"""

    def __init__(self, message: Optional[str] = None, model: Optional[str] = None, cause: Optional[Exception] = None):
        details = {'model': model} if model else None
        super().__init__(ERROR_CODES['MODEL_ERROR'][0], message, details, cause)


class ValidationError(BaseError):
    """验证错误异常"""

    def __init__(self, field: Optional[str] = None, message: Optional[str] = None):
        details = {'field': field} if field else None
        message = message or f"字段 {field} 验证失败" if field else None
        super().__init__(ERROR_CODES['VALIDATION_ERROR'][0], message, details)


class ErrorHandler:
    """
    异常处理器

    Usage:
        handler = ErrorHandler()

        @handler.catch
        def my_function():
            # 可能抛出异常的代码
            pass

        # 或者使用装饰器
        @error_handler
        def my_function():
            pass
    """

    def __init__(self, logger=None):
        self.logger = logger

    def catch(self, func):
        """
        装饰器：捕获异常并转换为标准化错误

        Args:
            func: 被装饰的函数

        Returns:
            装饰后的函数
        """

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BaseError as e:
                if self.logger:
                    self.logger.error(f"业务异常: {e.code} - {e.message}")
                raise
            except Exception as e:
                if self.logger:
                    self.logger.error(f"系统异常: {e}\n{traceback.format_exc()}")
                raise InternalError(str(e), cause=e)

        return wrapper

    def catch_with_result(self, func):
        """
        装饰器：捕获异常并返回错误结果（不抛出）

        Args:
            func: 被装饰的函数

        Returns:
            (success, result/error) 元组
        """

        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return True, result
            except BaseError as e:
                if self.logger:
                    self.logger.error(f"业务异常: {e.code} - {e.message}")
                return False, e.to_dict()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"系统异常: {e}\n{traceback.format_exc()}")
                return False, InternalError(str(e), cause=e).to_dict()

        return wrapper


def error_handler(func=None, logger=None):
    """
    装饰器：错误处理

    Args:
        func: 被装饰的函数
        logger: 日志记录器

    Returns:
        装饰后的函数

    Usage:
        @error_handler
        def my_function():
            pass

        @error_handler(logger=my_logger)
        def my_function():
            pass
    """

    handler = ErrorHandler(logger=logger)

    if func is None:
        return handler.catch

    return handler.catch(func)


def error_handler_with_result(func=None, logger=None):
    """
    装饰器：错误处理（返回结果模式）

    Args:
        func: 被装饰的函数
        logger: 日志记录器

    Returns:
        装饰后的函数

    Usage:
        @error_handler_with_result
        def my_function():
            pass

        success, result = my_function()
        if not success:
            # 处理错误
            pass
    """

    handler = ErrorHandler(logger=logger)

    if func is None:
        return handler.catch_with_result

    return handler.catch_with_result(func)


def format_error_response(error: Exception) -> Tuple[int, Dict[str, Any]]:
    """
    格式化错误响应

    Args:
        error: 异常对象

    Returns:
        (HTTP状态码, 错误响应字典)
    """

    status_code_map = {
        InvalidParamsError: 400,
        MissingParamsError: 400,
        InvalidFormatError: 400,
        ValidationError: 400,
        NotFoundError: 404,
        AlreadyExistsError: 409,
        AccessDeniedError: 403,
        UnauthorizedError: 401,
        ForbiddenError: 403,
        RateLimitedError: 429,
        DatabaseError: 500,
        CacheError: 500,
        InternalError: 500,
        ServiceUnavailableError: 503,
        TimeoutError: 504,
        DependencyError: 500,
        ModelError: 500,
    }

    if isinstance(error, BaseError):
        status_code = status_code_map.get(type(error), 500)
        return status_code, error.to_dict()

    status_code = 500
    error_dict = InternalError(str(error), cause=error).to_dict()
    return status_code, error_dict


def get_error_code(code: int) -> Tuple[int, str]:
    """
    获取错误码信息

    Args:
        code: 错误码

    Returns:
        (错误码, 错误描述)
    """
    return ERROR_CODES.get(code, (code, '未知错误'))


def is_success_code(code: int) -> bool:
    """
    判断是否为成功码

    Args:
        code: 错误码

    Returns:
        是否成功
    """
    return code == ERROR_CODES['SUCCESS'][0]
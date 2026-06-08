#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超时工具模块 - 为关键操作添加超时机制，避免长时间阻塞
"""

import asyncio
import functools
import signal
from typing import Callable, Any, Optional
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import asynccontextmanager

from src.core.logging.global_logger import get_logger

logger = get_logger("timeout_utils")


class TimeoutException(Exception):
    """超时异常"""
    pass


def timeout_decorator(seconds: float, error_message: Optional[str] = None):
    """
    同步函数超时装饰器

    Args:
        seconds: 超时时间（秒）
        error_message: 超时错误消息

    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            def timeout_handler(signum, frame):
                raise TimeoutException(
                    error_message or f"函数 {func.__name__} 执行超时（{seconds}秒）"
                )

            # 设置信号处理器
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)

            try:
                # 设置超时
                signal.alarm(int(seconds))
                result = func(*args, **kwargs)
                return result
            except TimeoutException as e:
                logger.error(f"函数超时: {e}")
                raise
            finally:
                # 恢复旧的信号处理器
                signal.signal(signal.SIGALRM, old_handler)
                signal.alarm(0)  # 取消超时

        return wrapper
    return decorator


async def async_timeout_decorator(seconds: float, error_message: Optional[str] = None):
    """
    异步函数超时装饰器

    Args:
        seconds: 超时时间（秒）
        error_message: 超时错误消息

    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                error_msg = error_message or f"异步函数 {func.__name__} 执行超时（{seconds}秒）"
                logger.error(f"异步函数超时: {error_msg}")
                raise TimeoutException(error_msg)

        return wrapper
    return decorator


@asynccontextmanager
async def timeout_context(seconds: float, error_message: Optional[str] = None):
    """
    异步超时上下文管理器

    Args:
        seconds: 超时时间（秒）
        error_message: 超时错误消息

    Yields:
        None

    Raises:
        TimeoutException: 超时时抛出
    """
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        error_msg = error_message or f"操作超时（{seconds}秒）"
        logger.error(f"上下文超时: {error_msg}")
        raise TimeoutException(error_msg)


def run_with_timeout(func: Callable, timeout: float, *args, **kwargs) -> Any:
    """
    在线程池中运行函数并设置超时

    Args:
        func: 要执行的函数
        timeout: 超时时间（秒）
        *args: 函数参数
        **kwargs: 函数关键字参数

    Returns:
        函数执行结果

    Raises:
        TimeoutException: 超时时抛出
    """
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            error_msg = f"函数 {func.__name__} 在线程池中执行超时（{timeout}秒）"
            logger.error(error_msg)
            future.cancel()  # 尝试取消任务
            raise TimeoutException(error_msg)


async def run_async_with_timeout(coro, timeout: float, error_message: Optional[str] = None):
    """
    运行异步协程并设置超时

    Args:
        coro: 异步协程
        timeout: 超时时间（秒）
        error_message: 超时错误消息

    Returns:
        协程执行结果

    Raises:
        TimeoutException: 超时时抛出
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        error_msg = error_message or f"异步操作超时（{timeout}秒）"
        logger.error(error_msg)
        raise TimeoutException(error_msg)


class TimeoutManager:
    """超时管理器 - 统一管理各种操作的超时配置"""

    # 默认超时配置
    DEFAULT_TIMEOUTS = {
        # HTTP请求超时
        "http_request": 10.0,
        "http_upload": 30.0,
        "http_download": 60.0,

        # 模型处理超时
        "model_inference": 30.0,
        "model_preprocessing": 10.0,
        "model_postprocessing": 5.0,

        # 数据库操作超时
        "db_query": 5.0,
        "db_write": 10.0,
        "db_transaction": 30.0,

        # 外部服务调用超时
        "external_service": 15.0,
        "model_service": 20.0,
        "search_service": 10.0,

        # 文件操作超时
        "file_read": 5.0,
        "file_write": 10.0,
        "file_upload": 30.0,

        # 其他操作超时
        "default": 10.0,
    }

    def __init__(self, custom_timeouts: Optional[dict] = None):
        """
        初始化超时管理器

        Args:
            custom_timeouts: 自定义超时配置
        """
        self.timeouts = self.DEFAULT_TIMEOUTS.copy()
        if custom_timeouts:
            self.timeouts.update(custom_timeouts)

        logger.info(f"超时管理器初始化完成，超时配置: {self.timeouts}")

    def get_timeout(self, operation: str) -> float:
        """
        获取指定操作的超时时间

        Args:
            operation: 操作名称

        Returns:
            超时时间（秒）
        """
        return self.timeouts.get(operation, self.timeouts["default"])

    def set_timeout(self, operation: str, timeout: float):
        """
        设置指定操作的超时时间

        Args:
            operation: 操作名称
            timeout: 超时时间（秒）
        """
        self.timeouts[operation] = timeout
        logger.info(f"更新超时配置: {operation} = {timeout}秒")

    async def execute_with_timeout(
        self,
        operation: str,
        coro,
        error_message: Optional[str] = None
    ) -> Any:
        """
        执行异步操作并应用超时配置

        Args:
            operation: 操作名称
            coro: 异步协程
            error_message: 超时错误消息

        Returns:
            操作结果

        Raises:
            TimeoutException: 超时时抛出
        """
        timeout = self.get_timeout(operation)
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            error_msg = error_message or f"操作 {operation} 超时（{timeout}秒）"
            logger.error(error_msg)
            raise TimeoutException(error_msg)


# 创建全局超时管理器实例
timeout_manager = TimeoutManager()


# 便捷函数
async def execute_with_timeout(operation: str, coro, error_message: Optional[str] = None) -> Any:
    """
    使用全局超时管理器执行异步操作

    Args:
        operation: 操作名称
        coro: 异步协程
        error_message: 超时错误消息

    Returns:
        操作结果

    Raises:
        TimeoutException: 超时时抛出
    """
    return await timeout_manager.execute_with_timeout(operation, coro, error_message)


def get_timeout(operation: str) -> float:
    """
    获取指定操作的超时时间

    Args:
        operation: 操作名称

    Returns:
        超时时间（秒）
    """
    return timeout_manager.get_timeout(operation)


def set_timeout(operation: str, timeout: float):
    """
    设置指定操作的超时时间

    Args:
        operation: 操作名称
        timeout: 超时时间（秒）
    """
    timeout_manager.set_timeout(operation, timeout)


# 使用示例
if __name__ == "__main__":
    import time

    # 示例1: 使用装饰器
    @timeout_decorator(2, "示例函数超时")
    def slow_function():
        time.sleep(5)
        return "完成"

    try:
        result = slow_function()
    except TimeoutException as e:
        print(f"捕获到超时异常: {e}")

    # 示例2: 使用异步装饰器
    @async_timeout_decorator(2, "异步示例函数超时")
    async def async_slow_function():
        await asyncio.sleep(5)
        return "异步完成"

    async def test_async():
        try:
            result = await async_slow_function()
        except TimeoutException as e:
            print(f"捕获到异步超时异常: {e}")

    asyncio.run(test_async())

    # 示例3: 使用上下文管理器
    async def test_context():
        try:
            async with timeout_context(2, "上下文操作超时"):
                await asyncio.sleep(5)
        except TimeoutException as e:
            print(f"捕获到上下文超时异常: {e}")

    asyncio.run(test_context())

    # 示例4: 使用超时管理器
    async def test_manager():
        try:
            result = await execute_with_timeout("model_inference", asyncio.sleep(5))
        except TimeoutException as e:
            print(f"捕获到管理器超时异常: {e}")

    asyncio.run(test_manager())
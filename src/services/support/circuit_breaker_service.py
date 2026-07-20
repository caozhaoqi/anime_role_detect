import time
import asyncio
import statistics
from typing import Optional, Callable, Any, Dict

from src.core.logging.global_logger import get_logger

logger = get_logger("circuit_breaker_service")


class CircuitState:
    """熔断器状态"""

    CLOSED = "closed"  # 正常状态
    OPEN = "open"  # 熔断状态
    HALF_OPEN = "half_open"  # 半开状态


class CircuitBreaker:
    """熔断器实现"""

    def __init__(
        self,
        failure_threshold: int = 5,  # 失败阈值
        recovery_timeout: int = 30,  # 恢复超时时间（秒）
        success_threshold: int = 3,  # 半开状态下的成功阈值
        name: str = "default",
    ):
        """初始化熔断器"""
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.request_times = []
        self.error_rates = []

    def _should_open(self) -> bool:
        """判断是否应该打开熔断器"""
        return self.failure_count >= self.failure_threshold

    def _should_close(self) -> bool:
        """判断是否应该关闭熔断器"""
        return self.success_count >= self.success_threshold

    def _is_recovered(self) -> bool:
        """判断是否已恢复"""
        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _reset(self):
        """重置熔断器"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.request_times = []

    def _record_request_time(self, duration: float):
        """记录请求时间"""
        self.request_times.append(duration)
        if len(self.request_times) > 100:
            self.request_times.pop(0)

    def _calculate_error_rate(self, success: bool):
        """计算错误率"""
        self.error_rates.append(0 if success else 1)
        if len(self.error_rates) > 100:
            self.error_rates.pop(0)

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """执行函数并应用熔断策略"""
        start_time = time.time()

        # 检查熔断器状态
        if self.state == CircuitState.OPEN:
            if self._is_recovered():
                # 进入半开状态
                self.state = CircuitState.HALF_OPEN
                logger.info(f"熔断器 {self.name} 从 OPEN 状态转为 HALF_OPEN 状态")
            else:
                # 熔断状态，直接抛出异常
                logger.warning(f"熔断器 {self.name} 处于 OPEN 状态，拒绝请求")
                raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is open")

        try:
            # 执行函数
            result = func(*args, **kwargs)

            # 记录成功
            duration = time.time() - start_time
            self._record_request_time(duration)
            self._calculate_error_rate(True)

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self._should_close():
                    # 关闭熔断器
                    self._reset()
                    logger.info(f"熔断器 {self.name} 从 HALF_OPEN 状态转为 CLOSED 状态")
            else:
                # 正常状态，重置失败计数
                self.failure_count = 0

            return result
        except Exception as e:
            # 记录失败
            duration = time.time() - start_time
            self._record_request_time(duration)
            self._calculate_error_rate(False)

            if self.state == CircuitState.HALF_OPEN:
                # 半开状态下失败，重新打开
                self.state = CircuitState.OPEN
                self.last_failure_time = time.time()
                logger.warning(f"熔断器 {self.name} 从 HALF_OPEN 状态转为 OPEN 状态")
            else:
                # 正常状态下失败，增加失败计数
                self.failure_count += 1
                if self._should_open():
                    # 打开熔断器
                    self.state = CircuitState.OPEN
                    self.last_failure_time = time.time()
                    logger.warning(f"熔断器 {self.name} 从 CLOSED 状态转为 OPEN 状态")

            raise

    def get_state(self) -> str:
        """获取当前状态"""
        return self.state

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "avg_request_time": statistics.mean(self.request_times) if self.request_times else 0,
            "error_rate": sum(self.error_rates) / len(self.error_rates) if self.error_rates else 0,
        }


class CircuitBreakerOpenException(Exception):
    """熔断器打开异常"""

    pass


class CircuitBreakerService:
    """熔断器服务"""

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """获取或创建熔断器"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
        return self.circuit_breakers[name]

    def execute_with_fallback(
        self, name: str, func: Callable, fallback: Callable, *args, **kwargs
    ) -> Any:
        """执行函数，失败时使用降级策略（同步版本）"""
        circuit_breaker = self.get_circuit_breaker(name)

        try:
            return circuit_breaker.execute(func, *args, **kwargs)
        except Exception as e:
            logger.warning(f"执行 {name} 失败，使用降级策略: {e}")
            return fallback(*args, **kwargs)

    async def execute_with_fallback_async(
        self, name: str, func: Callable, fallback: Callable, *args, **kwargs
    ) -> Any:
        """执行异步函数，失败时使用降级策略（异步版本）"""
        circuit_breaker = self.get_circuit_breaker(name)

        try:
            # 检查熔断器状态（同步操作）
            if circuit_breaker.state == CircuitState.OPEN:
                if circuit_breaker._is_recovered():
                    circuit_breaker.state = CircuitState.HALF_OPEN
                    logger.info(f"熔断器 {name} 从 OPEN 状态转为 HALF_OPEN 状态")
                else:
                    logger.warning(f"熔断器 {name} 处于 OPEN 状态，拒绝请求")
                    raise CircuitBreakerOpenException(f"Circuit breaker {name} is open")

            # 直接 await async 函数（不通过线程池，避免 coroutine 泄漏）
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start_time

            # 记录成功
            circuit_breaker._record_request_time(duration)
            circuit_breaker._calculate_error_rate(True)

            if circuit_breaker.state == CircuitState.HALF_OPEN:
                circuit_breaker.success_count += 1
                if circuit_breaker._should_close():
                    circuit_breaker._reset()
                    logger.info(f"熔断器 {name} 从 HALF_OPEN 状态转为 CLOSED 状态")
            else:
                circuit_breaker.failure_count = 0

            return result
        except Exception as e:
            # 记录失败
            circuit_breaker.failure_count += 1
            circuit_breaker.last_failure_time = time.time()

            if circuit_breaker._should_open():
                circuit_breaker.state = CircuitState.OPEN
                logger.warning(f"熔断器 {name} 打开")

            logger.warning(f"执行 {name} 失败，使用降级策略: {e}")
            # 降级函数可能是同步或异步的
            if asyncio.iscoroutinefunction(fallback):
                return await fallback(*args, **kwargs)
            else:
                return await asyncio.to_thread(fallback, *args, **kwargs)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有熔断器的统计信息"""
        return {name: cb.get_stats() for name, cb in self.circuit_breakers.items()}


# 全局熔断器服务实例
_circuit_breaker_service: Optional[CircuitBreakerService] = None


def get_circuit_breaker_service() -> CircuitBreakerService:
    """获取熔断器服务实例"""
    global _circuit_breaker_service
    if _circuit_breaker_service is None:
        _circuit_breaker_service = CircuitBreakerService()
    return _circuit_breaker_service


def init_circuit_breaker_service():
    """初始化熔断器服务"""
    global _circuit_breaker_service
    if _circuit_breaker_service is None:
        _circuit_breaker_service = CircuitBreakerService()
        logger.info("熔断器服务初始化完成")
    return _circuit_breaker_service


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """获取熔断器"""
    return get_circuit_breaker_service().get_circuit_breaker(name, **kwargs)


def execute_with_fallback(name: str, func: Callable, fallback: Callable, *args, **kwargs) -> Any:
    """执行函数，失败时使用降级策略（同步版本）"""
    return get_circuit_breaker_service().execute_with_fallback(
        name, func, fallback, *args, **kwargs
    )


async def execute_with_fallback_async(name: str, func: Callable, fallback: Callable, *args, **kwargs) -> Any:
    """执行异步函数，失败时使用降级策略（异步版本）"""
    return await get_circuit_breaker_service().execute_with_fallback_async(
        name, func, fallback, *args, **kwargs
    )


def get_all_circuit_breaker_stats() -> Dict[str, Dict[str, Any]]:
    """获取所有熔断器的统计信息"""
    return get_circuit_breaker_service().get_all_stats()
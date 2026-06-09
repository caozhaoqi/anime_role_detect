import time
from fastapi import Request, Response
from prometheus_client import Gauge, Summary

from src.core.logging.global_logger import get_logger
from src.services.support.monitoring_service import get_monitoring_service

logger = get_logger("monitoring_middleware")

# 定义监控指标

# 活跃请求数
ACTIVE_REQUESTS = Gauge("anime_role_detect_active_requests", "Number of active requests")

# 请求大小
REQUEST_SIZE = Summary("anime_role_detect_request_size_bytes", "Request size in bytes")

# 响应大小
RESPONSE_SIZE = Summary("anime_role_detect_response_size_bytes", "Response size in bytes")

# 服务启动时间
SERVICE_START_TIME = Gauge(
    "anime_role_detect_service_start_time", "Service start time in seconds since epoch"
)

SERVICE_START_TIME.set(time.time())


async def monitoring_middleware(request: Request, call_next):
    """监控中间件"""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()

    try:
        # 处理请求
        # 注意：不读取 request.body()，避免干扰文件上传
        response = await call_next(request)

        # 记录响应大小
        if hasattr(response, "body") and callable(getattr(response, "body", None)):
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            RESPONSE_SIZE.observe(len(body))

        # 记录请求信息
        endpoint = request.url.path
        method = request.method
        status = response.status_code

        # 使用监控服务记录请求
        monitoring = get_monitoring_service()
        monitoring.record_request(endpoint, method, status)

        duration = time.time() - start_time
        monitoring.record_request_duration(endpoint, method, duration)

        return response
    except Exception as e:
        # 记录错误请求
        endpoint = request.url.path
        method = request.method

        # 使用监控服务记录错误
        monitoring = get_monitoring_service()
        monitoring.record_request(endpoint, method, 500)
        monitoring.record_error("exception", endpoint)

        duration = time.time() - start_time
        monitoring.record_request_duration(endpoint, method, duration)

        raise
    finally:
        ACTIVE_REQUESTS.dec()


def get_service_monitor():
    """获取服务监控信息"""
    from src.services.support.monitoring_service import get_monitoring_service
    from src.services.cache_service import get_cache_stats

    monitoring = get_monitoring_service()
    cache_stats = get_cache_stats()

    return {
        "service": {
            "start_time": SERVICE_START_TIME._value.get(),
            "uptime": time.time() - SERVICE_START_TIME._value.get(),
        },
        "cache": cache_stats,
        "metrics": {"active_requests": ACTIVE_REQUESTS._value.get()},
    }

import time
from fastapi import Request, Response
from starlette.responses import StreamingResponse
from prometheus_client import Gauge, Summary

from src.core.logging import get_enhanced_logger as get_logger
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
        response = await call_next(request)

        # P1-2: 包装 body_iterator 透传 chunk 的同时增量统计大小
        # 不再全量消费 body + 重建 Response，避免大响应体的内存拷贝开销
        original_body_iterator = response.body_iterator

        async def _wrap_body_iterator(body_iterator):
            """透传 chunk 并统计总字节数"""
            total = 0
            async for chunk in body_iterator:
                total += len(chunk)
                yield chunk
            RESPONSE_SIZE.observe(total)

        response = StreamingResponse(
            _wrap_body_iterator(original_body_iterator),
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

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

from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from typing import Optional

from src.core.logging.global_logger import get_logger

logger = get_logger("monitoring_service")

# 定义监控指标

# 请求计数
REQUEST_COUNT = Counter(
    "anime_role_detect_requests_total", "Total number of requests", ["endpoint", "method", "status"]
)

# 请求处理时间
REQUEST_DURATION = Histogram(
    "anime_role_detect_request_duration_seconds",
    "Request processing time in seconds",
    ["endpoint", "method"],
)

# 模型推理时间
MODEL_INFERENCE_TIME = Histogram(
    "anime_role_detect_model_inference_seconds", "Model inference time in seconds", ["model_name"]
)

# 缓存命中率
CACHE_HIT_RATE = Gauge("anime_role_detect_cache_hit_rate", "Cache hit rate", ["cache_type"])

# 缓存统计
CACHE_REQUESTS = Counter(
    "anime_role_detect_cache_requests_total",
    "Total number of cache requests",
    ["cache_type", "result"],  # result: hit or miss
)

# 错误计数
ERROR_COUNT = Counter(
    "anime_role_detect_errors_total", "Total number of errors", ["error_type", "endpoint"]
)

# 内存使用
MEMORY_USAGE = Gauge("anime_role_detect_memory_usage_bytes", "Memory usage in bytes")

# 服务健康状态
SERVICE_HEALTH = Gauge(
    "anime_role_detect_service_health", "Service health status (1=healthy, 0=unhealthy)"
)

# 模型服务状态
MODEL_SERVICE_STATUS = Gauge(
    "anime_role_detect_model_service_status", "Model service status (1=healthy, 0=unhealthy)"
)

# 图像处理统计
IMAGE_PROCESSING_COUNT = Counter(
    "anime_role_detect_image_processing_total",
    "Total number of image processing operations",
    ["operation", "result"],
)

# 标签提取统计
TAG_EXTRACTION_COUNT = Counter(
    "anime_role_detect_tag_extraction_total",
    "Total number of tag extraction operations",
    ["method", "result"],
)

# 多角色检测统计
MULTI_ROLE_DETECTION_COUNT = Counter(
    "anime_role_detect_multi_role_detection_total",
    "Total number of multi-role detection operations",
    ["result"],
)


class MonitoringService:
    """监控服务"""

    _instance: Optional["MonitoringService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "initialized") and self.initialized:
            return

        self.initialized = True
        self.start_time = time.time()

        # 设置初始健康状态
        SERVICE_HEALTH.set(1)
        MODEL_SERVICE_STATUS.set(0)

        logger.info("监控服务初始化完成")

    def record_request(self, endpoint: str, method: str, status: int):
        """记录请求"""
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()

    def record_request_duration(self, endpoint: str, method: str, duration: float):
        """记录请求处理时间"""
        REQUEST_DURATION.labels(endpoint=endpoint, method=method).observe(duration)

    def record_model_inference(self, model_name: str, duration: float):
        """记录模型推理时间"""
        MODEL_INFERENCE_TIME.labels(model_name=model_name).observe(duration)

    def record_cache_request(self, cache_type: str, hit: bool):
        """记录缓存请求"""
        result = "hit" if hit else "miss"
        CACHE_REQUESTS.labels(cache_type=cache_type, result=result).inc()

    def update_cache_hit_rate(self, cache_type: str, hit_rate: float):
        """更新缓存命中率"""
        CACHE_HIT_RATE.labels(cache_type=cache_type).set(hit_rate)

    def record_error(self, error_type: str, endpoint: str):
        """记录错误"""
        ERROR_COUNT.labels(error_type=error_type, endpoint=endpoint).inc()

    def update_memory_usage(self, usage: int):
        """更新内存使用"""
        MEMORY_USAGE.set(usage)

    def set_service_health(self, healthy: bool):
        """设置服务健康状态"""
        SERVICE_HEALTH.set(1 if healthy else 0)

    def set_model_service_status(self, healthy: bool):
        """设置模型服务状态"""
        MODEL_SERVICE_STATUS.set(1 if healthy else 0)

    def record_image_processing(self, operation: str, success: bool):
        """记录图像处理"""
        result = "success" if success else "failure"
        IMAGE_PROCESSING_COUNT.labels(operation=operation, result=result).inc()

    def record_tag_extraction(self, method: str, success: bool):
        """记录标签提取"""
        result = "success" if success else "failure"
        TAG_EXTRACTION_COUNT.labels(method=method, result=result).inc()

    def record_multi_role_detection(self, success: bool):
        """记录多角色检测"""
        result = "success" if success else "failure"
        MULTI_ROLE_DETECTION_COUNT.labels(result=result).inc()


# 全局监控服务实例
_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """获取监控服务实例"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service


def init_monitoring_service():
    """初始化监控服务"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
        logger.info("监控服务初始化完成")
    return _monitoring_service


# 装饰器


def monitor_request(endpoint: str):
    """监控请求装饰器"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            monitoring = get_monitoring_service()

            try:
                result = await func(*args, **kwargs)
                status = 200
                monitoring.record_request(endpoint, "POST", status)
                return result
            except Exception as e:
                status = 500
                monitoring.record_request(endpoint, "POST", status)
                monitoring.record_error("exception", endpoint)
                raise
            finally:
                duration = time.time() - start_time
                monitoring.record_request_duration(endpoint, "POST", duration)

        return wrapper

    return decorator


def monitor_model_inference(model_name: str):
    """监控模型推理装饰器"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            monitoring = get_monitoring_service()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                monitoring.record_model_inference(model_name, duration)

        return wrapper

    return decorator
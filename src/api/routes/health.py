"""
健康检查和监控路由
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

import time
from datetime import datetime
from fastapi import APIRouter

from src.core.logging.global_logger import get_logger

logger = get_logger("api.routes.health")

router = APIRouter()


@router.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "Anime Role Detect API",
        "version": "1.0.0",
        "timestamp": time.time(),
    }


@router.get("/live")
async def liveness_check():
    """K8s liveness 端点 - 进程存活检查，只要进程在就返回 OK"""
    return {"status": "alive"}


@router.get("/ready")
async def readiness_check():
    """K8s readiness 端点 - 服务就绪检查，只有核心依赖就绪才返回 OK"""
    checks = {"api": True}
    ready = True

    # 检查模型是否已加载
    try:
        from src.services.processor.model_loader import _model_cache
        checks["model"] = bool(_model_cache)
        if not _model_cache:
            ready = False
    except Exception:
        checks["model"] = False
        ready = False

    # 检查数据库是否可用
    try:
        from src.core.config.database import _local_session
        checks["database"] = _local_session is not None
    except Exception:
        checks["database"] = False

    # 检查缓存是否可用（降级模式也算就绪）
    try:
        from src.services.cache_service import get_cache_manager
        cache_manager = get_cache_manager()
        checks["cache"] = cache_manager is not None
    except Exception:
        checks["cache"] = False

    status_code = 200 if ready else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )


@router.get("/api/health/detailed")
async def detailed_health_check():
    """详细健康检查"""
    health_status = {"status": "healthy", "timestamp": datetime.now().isoformat(), "services": {}}
    overall_healthy = True

    try:
        from src.services.cache_service import get_cache_manager
        cache_manager = get_cache_manager()
        cache_stats = cache_manager.get_stats()
        cache_available = cache_stats.get("available", True)
        health_status["services"]["cache"] = {
            "status": "up" if cache_available else "down",
            "type": "redis" if cache_stats.get("available") else "local",
            "stats": cache_stats,
        }
        if not cache_available:
            overall_healthy = False
    except Exception as e:
        logger.error(f"健康检查-缓存服务失败: {e}")
        health_status["services"]["cache"] = {"status": "down", "error": str(e)}
        overall_healthy = False

    try:
        from src.services.support.monitoring_service import get_monitoring_service
        monitoring_service = get_monitoring_service()
        health_status["services"]["monitoring"] = {"status": "up"}
    except Exception as e:
        logger.error(f"健康检查-监控服务失败: {e}")
        health_status["services"]["monitoring"] = {"status": "down", "error": str(e)}

    try:
        try:
            from src.services.cache_service.redis_cache import get_redis_cache
            redis_cache = get_redis_cache()
            redis_ping = redis_cache.redis_client.ping() if redis_cache.available else False
            health_status["services"]["redis"] = {"status": "up" if redis_ping else "down"}
        except Exception:
            health_status["services"]["redis"] = {"status": "not_configured"}
    except Exception as e:
        logger.error(f"健康检查-Redis失败: {e}")
        health_status["services"]["redis"] = {"status": "down", "error": str(e)}

    try:
        from src.services.model.recognition_service import get_recognition_service
        recognition_service = get_recognition_service()
        record_count = len(recognition_service.records)
        health_status["services"]["recognition"] = {"status": "up", "record_count": record_count}
    except Exception as e:
        logger.error(f"健康检查-识别记录服务失败: {e}")
        health_status["services"]["recognition"] = {"status": "down", "error": str(e)}

    try:
        from src.services.messaging.message_queue_service import MessageQueueService
        mq_service = MessageQueueService()
        mq_status = "up" if mq_service.connection and mq_service.channel else "down"
        health_status["services"]["message_queue"] = {"status": mq_status}
    except Exception as e:
        logger.error(f"健康检查-消息队列失败: {e}")
        health_status["services"]["message_queue"] = {"status": "down", "error": str(e)}

    try:
        import psutil
        memory = psutil.virtual_memory()
        health_status["services"]["system"] = {
            "status": "up",
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / (1024 * 1024),
        }
    except ImportError:
        health_status["services"]["system"] = {"status": "not_monitored"}
    except Exception as e:
        logger.error(f"健康检查-系统信息失败: {e}")
        health_status["services"]["system"] = {"status": "unknown", "error": str(e)}

    if not overall_healthy:
        health_status["status"] = "degraded"

    return health_status


@router.get("/api/monitoring")
async def get_monitoring_info():
    """获取监控信息"""
    try:
        from src.middleware.monitoring import get_service_monitor
        monitor_info = get_service_monitor()
        return {"status": "ok", "monitoring": monitor_info}
    except Exception as e:
        logger.error(f"获取监控信息失败: {e}")
        return {"status": "error", "error": str(e)}

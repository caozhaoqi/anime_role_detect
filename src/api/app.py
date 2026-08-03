"""
Anime Role Detect API 主入口
只负责初始化、挂载配置和生命周期管理
"""
import sys
import os
from contextlib import asynccontextmanager

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

from fastapi import FastAPI
from fastapi.responses import Response

from src.core.logging import get_enhanced_logger as get_logger
from src.core.config.service_config import get_service_config
from src.core.logging_setup import setup_logging

config = get_service_config()
logger = get_logger("api")

# 2.5 结构化日志：统一 loguru 输出为 JSON 行（幂等，仅配置一次）
setup_logging("api-service")

# 设置缓存目录
os.environ["HF_HOME"] = config.HF_CACHE_DIR
os.environ["KERAS_HOME"] = config.KERAS_CACHE_DIR
os.makedirs(config.HF_CACHE_DIR, exist_ok=True)
os.makedirs(config.KERAS_CACHE_DIR, exist_ok=True)


# lifespan 上下文管理器（替代废弃的 on_event("startup")）
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # Startup
    from src.api.lifecycle import _init_services, _shutdown_services
    logger.info("应用启动 - 初始化服务")
    await _init_services()
    logger.info("所有服务初始化完成")
    yield
    # Shutdown
    logger.info("应用关闭 - 清理资源")
    await _shutdown_services()


# 创建FastAPI应用实例
app = FastAPI(
    title="Anime Role Detect API",
    description="Anime role detection API - 用于检测和分类动画角色的API服务",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# 配置中间件、异常处理器、路由（启动事件已迁移到 lifespan）
from src.api.lifecycle import setup_middlewares, setup_exception_handlers, setup_routers

setup_exception_handlers(app)
setup_middlewares(app)
setup_routers(app)


# Prometheus 指标端点（保持在此处因为它是基础设施入口）
@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点"""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return Response(content="# Prometheus client not installed", media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

"""
Anime Role Detect API 主入口
只负责初始化、挂载配置和生命周期管理
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.responses import Response

from src.core.logging.global_logger import get_logger
from src.core.config.service_config import get_service_config

config = get_service_config()
logger = get_logger("api")

# 设置缓存目录
os.environ["HF_HOME"] = config.HF_CACHE_DIR
os.environ["KERAS_HOME"] = config.KERAS_CACHE_DIR
os.makedirs(config.HF_CACHE_DIR, exist_ok=True)
os.makedirs(config.KERAS_CACHE_DIR, exist_ok=True)

# 创建FastAPI应用实例
app = FastAPI(
    title="Anime Role Detect API",
    description="Anime role detection API - 用于检测和分类动画角色的API服务",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# 配置中间件、异常处理器、路由和启动事件
from src.api.lifecycle import setup_middlewares, setup_exception_handlers, setup_routers, setup_startup_handler

setup_exception_handlers(app)
setup_middlewares(app)
setup_routers(app)
setup_startup_handler(app)


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
    uvicorn.run(app, host="127.0.0.1", port=8001)
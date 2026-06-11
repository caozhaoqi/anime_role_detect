"""
中间件配置模块 - 统一管理FastAPI中间件
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.logging.global_logger import get_logger

logger = get_logger("api.middleware")


def setup_middlewares(app: FastAPI) -> None:
    """配置所有中间件"""

    # 1. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. 监控中间件
    try:
        from src.middleware.monitoring import monitoring_middleware
        app.middleware("http")(monitoring_middleware)
        logger.info("监控中间件加载成功")
    except Exception as e:
        logger.warning(f"导入监控中间件失败: {e}")

    # 3. 链路追踪中间件
    try:
        from src.middleware.tracing import TracingMiddleware
        app.add_middleware(TracingMiddleware)
        logger.info("链路追踪中间件加载成功")
    except Exception as e:
        logger.warning(f"导入链路追踪中间件失败: {e}")

    # 4. 认证中间件
    try:
        from src.middleware.auth_enhanced import auth_middleware
        app.middleware("http")(auth_middleware)
        logger.info("认证中间件加载成功")
    except Exception as e:
        logger.warning(f"导入认证中间件失败: {e}")

    logger.info("所有中间件配置完成")


def setup_exception_handlers(app: FastAPI) -> None:
    """配置全局异常处理器"""
    try:
        from src.core.error.error_handler import global_exception_handler
        app.add_exception_handler(Exception, global_exception_handler)
        logger.info("全局异常处理器加载成功")
    except Exception as e:
        logger.warning(f"导入全局异常处理器失败: {e}")


def setup_routers(app: FastAPI) -> None:
    """注册所有路由模块"""
    routers = [
        # 核心业务路由
        ("src.api.routes.classification", None),
        ("src.api.routes.health", None),
        ("src.api.routes.auth", None),
        ("src.api.routes.models", None),
        ("src.api.routes.history", None),
        ("src.api.routes.misc", None),
        # 已有独立路由
        ("src.api.routes.search_routes", None),
        ("src.api.routes.onnx_inference", None),
        ("src.api.routes.collector", "/api/collector"),
        ("src.api.routes.async_inference", None),
        ("src.api.routes.cleaning_routes", None),
        ("src.api.routes.tracing", None),
        ("src.api.routes.video_routes", None),
    ]

    for module_path, prefix in routers:
        try:
            __import__(module_path, fromlist=["router"])
            router_mod = sys.modules[module_path]
            if prefix:
                app.include_router(router_mod.router, prefix=prefix)
            else:
                app.include_router(router_mod.router)
            logger.info(f"路由 {module_path} 加载成功")
        except Exception as e:
            logger.warning(f"导入路由 {module_path} 失败: {e}")

    logger.info("所有路由注册完成")


def setup_startup_handler(app: FastAPI) -> None:
    """配置启动事件"""
    @app.on_event("startup")
    async def startup_event():
        """启动事件 - 初始化所有服务"""
        try:
            from src.services.support.auth_service import init_auth_service
            init_auth_service()
            logger.info("认证服务初始化完成")

            from src.services.cache_service import init_cache_manager
            init_cache_manager()
            logger.info("缓存管理器初始化完成")

            from src.services.support.monitoring_service import init_monitoring_service
            init_monitoring_service()
            logger.info("监控服务初始化完成")

            from src.services.messaging.message_queue_service import init_message_queue_service
            init_message_queue_service()
            logger.info("消息队列服务初始化完成")

            from src.services.support.circuit_breaker_service import init_circuit_breaker_service
            init_circuit_breaker_service()
            logger.info("熔断器服务初始化完成")

            from src.services.model.model_version_service import init_model_version_service
            init_model_version_service()
            logger.info("模型版本服务初始化完成")

            from src.services.model.multi_model_service import init_multi_model_service
            init_multi_model_service()
            logger.info("多模型服务初始化完成")

            from src.services.processor.model_loader import load_models
            load_models()
            logger.info("模型加载完成")
        except Exception as e:
            logger.error(f"服务初始化失败: {e}")

    logger.info("启动事件处理程序配置完成")
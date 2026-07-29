"""
中间件配置模块 - 统一管理FastAPI中间件
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("api.middleware")


def setup_middlewares(app: FastAPI) -> None:
    """配置所有中间件"""

    # 1. CORS - 生产环境应通过 CORS_ALLOWED_ORIGINS 环境变量指定具体 origins
    _cors_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    if _cors_origins_env:
        allowed_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    else:
        allowed_origins = [
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
        ]
    allow_credentials = True
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "Accept",
            "Origin",
            "X-CSRF-Token",
        ],
    )
    logger.info(f"CORS配置: origins={allowed_origins}, allow_credentials={allow_credentials}")

    # 2. OpenTelemetry 插桩（可选，通过 OTEL_ENABLED 环境变量控制）
    try:
        from src.utils.monitoring.opentelemetry import instrument_app
        instrument_app(app, service_name="api-service")
        logger.info("OpenTelemetry 插桩加载成功")
    except Exception as e:
        logger.warning(f"导入 OpenTelemetry 插桩失败: {e}")

    # 3. 监控中间件
    try:
        from src.middleware.monitoring import monitoring_middleware
        app.middleware("http")(monitoring_middleware)
        logger.info("监控中间件加载成功")
    except Exception as e:
        logger.warning(f"导入监控中间件失败: {e}")

    # 4. 链路追踪中间件
    try:
        from src.middleware.tracing import TracingMiddleware
        app.add_middleware(TracingMiddleware)
        logger.info("链路追踪中间件加载成功")
    except Exception as e:
        logger.warning(f"导入链路追踪中间件失败: {e}")

    # 5. 认证中间件
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
        ("src.api.routes.version", None),
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
    """配置启动事件（已迁移到 lifespan，保留兼容）"""
    @app.on_event("startup")
    async def startup_event():
        """启动事件 - 初始化所有服务"""
        await _init_services()

    logger.info("启动事件处理程序配置完成（兼容模式，建议迁移到 lifespan）")


async def _init_services():
    """初始化所有服务"""
    # P1-4: 初始化全局 HttpClientManager（aiohttp.ClientSession 单例）
    try:
        from src.services.processor.model_processor import HttpClientManager
        HttpClientManager.init_session()
        logger.info("HttpClientManager 初始化完成")
    except Exception as e:
        logger.warning(f"HttpClientManager 初始化失败（可选）: {e}")

    services_to_init = [
        ("认证服务", "src.services.support.auth_service", "init_auth_service"),
        ("缓存管理器", "src.services.cache_service", "init_cache_manager"),
        ("监控服务", "src.services.support.monitoring_service", "init_monitoring_service"),
        ("消息队列服务", "src.services.messaging.message_queue_service", "init_message_queue_service"),
        ("熔断器服务", "src.services.support.circuit_breaker_service", "init_circuit_breaker_service"),
        ("模型版本服务", "src.services.model.model_version_service", "init_model_version_service"),
        ("多模型服务", "src.services.model.multi_model_service", "init_multi_model_service"),
        ("模型加载", "src.services.processor.model_loader", "load_models"),
    ]

    for service_name, module_path, func_name in services_to_init:
        try:
            __import__(module_path, fromlist=[func_name])
            module = sys.modules[module_path]
            func = getattr(module, func_name)
            func()
            logger.info(f"{service_name}初始化完成")
        except ImportError as e:
            logger.warning(f"{service_name}导入失败（可选模块）: {e}")
        except Exception as e:
            logger.error(f"{service_name}初始化失败: {e}")


async def _shutdown_services():
    """关闭所有服务（P1-4: 清理 HttpClientManager）"""
    try:
        from src.services.processor.model_processor import HttpClientManager
        await HttpClientManager.close_session()
        logger.info("HttpClientManager session 已关闭")
    except Exception as e:
        logger.warning(f"HttpClientManager 关闭失败: {e}")

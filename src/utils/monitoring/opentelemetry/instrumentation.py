"""
OpenTelemetry 应用程序插桩
自动为 FastAPI 应用添加追踪支持
"""

import os
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


def instrument_app(app: FastAPI, service_name: str = "anime_role_detect") -> None:
    """
    为 FastAPI 应用添加 OpenTelemetry 插桩

    Args:
        app: FastAPI 应用实例
        service_name: 服务名称
    """
    enabled = os.environ.get("OTEL_ENABLED", "false").lower() == "true"
    
    if not enabled:
        return

    from .otel_tracer import init_otel_tracer
    init_otel_tracer(service_name)

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=None,
        excluded_urls=[
            "/health",
            "/api/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ],
    )
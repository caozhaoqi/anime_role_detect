"""
OpenTelemetry 集成模块
提供标准化的分布式链路追踪支持
"""

from .otel_tracer import init_otel_tracer, get_otel_tracer, shutdown_otel
from .instrumentation import instrument_app

__all__ = [
    "init_otel_tracer",
    "get_otel_tracer",
    "shutdown_otel",
    "instrument_app",
]
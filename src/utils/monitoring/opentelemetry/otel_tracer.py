"""
OpenTelemetry Tracer 封装
提供与自定义追踪系统兼容的接口
"""

import os
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Tracer

_otel_tracer: Optional[Tracer] = None


def init_otel_tracer(service_name: str = "anime_role_detect") -> Tracer:
    """
    初始化 OpenTelemetry Tracer

    Args:
        service_name: 服务名称

    Returns:
        OpenTelemetry Tracer 实例
    """
    global _otel_tracer

    otlp_endpoint = os.environ.get("OTLP_ENDPOINT", "")
    
    resource = Resource(attributes={
        "service.name": service_name,
        "service.version": "1.0.0",
    })

    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    _otel_tracer = trace.get_tracer(service_name)

    return _otel_tracer


def get_otel_tracer(service_name: str = "anime_role_detect") -> Tracer:
    """
    获取 OpenTelemetry Tracer 实例

    Args:
        service_name: 服务名称

    Returns:
        OpenTelemetry Tracer 实例
    """
    global _otel_tracer
    if _otel_tracer is None:
        _otel_tracer = init_otel_tracer(service_name)
    return _otel_tracer


def shutdown_otel():
    """关闭 OpenTelemetry 资源"""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
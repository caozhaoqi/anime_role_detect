"""
OpenTelemetry Tracer 封装
提供与自定义追踪系统兼容的接口
"""

import os
import traceback
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
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

    try:
        otlp_endpoint = os.environ.get("OTLP_ENDPOINT", "")
        
        resource = Resource(attributes={
            "service.name": service_name,
            "service.version": "1.0.0",
        })

        provider = TracerProvider(resource=resource)

        try:
            from .otel_exporter import TraceStorageSpanExporter
            exporter = TraceStorageSpanExporter()
            processor = SimpleSpanProcessor(exporter)
            provider.add_span_processor(processor)
            print(f"✅ 已添加 TraceStorageSpanExporter")
        except Exception as e:
            print(f"⚠️  TraceStorageSpanExporter 添加失败: {e}")

        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            otlp_processor = BatchSpanProcessor(otlp_exporter)
            provider.add_span_processor(otlp_processor)
            print(f"✅ 已添加 OTLP Exporter: {otlp_endpoint}")

        trace.set_tracer_provider(provider)
        _otel_tracer = trace.get_tracer(service_name)

        print(f"✅ OpenTelemetry Tracer 初始化成功 - 服务: {service_name}")
        return _otel_tracer
        
    except Exception as e:
        print(f"❌ OpenTelemetry Tracer 初始化失败: {e}")
        print(f"   堆栈: {traceback.format_exc()}")
        raise


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
    try:
        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
            print("✅ OpenTelemetry 资源已关闭")
    except Exception as e:
        print(f"❌ 关闭 OpenTelemetry 资源失败: {e}")
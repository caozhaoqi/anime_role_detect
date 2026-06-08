"""
API调用链路追踪模块

参考OpenTelemetry、SkyWalking和Zipkin的设计理念，实现分布式链路追踪功能。

核心组件：
- Span: 代表一次操作的执行单元
- Trace: 代表一个完整的请求链路
- SpanContext: 包含Span的上下文信息（trace_id, span_id等）
- Tracer: 负责创建和管理Span

使用方式：
    from src.utils.monitoring.tracing import get_tracer
    
    tracer = get_tracer("my_service")
    
    with tracer.start_as_current_span("operation_name") as span:
        span.set_attribute("key", "value")
        # 执行操作
        span.add_event("event_name")
"""

from .tracer import Tracer, get_tracer, init_tracer
from .span import Span, SpanKind, StatusCode
from .trace import Trace
from .span_context import SpanContext

__all__ = [
    "Tracer",
    "get_tracer",
    "init_tracer",
    "Span",
    "SpanKind",
    "StatusCode",
    "Trace",
    "SpanContext",
]

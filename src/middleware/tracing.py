"""
链路追踪中间件 - 自动拦截API请求并生成追踪数据

参考OpenTelemetry的Instrumentation设计
"""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.monitoring.tracing import get_tracer, SpanKind, StatusCode
from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("tracing_middleware")


class TracingMiddleware(BaseHTTPMiddleware):
    """
    链路追踪中间件，自动为每个HTTP请求创建Span
    """

    def __init__(self, app, excluded_paths: list = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/health",
            "/api/health",
            "/api/status",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        self.tracer = get_tracer()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求，创建追踪Span
        
        Args:
            request: HTTP请求对象
            call_next: 下一个中间件或处理函数
        
        Returns:
            HTTP响应对象
        """
        # 检查是否需要排除
        path = request.url.path
        if any(path.startswith(excluded) for excluded in self.excluded_paths):
            return await call_next(request)

        # 创建Span名称
        span_name = f"{request.method} {path}"
        
        # 解析trace_id（如果请求头中包含）
        trace_id = request.headers.get("X-Trace-Id")
        parent_span_id = request.headers.get("X-Parent-Span-Id")

        # 启动Span
        span = self.tracer.start_span(
            name=span_name,
            kind=SpanKind.SERVER,
            parent_span_id=parent_span_id,
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.path": path,
                "http.query_params": str(request.query_params),
                "client.ip": request.client.host if request.client else "unknown",
                "client.port": request.client.port if request.client else 0,
                # P3-1: 仅记录关键请求头，避免记录全部 headers 导致内存膨胀
                "request.headers": {
                    "content_type": request.headers.get("content-type"),
                    "content_length": request.headers.get("content-length"),
                    "user_agent": request.headers.get("user-agent"),
                },
            },
        )

        # 设置trace_id到请求状态中，方便后续使用
        request.state.trace_id = span.context.trace_id
        request.state.span_id = span.context.span_id

        start_time = time.time()
        response = None

        try:
            # 执行下一个中间件或处理函数
            response = await call_next(request)

            # 记录响应状态码
            span.set_attribute("http.status_code", response.status_code)

            # 根据状态码设置状态
            if response.status_code >= 500:
                span.set_status(StatusCode.ERROR, f"HTTP {response.status_code}")
            elif response.status_code >= 400:
                span.set_status(StatusCode.OK)  # 客户端错误不算追踪错误
            else:
                span.set_status(StatusCode.OK)

            return response

        except Exception as e:
            # 记录异常
            span.set_status(StatusCode.ERROR, str(e))
            span.add_event(
                "exception",
                {
                    "type": type(e).__name__,
                    "message": str(e),
                },
            )
            logger.error(f"请求处理异常: {e}")
            raise

        finally:
            # 结束Span
            duration = time.time() - start_time
            span.set_attribute("duration_ms", round(duration * 1000, 2))
            span.end()
            
            # 结束Trace
            self.tracer.end_trace(span.context.trace_id)
            
            logger.debug(
                f"追踪完成 - trace_id: {span.context.trace_id[:8]}..., "
                f"span_id: {span.context.span_id[:8]}..., "
                f"duration: {round(duration * 1000, 2)}ms, "
                f"status: {span.status.code.value}"
            )


def get_trace_context(request: Request) -> dict:
    """
    从请求中获取追踪上下文
    
    Args:
        request: HTTP请求对象
    
    Returns:
        追踪上下文字典
    """
    return {
        "trace_id": getattr(request.state, "trace_id", None),
        "span_id": getattr(request.state, "span_id", None),
    }


def inject_trace_headers(response: Response, trace_id: str, span_id: str):
    """
    将追踪信息注入到响应头中
    
    Args:
        response: HTTP响应对象
        trace_id: 追踪ID
        span_id: Span ID
    """
    response.headers["X-Trace-Id"] = trace_id
    response.headers["X-Span-Id"] = span_id

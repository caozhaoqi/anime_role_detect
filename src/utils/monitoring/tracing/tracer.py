"""
Tracer - 负责创建和管理Span

参考OpenTelemetry的Tracer设计
"""

import time
from typing import Optional, Dict, Any, List
from threading import local

from .span import Span, SpanKind
from .trace import Trace
from .span_context import SpanContext


class Tracer:
    """
    负责创建和管理Span的追踪器
    """

    _instance: Optional["Tracer"] = None
    _thread_local = local()

    def __new__(cls, service_name: str = "anime_role_detect"):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, service_name: str = "anime_role_detect"):
        """
        初始化Tracer
        
        Args:
            service_name: 服务名称，用于标识追踪来源
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.service_name = service_name
        self._traces: Dict[str, Trace] = {}
        self._active_trace: Optional[Trace] = None
        self._active_span: Optional[Span] = None
        self._initialized = True

    @property
    def active_trace(self) -> Optional[Trace]:
        """获取当前活跃的Trace"""
        return getattr(self._thread_local, "active_trace", None)

    @active_trace.setter
    def active_trace(self, trace: Optional[Trace]):
        """设置当前活跃的Trace"""
        setattr(self._thread_local, "active_trace", trace)

    @property
    def active_span(self) -> Optional[Span]:
        """获取当前活跃的Span"""
        return getattr(self._thread_local, "active_span", None)

    @active_span.setter
    def active_span(self, span: Optional[Span]):
        """设置当前活跃的Span"""
        setattr(self._thread_local, "active_span", span)

    def start_trace(self, trace_id: Optional[str] = None) -> Trace:
        """
        启动一个新的Trace
        
        Args:
            trace_id: 追踪ID，如果不指定会自动生成
        
        Returns:
            Trace对象
        """
        trace = Trace(trace_id)
        self._traces[trace.trace_id] = trace
        self.active_trace = trace
        return trace

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """
        创建并启动一个新的Span
        
        Args:
            name: Span名称
            kind: Span类型
            parent_span_id: 父Span ID，如果不指定则使用当前活跃Span
            attributes: 初始属性
        
        Returns:
            Span对象
        """
        # 获取或创建Trace
        if self.active_trace is None:
            self.start_trace()

        # 确定父Span ID
        actual_parent_id = parent_span_id
        if actual_parent_id is None and self.active_span is not None:
            actual_parent_id = self.active_span.context.span_id

        # 创建SpanContext
        context = SpanContext(
            trace_id=self.active_trace.trace_id,
            parent_span_id=actual_parent_id,
        )

        # 创建Span
        span = Span(name=name, context=context, kind=kind)
        
        # 设置初始属性
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        # 设置服务名称属性
        span.set_attribute("service.name", self.service_name)

        # 添加到Trace
        if self.active_trace:
            self.active_trace.add_span(span)

        # 设置为当前活跃Span
        self.active_span = span

        return span

    def start_as_current_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """
        创建并启动一个新的Span，并将其设置为当前Span（上下文管理器模式）
        
        Args:
            name: Span名称
            kind: Span类型
            parent_span_id: 父Span ID
            attributes: 初始属性
        
        Returns:
            Span对象（可用于with语句）
        """
        return self.start_span(name, kind, parent_span_id, attributes)

    def end_span(self, span: Span):
        """
        结束指定的Span
        
        Args:
            span: 要结束的Span
        """
        span.end()
        
        # 如果当前活跃Span是该Span的子Span，不更新
        # 否则更新为父Span
        if self.active_span == span:
            if span.context.parent_span_id and self.active_trace:
                parent_span = self.active_trace.get_span_by_id(span.context.parent_span_id)
                self.active_span = parent_span
            else:
                self.active_span = None

    def end_trace(self, trace_id: str):
        """
        结束指定的Trace
        
        Args:
            trace_id: Trace ID
        """
        if trace_id in self._traces:
            trace = self._traces[trace_id]
            trace.end()
            
            # 将Trace存储到存储服务
            try:
                from src.services.trace_storage_service import get_trace_storage_service
                storage = get_trace_storage_service()
                storage.store_trace(trace)
            except Exception as e:
                import logging
                logger = logging.getLogger("tracer")
                logger.debug(f"存储Trace失败: {e}")
        
        if self.active_trace and self.active_trace.trace_id == trace_id:
            self.active_trace = None
            self.active_span = None

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """
        获取指定的Trace
        
        Args:
            trace_id: Trace ID
        
        Returns:
            Trace对象，如果不存在返回None
        """
        return self._traces.get(trace_id)

    def get_all_traces(self) -> List[Trace]:
        """获取所有Trace"""
        return list(self._traces.values())

    def clear_traces(self):
        """清空所有Trace"""
        self._traces.clear()
        self.active_trace = None
        self.active_span = None

    def record_exception(self, exception: Exception, span: Optional[Span] = None):
        """
        记录异常信息到Span
        
        Args:
            exception: 异常对象
            span: 目标Span，如果不指定则使用当前活跃Span
        """
        target_span = span or self.active_span
        if target_span:
            target_span.set_status(StatusCode.ERROR, str(exception))
            target_span.add_event(
                "exception",
                {
                    "type": type(exception).__name__,
                    "message": str(exception),
                    "timestamp": time.time(),
                },
            )

    def __repr__(self) -> str:
        return f"Tracer(service_name={self.service_name}, trace_count={len(self._traces)})"


# 全局Tracer实例
_tracer: Optional[Tracer] = None


def init_tracer(service_name: str = "anime_role_detect") -> Tracer:
    """
    初始化全局Tracer
    
    Args:
        service_name: 服务名称
    
    Returns:
        Tracer实例
    """
    global _tracer
    _tracer = Tracer(service_name)
    return _tracer


def get_tracer(service_name: str = "anime_role_detect") -> Tracer:
    """
    获取全局Tracer实例
    
    Args:
        service_name: 服务名称（仅在首次调用时有效）
    
    Returns:
        Tracer实例
    """
    global _tracer
    if _tracer is None:
        _tracer = Tracer(service_name)
    return _tracer


# 导入StatusCode
from .span import StatusCode

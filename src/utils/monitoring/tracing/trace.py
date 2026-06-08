"""
Trace - 代表一个完整的请求链路

参考OpenTelemetry和Zipkin的Trace设计
"""

import time
from typing import Dict, List, Optional
from datetime import datetime

from .span import Span, SpanKind, StatusCode
from .span_context import SpanContext


class Trace:
    """
    代表一个完整的请求链路，包含多个Span
    """

    def __init__(self, trace_id: Optional[str] = None):
        """
        初始化Trace
        
        Args:
            trace_id: 追踪ID
        """
        self.trace_id = trace_id or SpanContext._generate_trace_id()
        self.spans: List[Span] = []
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self._is_active = True

    def add_span(self, span: Span):
        """添加Span"""
        if self._is_active:
            self.spans.append(span)

    def end(self, end_time: Optional[float] = None):
        """结束Trace"""
        self.end_time = end_time or time.time()
        self._is_active = False

        # 确保所有子Span都已结束
        for span in self.spans:
            if span.end_time is None:
                span.end(self.end_time)

    @property
    def duration(self) -> float:
        """获取总持续时间（秒）"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def status(self) -> StatusCode:
        """获取整体状态"""
        for span in self.spans:
            if span.status.code == StatusCode.ERROR:
                return StatusCode.ERROR
        return StatusCode.OK

    @property
    def root_span(self) -> Optional[Span]:
        """获取根Span（parent_span_id为空的Span）"""
        for span in self.spans:
            if span.context.parent_span_id is None:
                return span
        return self.spans[0] if self.spans else None

    def get_spans_by_kind(self, kind: SpanKind) -> List[Span]:
        """按类型获取Span列表"""
        return [span for span in self.spans if span.kind == kind]

    def get_span_by_id(self, span_id: str) -> Optional[Span]:
        """根据span_id获取Span"""
        for span in self.spans:
            if span.context.span_id == span_id:
                return span
        return None

    def get_child_spans(self, parent_span_id: str) -> List[Span]:
        """获取子Span列表"""
        return [span for span in self.spans if span.context.parent_span_id == parent_span_id]

    def to_dict(self, include_spans: bool = True) -> dict:
        """转换为字典表示"""
        result = {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration * 1000, 2),
            "status": self.status.value,
            "span_count": len(self.spans),
            "start_time_human": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time_human": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
        }
        
        if include_spans:
            result["spans"] = [span.to_dict() for span in self.spans]
        
        return result

    def build_tree(self) -> dict:
        """构建Span树形结构"""
        span_dict = {span.context.span_id: span.to_dict() for span in self.spans}
        
        for span_id, span_data in span_dict.items():
            span_data["children"] = []
        
        for span_id, span_data in span_dict.items():
            parent_id = span_data.get("parent_span_id")
            if parent_id and parent_id in span_dict:
                span_dict[parent_id]["children"].append(span_data)
        
        # 找到根节点
        root = None
        for span_id, span_data in span_dict.items():
            if not span_data.get("parent_span_id"):
                root = span_data
                break
        
        return root or {}

    def __repr__(self) -> str:
        return (
            f"Trace(trace_id={self.trace_id[:8]}..., span_count={len(self.spans)}, "
            f"duration={round(self.duration * 1000, 2)}ms, status={self.status.value})"
        )

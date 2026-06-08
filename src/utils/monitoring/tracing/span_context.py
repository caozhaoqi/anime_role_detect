"""
SpanContext - 包含Span的上下文信息

参考OpenTelemetry的SpanContext设计
"""

import uuid
from typing import Optional


class SpanContext:
    """
    Span的上下文信息，包含trace_id、span_id等核心标识符
    """

    def __init__(
        self,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        trace_flags: int = 1,  # 1 = sampled
        trace_state: Optional[str] = None,
    ):
        """
        初始化SpanContext
        
        Args:
            trace_id: 追踪ID，唯一标识一个请求链路
            span_id: Span ID，唯一标识一个操作单元
            parent_span_id: 父Span ID
            trace_flags: 追踪标志位，1表示采样，0表示不采样
            trace_state: 追踪状态信息
        """
        self.trace_id = trace_id or self._generate_trace_id()
        self.span_id = span_id or self._generate_span_id()
        self.parent_span_id = parent_span_id
        self.trace_flags = trace_flags
        self.trace_state = trace_state

    @staticmethod
    def _generate_trace_id() -> str:
        """生成唯一的trace_id（16进制，32位字符）"""
        return uuid.uuid4().hex

    @staticmethod
    def _generate_span_id() -> str:
        """生成唯一的span_id（16进制，16位字符）"""
        return uuid.uuid4().hex[:16]

    def is_sampled(self) -> bool:
        """判断是否被采样"""
        return (self.trace_flags & 0x01) == 0x01

    def to_dict(self) -> dict:
        """转换为字典表示"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_flags": self.trace_flags,
            "trace_state": self.trace_state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpanContext":
        """从字典创建SpanContext"""
        return cls(
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            parent_span_id=data.get("parent_span_id"),
            trace_flags=data.get("trace_flags", 1),
            trace_state=data.get("trace_state"),
        )

    def __repr__(self) -> str:
        return (
            f"SpanContext(trace_id={self.trace_id[:8]}..., span_id={self.span_id[:8]}..., "
            f"parent_span_id={self.parent_span_id[:8]}... if self.parent_span_id else None)"
        )

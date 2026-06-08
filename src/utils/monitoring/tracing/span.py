"""
Span - 代表一次操作的执行单元

参考OpenTelemetry的Span设计
"""

import time
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime

from .span_context import SpanContext


class SpanKind(Enum):
    """
    Span类型枚举
    """
    UNSPECIFIED = "UNSPECIFIED"
    INTERNAL = "INTERNAL"      # 内部操作
    SERVER = "SERVER"          # 服务端接收请求
    CLIENT = "CLIENT"          # 客户端发起请求
    PRODUCER = "PRODUCER"      # 消息生产者
    CONSUMER = "CONSUMER"      # 消息消费者


class StatusCode(Enum):
    """
    Span状态码枚举
    """
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class Status:
    """
    Span的状态信息
    """

    def __init__(self, code: StatusCode = StatusCode.UNSET, message: Optional[str] = None):
        self.code = code
        self.message = message

    def to_dict(self) -> dict:
        return {"code": self.code.value, "message": self.message}


class Event:
    """
    Span中的事件记录
    """

    def __init__(self, name: str, timestamp: Optional[float] = None, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.timestamp = timestamp or time.time()
        self.attributes = attributes or {}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": self.attributes,
        }


class Span:
    """
    代表一次操作的执行单元
    """

    def __init__(
        self,
        name: str,
        context: SpanContext,
        kind: SpanKind = SpanKind.INTERNAL,
        start_time: Optional[float] = None,
    ):
        """
        初始化Span
        
        Args:
            name: Span名称，描述操作类型
            context: Span上下文信息
            kind: Span类型
            start_time: 开始时间（时间戳）
        """
        self.name = name
        self.context = context
        self.kind = kind
        self.start_time = start_time or time.time()
        self.end_time: Optional[float] = None
        self.status = Status()
        self.attributes: Dict[str, Any] = {}
        self.events: List[Event] = []
        self._is_recording = True

    def set_attribute(self, key: str, value: Any):
        """设置属性"""
        if self._is_recording:
            self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """添加事件"""
        if self._is_recording:
            self.events.append(Event(name, attributes=attributes))

    def set_status(self, code: StatusCode, message: Optional[str] = None):
        """设置状态"""
        if self._is_recording:
            self.status = Status(code, message)

    def end(self, end_time: Optional[float] = None):
        """结束Span"""
        if self._is_recording:
            self.end_time = end_time or time.time()
            self._is_recording = False

    @property
    def duration(self) -> float:
        """获取持续时间（秒）"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """转换为字典表示"""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration * 1000, 2),
            "status": self.status.to_dict(),
            "attributes": self.attributes,
            "events": [event.to_dict() for event in self.events],
            "start_time_human": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time_human": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
        }

    def __enter__(self):
        """进入上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        if exc_type is not None:
            self.set_status(StatusCode.ERROR, str(exc_val))
        else:
            self.set_status(StatusCode.OK)
        self.end()

    def __repr__(self) -> str:
        return (
            f"Span(name={self.name}, trace_id={self.context.trace_id[:8]}..., "
            f"span_id={self.context.span_id[:8]}..., kind={self.kind.value}, "
            f"duration={round(self.duration * 1000, 2)}ms)"
        )

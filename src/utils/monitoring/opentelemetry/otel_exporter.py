"""
OpenTelemetry 自定义 Span Exporter
将追踪数据导出到项目的 TraceStorageService
"""

import os
from typing import Sequence
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Span
from opentelemetry.trace.status import StatusCode

from src.services.support.trace_storage_service import get_trace_storage_service
from src.utils.monitoring.tracing.trace import Trace
from src.utils.monitoring.tracing.span import Span as LocalSpan, SpanContext


class TraceStorageSpanExporter(SpanExporter):
    """
    自定义 Span Exporter，将追踪数据导出到项目的 TraceStorageService
    """

    def __init__(self):
        self.storage = get_trace_storage_service()
        print(f"✅ TraceStorageSpanExporter 初始化完成")

    def export(self, spans: Sequence[Span]) -> SpanExportResult:
        """
        导出 Span 到追踪存储服务
        
        Args:
            spans: Span 序列
            
        Returns:
            SpanExportResult
        """
        try:
            if not spans:
                return SpanExportResult.SUCCESS

            print(f"📤 收到 {len(spans)} 个 Span 需要导出")

            trace_map = {}
            
            for span in spans:
                span_context = span.get_span_context()
                trace_id = format(span_context.trace_id, "032x")
                
                if trace_id not in trace_map:
                    trace_map[trace_id] = {
                        "trace_id": trace_id,
                        "spans": [],
                        "start_time": float("inf"),
                        "end_time": 0,
                        "status": "SUCCESS",
                    }

                span_data = self._convert_span(span)
                trace_map[trace_id]["spans"].append(span_data)
                
                if span_data["start_time"] < trace_map[trace_id]["start_time"]:
                    trace_map[trace_id]["start_time"] = span_data["start_time"]
                if span_data["end_time"] > trace_map[trace_id]["end_time"]:
                    trace_map[trace_id]["end_time"] = span_data["end_time"]
                
                if span_data["status"] == "ERROR":
                    trace_map[trace_id]["status"] = "ERROR"

            for trace_id, trace_data in trace_map.items():
                trace_data["duration_ms"] = (trace_data["end_time"] - trace_data["start_time"]) * 1000
                
                local_trace = Trace(trace_id=trace_id)
                local_trace.start_time = trace_data["start_time"]
                local_trace.end_time = trace_data["end_time"]
                
                for span_data in trace_data["spans"]:
                    context = SpanContext(
                        trace_id=span_data["trace_id"],
                        span_id=span_data["span_id"],
                        parent_span_id=span_data.get("parent_id"),
                    )
                    local_span = LocalSpan(
                        name=span_data["name"],
                        context=context,
                        start_time=span_data["start_time"],
                    )
                    local_span.end_time = span_data["end_time"]
                    local_span.attributes = span_data["attributes"]
                    local_trace.add_span(local_span)
                
                self.storage.store_trace(local_trace)
                print(f"✅ 已存储 Trace: {trace_id[:16]}...")

            return SpanExportResult.SUCCESS

        except Exception as e:
            import traceback
            print(f"❌ 导出追踪数据失败: {e}")
            print(f"   堆栈: {traceback.format_exc()}")
            return SpanExportResult.FAILURE

    def shutdown(self):
        """关闭 Exporter"""
        pass

    def _convert_span(self, span: Span) -> dict:
        """
        将 OpenTelemetry Span 转换为字典格式
        
        Args:
            span: OpenTelemetry Span
            
        Returns:
            转换后的字典
        """
        span_context = span.get_span_context()
        parent_span_id = span.parent.span_id if span.parent else None
        
        status_code = span.status.status_code
        status = "ERROR" if status_code == StatusCode.ERROR else "SUCCESS"
        
        attributes = {}
        for key, value in span.attributes.items():
            try:
                attributes[key] = str(value)
            except:
                attributes[key] = "N/A"
        
        service_name = attributes.get("service.name", "")
        if not service_name:
            service_name = attributes.get("http.host", "")
        
        start_time = span.start_time
        end_time = span.end_time
        
        if hasattr(start_time, 'timestamp'):
            start_timestamp = start_time.timestamp()
        else:
            start_timestamp = start_time / 1e9
        
        if hasattr(end_time, 'timestamp'):
            end_timestamp = end_time.timestamp()
        else:
            end_timestamp = end_time / 1e9
        
        return {
            "span_id": format(span_context.span_id, "016x"),
            "parent_id": format(parent_span_id, "016x") if parent_span_id else None,
            "trace_id": format(span_context.trace_id, "032x"),
            "name": span.name,
            "start_time": start_timestamp,
            "end_time": end_timestamp,
            "duration_ms": (end_timestamp - start_timestamp) * 1000,
            "attributes": attributes,
            "status": status,
            "service": service_name,
        }
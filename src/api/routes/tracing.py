"""
链路追踪API路由 - 提供链路查询和聚合接口
"""

from fastapi import APIRouter, Query
from typing import Optional, List, Dict, Any
from datetime import datetime
import time

from src.services.support.trace_storage_service import get_trace_storage_service

router = APIRouter(prefix="/api/tracing", tags=["tracing"])


@router.get("/traces", summary="获取追踪列表")
async def get_traces(
    limit: int = Query(50, ge=1, le=200, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量"),
):
    """
    获取追踪列表
    
    Args:
        limit: 返回数量限制
        offset: 偏移量
    
    Returns:
        追踪列表
    """
    storage = get_trace_storage_service()
    traces = storage.get_recent_traces(limit + offset)
    
    # 应用偏移量
    if offset > 0:
        traces = traces[offset:]
    
    return {
        "success": True,
        "data": traces,
        "count": len(traces),
        "total": storage.get_trace_count(),
    }


@router.get("/traces/{trace_id}", summary="获取单个追踪详情")
async def get_trace(trace_id: str):
    """
    获取单个追踪详情
    
    Args:
        trace_id: 追踪ID
    
    Returns:
        追踪详情
    """
    storage = get_trace_storage_service()
    trace = storage.get_trace(trace_id)
    
    if not trace:
        return {"success": False, "message": "追踪记录不存在", "data": None}
    
    return {"success": True, "data": trace}


@router.get("/traces/{trace_id}/tree", summary="获取追踪树形结构")
async def get_trace_tree(trace_id: str):
    """
    获取追踪树形结构（Span层级关系）
    
    Args:
        trace_id: 追踪ID
    
    Returns:
        追踪树形结构
    """
    storage = get_trace_storage_service()
    trace = storage.get_trace(trace_id)
    
    if not trace:
        return {"success": False, "message": "追踪记录不存在", "data": None}
    
    # 构建树形结构
    span_dict = {span["span_id"]: span for span in trace.get("spans", [])}
    
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
    
    return {"success": True, "data": root or {}}


@router.get("/traces/{trace_id}/stats", summary="获取追踪统计信息")
async def get_trace_stats(trace_id: str):
    """
    获取追踪统计信息
    
    Args:
        trace_id: 追踪ID
    
    Returns:
        追踪统计信息
    """
    storage = get_trace_storage_service()
    trace = storage.get_trace(trace_id)
    
    if not trace:
        return {"success": False, "message": "追踪记录不存在", "data": None}
    
    spans = trace.get("spans", [])
    
    # 计算统计信息
    total_duration = sum(span.get("duration_ms", 0) for span in spans)
    error_count = sum(1 for span in spans if span.get("status", {}).get("code") == "ERROR")
    
    # 按类型分组
    spans_by_kind = {}
    for span in spans:
        kind = span.get("kind", "UNKNOWN")
        if kind not in spans_by_kind:
            spans_by_kind[kind] = []
        spans_by_kind[kind].append(span)
    
    stats = {
        "trace_id": trace_id,
        "total_duration_ms": trace.get("duration_ms", 0),
        "span_count": len(spans),
        "error_count": error_count,
        "success_count": len(spans) - error_count,
        "span_kinds": {
            kind: {
                "count": len(span_list),
                "avg_duration_ms": round(
                    sum(s.get("duration_ms", 0) for s in span_list) / len(span_list),
                    2,
                ),
            }
            for kind, span_list in spans_by_kind.items()
        },
    }
    
    return {"success": True, "data": stats}


@router.delete("/traces/{trace_id}", summary="删除追踪记录")
async def delete_trace(trace_id: str):
    """
    删除追踪记录
    
    Args:
        trace_id: 追踪ID
    
    Returns:
        删除结果
    """
    storage = get_trace_storage_service()
    
    if not storage.get_trace(trace_id):
        return {"success": False, "message": "追踪记录不存在"}
    
    storage.delete_trace(trace_id)
    return {"success": True, "message": "追踪记录已删除"}


@router.delete("/traces", summary="批量删除追踪记录")
async def delete_all_traces():
    """
    批量删除所有追踪记录
    
    Returns:
        删除结果
    """
    storage = get_trace_storage_service()
    count = storage.get_trace_count()
    
    # 逐个删除（因为MemoryStorage没有批量删除方法）
    traces = storage.get_all_traces()
    for trace in traces:
        storage.delete_trace(trace["trace_id"])
    
    return {"success": True, "message": f"已删除 {count} 条追踪记录"}


@router.get("/stats", summary="获取追踪聚合统计")
async def get_tracing_stats(
    hours: int = Query(24, ge=1, le=72, description="统计时间范围（小时）"),
):
    """
    获取追踪聚合统计信息
    
    Args:
        hours: 统计时间范围（小时）
    
    Returns:
        聚合统计信息
    """
    storage = get_trace_storage_service()
    stats = storage.get_aggregated_stats(hours)
    
    return {"success": True, "data": stats}


@router.get("/search", summary="搜索追踪记录")
async def search_traces(
    endpoint: Optional[str] = Query(None, description="端点路径（模糊匹配）"),
    status: Optional[str] = Query(None, description="状态（OK/ERROR）"),
    min_duration_ms: Optional[float] = Query(None, description="最小持续时间（毫秒）"),
    max_duration_ms: Optional[float] = Query(None, description="最大持续时间（毫秒）"),
    limit: int = Query(50, ge=1, le=200, description="返回数量限制"),
):
    """
    搜索追踪记录
    
    Args:
        endpoint: 端点路径（模糊匹配）
        status: 状态（OK/ERROR）
        min_duration_ms: 最小持续时间（毫秒）
        max_duration_ms: 最大持续时间（毫秒）
        limit: 返回数量限制
    
    Returns:
        搜索结果
    """
    storage = get_trace_storage_service()
    traces = storage.search_traces(
        endpoint=endpoint,
        status=status,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
        limit=limit,
    )
    
    return {
        "success": True,
        "data": traces,
        "count": len(traces),
        "filters": {
            "endpoint": endpoint,
            "status": status,
            "min_duration_ms": min_duration_ms,
            "max_duration_ms": max_duration_ms,
        },
    }


@router.get("/health", summary="追踪服务健康检查")
async def tracing_health():
    """
    追踪服务健康检查
    
    Returns:
        健康状态
    """
    storage = get_trace_storage_service()
    
    try:
        count = storage.get_trace_count()
        return {
            "success": True,
            "status": "healthy",
            "trace_count": count,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@router.post("/cleanup", summary="清理过期追踪记录")
async def cleanup_expired_traces(
    max_age_hours: int = Query(24, ge=1, description="最大保留时长（小时）"),
):
    """
    清理过期追踪记录
    
    Args:
        max_age_hours: 最大保留时长（小时）
    
    Returns:
        清理结果
    """
    storage = get_trace_storage_service()
    storage.clear_expired_traces(max_age_hours)
    
    return {
        "success": True,
        "message": f"已清理超过 {max_age_hours} 小时的追踪记录",
        "remaining_count": storage.get_trace_count(),
    }

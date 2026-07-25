#!/usr/bin/env python3
"""
监控面板路由模块 - 集成到 API Gateway
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import json
from datetime import datetime
from pathlib import Path

router = APIRouter(prefix="/monitor", tags=["monitoring"])

templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/", response_class=HTMLResponse)
async def monitor_dashboard(request: Request):
    """监控面板主页"""
    services = await _get_services()
    stats = await _get_stats()
    traces = await _get_recent_traces()
    cleaning_summary, cleaning_tasks = await _get_cleaning_progress()

    nav_items = [
        {"url": "/monitor", "label": "仪表盘", "icon": "fa-tachometer-alt", "active": True},
        {"url": "/monitor/services", "label": "服务状态", "icon": "fa-server", "active": False},
        {"url": "/monitor/tracing", "label": "链路追踪", "icon": "fa-link", "active": False},
        {"url": "/monitor/cleaning", "label": "数据清理", "icon": "fa-trash-alt", "active": False},
        {"url": "/logs", "label": "日志查看", "icon": "fa-file-alt", "active": False},
    ]

    return templates.TemplateResponse("monitor/dashboard.html", {
        "request": request,
        "title": "系统监控",
        "page_title": "仪表盘",
        "page_description": "实时监控系统状态、服务健康、链路追踪和数据清理进度",
        "nav_items": nav_items,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "services": services,
        "stats": stats,
        "traces": traces,
        "cleaning_summary": cleaning_summary,
        "cleaning_tasks": cleaning_tasks,
    })


@router.get("/services", response_class=JSONResponse)
async def services_status():
    """获取所有服务状态"""
    return await _get_services()


@router.get("/tracing", response_class=HTMLResponse)
async def tracing_page(request: Request):
    """链路追踪页面"""
    traces = await _get_recent_traces()
    stats = await _get_stats()

    nav_items = [
        {"url": "/monitor/", "label": "仪表盘", "icon": "fa-tachometer-alt", "active": False},
        {"url": "/monitor/services", "label": "服务状态", "icon": "fa-server", "active": False},
        {"url": "/monitor/tracing", "label": "链路追踪", "icon": "fa-link", "active": True},
        {"url": "/monitor/cleaning", "label": "数据清理", "icon": "fa-trash-alt", "active": False},
        {"url": "/logs/", "label": "日志查看", "icon": "fa-file-alt", "active": False},
    ]

    return templates.TemplateResponse("monitor/tracing.html", {
        "request": request,
        "title": "链路追踪",
        "page_title": "链路追踪",
        "page_description": "查看请求链路追踪记录",
        "nav_items": nav_items,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "traces": traces,
        "stats": stats,
    })


@router.get("/tracing/recent", response_class=JSONResponse)
async def recent_traces():
    """获取最近追踪记录"""
    return await _get_recent_traces()


@router.get("/tracing/{trace_id}", response_class=HTMLResponse)
async def trace_detail(request: Request, trace_id: str):
    """查看追踪详情"""
    trace = await _get_trace(trace_id)
    return templates.TemplateResponse("monitor/trace_detail.html", {
        "request": request,
        "trace": trace,
        "trace_id": trace_id,
    })


@router.get("/cleaning/progress", response_class=JSONResponse)
async def cleaning_progress():
    """获取数据清理进度"""
    return await _get_cleaning_progress_raw()


@router.post("/cleaning/reset", response_class=JSONResponse)
async def reset_cleaning():
    """重置数据清理进度"""
    progress_file = Path("data/cleaning_progress.json")
    if progress_file.exists():
        progress_file.unlink()
    return {"status": "success", "message": "清理进度已重置"}


@router.get("/tracing/stats", response_class=JSONResponse)
async def tracing_stats(hours: int = 24):
    """获取追踪统计信息"""
    try:
        from src.services.support.trace_storage_service import get_trace_storage_service
        storage = get_trace_storage_service()
        return storage.get_aggregated_stats(hours)
    except Exception:
        return {
            "total_traces": 0,
            "success_count": 0,
            "error_count": 0,
            "error_rate": 0,
            "avg_duration_ms": 0,
            "max_duration_ms": 0,
            "time_range": hours,
        }


@router.get("/tracing/search", response_class=JSONResponse)
async def search_traces(endpoint: str = "", status: str = "", min_duration: int = 0, max_duration: int = 0):
    """搜索追踪记录"""
    try:
        from src.services.support.trace_storage_service import get_trace_storage_service
        storage = get_trace_storage_service()
        return storage.search_traces(endpoint, status, min_duration, max_duration)
    except Exception:
        return []


async def _get_services():
    """获取服务状态"""
    try:
        from src.core.config.service_config import get_service_config
        config = get_service_config()
        services_info = {
            "api-service": {"name": "API Service", "url": config.CORE_API_URL},
            "api-gateway": {"name": "API Gateway", "url": config.API_GATEWAY_URL},
            "model-service": {"name": "Model Service", "url": config.MODEL_SERVICE_URL},
            "search-service": {"name": "Search Service", "url": config.SEARCH_SERVICE_URL},
            "multimedia-service": {"name": "Multimedia Service", "url": config.MULTIMEDIA_SERVICE_URL},
        }
        import httpx
        results = []
        async with httpx.AsyncClient(timeout=5.0) as client:
            for name, info in services_info.items():
                try:
                    response = await client.get(f"{info['url']}/api/health")
                    results.append({
                        "name": info["name"],
                        "status": "healthy" if response.status_code == 200 else "unhealthy",
                        "url": info["url"],
                        "response_time": response.elapsed.total_seconds() * 1000,
                    })
                except Exception:
                    results.append({
                        "name": info["name"],
                        "status": "unhealthy",
                        "url": info["url"],
                        "response_time": 0,
                    })
        return results
    except Exception:
        return [
            {"name": "API Service", "status": "unhealthy", "url": "http://localhost:8001", "response_time": 0},
            {"name": "API Gateway", "status": "healthy", "url": "http://localhost:8080", "response_time": 0},
            {"name": "Model Service", "status": "unhealthy", "url": "http://localhost:8000", "response_time": 0},
            {"name": "Search Service", "status": "unhealthy", "url": "http://localhost:8003", "response_time": 0},
            {"name": "Multimedia Service", "status": "unhealthy", "url": "http://localhost:8002", "response_time": 0},
        ]


async def _get_stats():
    """获取统计数据"""
    traces_stats = await tracing_stats(24)
    return [
        {"label": "总请求数", "value": traces_stats.get("total_traces", 0), "icon": "fa-server", "color": "indigo"},
        {"label": "成功请求", "value": traces_stats.get("success_count", 0), "icon": "fa-check-circle", "color": "green"},
        {"label": "错误请求", "value": traces_stats.get("error_count", 0), "icon": "fa-times-circle", "color": "red"},
        {"label": "错误率", "value": f"{traces_stats.get('error_rate', 0)}%", "icon": "fa-exclamation-triangle", "color": "yellow"},
        {"label": "平均耗时", "value": f"{traces_stats.get('avg_duration_ms', 0)}ms", "icon": "fa-clock", "color": "blue"},
        {"label": "最大耗时", "value": f"{traces_stats.get('max_duration_ms', 0)}ms", "icon": "fa-tachometer-alt", "color": "purple"},
    ]


async def _get_recent_traces(limit: int = 10):
    """获取最近追踪记录"""
    try:
        from src.services.support.trace_storage_service import get_trace_storage_service
        storage = get_trace_storage_service()
        traces = storage.get_recent_traces(limit)
        return traces
    except Exception:
        return []


async def _get_trace(trace_id: str):
    """获取追踪详情"""
    try:
        from src.services.support.trace_storage_service import get_trace_storage_service
        storage = get_trace_storage_service()
        return storage.get_trace(trace_id)
    except Exception:
        return None


async def _get_cleaning_progress_raw():
    """获取原始清理进度数据"""
    progress_file = Path("data/cleaning_progress.json")
    if progress_file.exists():
        try:
            return json.loads(progress_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_processed": 0,
            "total_valid": 0,
            "total_rejected": 0,
            "total_duplicates": 0,
            "avg_confidence": 0,
            "avg_quality_score": 0,
        },
        "tasks": {},
    }


async def _get_cleaning_progress():
    """获取清理进度（格式化）"""
    data = await _get_cleaning_progress_raw()
    summary = data.get("summary", {})
    cleaning_summary = [
        {"label": "已处理", "value": summary.get("total_processed", 0)},
        {"label": "有效", "value": summary.get("total_valid", 0)},
        {"label": "已拒绝", "value": summary.get("total_rejected", 0)},
        {"label": "重复", "value": summary.get("total_duplicates", 0)},
        {"label": "置信度", "value": f"{summary.get('avg_confidence', 0):.2f}"},
        {"label": "质量分", "value": f"{summary.get('avg_quality_score', 0):.2f}"},
    ]
    tasks = data.get("tasks", {})
    cleaning_tasks = []
    for task_id, task in tasks.items():
        cleaning_tasks.append({
            "id": task_id,
            "name": task.get("name", task_id),
            "status": task.get("status", "pending"),
            "progress": task.get("progress", 0),
            "completed": task.get("completed", 0),
            "total": task.get("total", 0),
        })
    return cleaning_summary, cleaning_tasks

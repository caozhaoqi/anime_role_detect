#!/usr/bin/env python3
"""
日志查看器路由模块 - 集成到 API Gateway
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import re

router = APIRouter(prefix="/logs", tags=["logging"])

templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

project_root = Path(__file__).parent.parent.parent.parent.parent
LOG_DIR = project_root / "logs"


@router.get("/", response_class=HTMLResponse)
async def logs_index(request: Request):
    """日志查看器主页"""
    services = await _get_log_services()
    log_lines = await _get_log_lines("unified", 100)
    log_stats = await _get_log_stats()

    nav_items = [
        {"url": "/monitor", "label": "仪表盘", "icon": "fa-tachometer-alt", "active": False},
        {"url": "/monitor/services", "label": "服务状态", "icon": "fa-server", "active": False},
        {"url": "/monitor/tracing", "label": "链路追踪", "icon": "fa-link", "active": False},
        {"url": "/monitor/cleaning", "label": "数据清理", "icon": "fa-trash-alt", "active": False},
        {"url": "/logs", "label": "日志查看", "icon": "fa-file-alt", "active": True},
    ]

    return templates.TemplateResponse("logs/index.html", {
        "request": request,
        "title": "日志查看",
        "page_title": "日志查看器",
        "page_description": "实时查看和搜索系统日志",
        "nav_items": nav_items,
        "last_update": "",
        "services": services,
        "log_lines": log_lines,
        "log_stats": log_stats,
    })


@router.get("/tail", response_class=JSONResponse)
async def tail_log(service: str = "unified", lines: int = 100):
    """获取日志尾部"""
    log_lines = await _get_log_lines(service, lines)
    return {"lines": log_lines, "service": service}


@router.get("/services", response_class=JSONResponse)
async def log_services():
    """获取可用日志服务列表"""
    return await _get_log_services()


@router.get("/stats", response_class=JSONResponse)
async def log_stats():
    """获取日志统计"""
    return await _get_log_stats()


async def _get_log_services():
    """获取可用日志服务"""
    services = ["unified"]
    if LOG_DIR.exists():
        for path in LOG_DIR.rglob("*.log"):
            name = path.stem
            if name not in services:
                services.append(name)
    return services


async def _get_log_lines(service: str, lines: int = 100):
    """获取日志行"""
    if service == "unified":
        log_file = LOG_DIR / "anime_role_detect_unified.log"
        if not log_file.exists():
            log_file = LOG_DIR / "unified.log"
    else:
        log_file = LOG_DIR / f"{service}.log"

    if not log_file.exists():
        for path in LOG_DIR.rglob(f"{service}.log"):
            log_file = path
            break
        if not log_file.exists():
            for path in LOG_DIR.rglob(f"*_{service}.log"):
                log_file = path
                break

    if not log_file.exists():
        return ["日志文件不存在"]

    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
        all_lines = content.split("\n")
        return all_lines[-lines:]
    except Exception as e:
        return [f"读取日志失败: {e}"]


async def _get_log_stats():
    """获取日志统计"""
    unified_log = LOG_DIR / "anime_role_detect_unified.log"
    if not unified_log.exists():
        unified_log = LOG_DIR / "unified.log"
    stats = {
        "total_lines": 0,
        "error_count": 0,
        "warn_count": 0,
        "info_count": 0,
    }

    if unified_log.exists():
        try:
            content = unified_log.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")
            stats["total_lines"] = len(lines)
            for line in lines:
                upper = line.upper()
                if "[ERROR]" in upper:
                    stats["error_count"] += 1
                elif "[WARN]" in upper:
                    stats["warn_count"] += 1
                elif "[INFO]" in upper:
                    stats["info_count"] += 1
        except Exception:
            pass

    return [
        {"label": "总行数", "value": stats["total_lines"]},
        {"label": "ERROR", "value": stats["error_count"]},
        {"label": "WARN", "value": stats["warn_count"]},
        {"label": "INFO", "value": stats["info_count"]},
    ]

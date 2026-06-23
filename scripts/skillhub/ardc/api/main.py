#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 主入口 - 生产环境优化版
提供技能仓库 RESTful API，支持请求日志、全局异常处理、版本号比对、缓存优化等
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uuid
from datetime import datetime, timezone
from typing import List, Optional
import json

from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel


from ardc.api.auth import router as auth_router, get_current_developer, oauth2_scheme
from ardc.api.v1 import router as v1_router
from ardc.utils.logging import get_logger, get_request_logger, set_request_context
from ardc.config import settings

logger = get_logger(__name__)
request_logger = get_request_logger()

# 1. 初始化 FastAPI，禁用默认文档路径以避免冲突
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    docs_url=None,  # 禁用默认 Swagger
    redoc_url=None,  # 禁用默认 ReDoc
    openapi_url=None,  # 禁用默认 OpenAPI JSON
)

# 包含认证路由
app.include_router(auth_router)

# 包含 v1 版本路由
app.include_router(v1_router, prefix="/api")

# CORS 配置 - 使用统一配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.allowed_origins,
    allow_credentials=settings.cors.allow_credentials,
    allow_methods=settings.cors.allow_methods,
    allow_headers=settings.cors.allow_headers,
)


# ==================== 请求日志中间件 ====================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件 - 记录所有请求的详细信息，支持请求追踪"""
    request_id = str(uuid.uuid4())[:8]
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = str(request.url.path)

    # 从请求头获取追踪信息（支持分布式追踪）
    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    span_id = request.headers.get("X-Span-ID", str(uuid.uuid4())[:8])

    # 设置请求上下文（包含追踪信息）
    set_request_context(
        request_id=request_id, client_ip=client_ip, trace_id=trace_id, span_id=span_id
    )

    # 获取请求体大小
    try:
        body_size = int(request.headers.get("Content-Length", 0))
    except ValueError:
        body_size = 0

    logger.info(f"📥 [{request_id}] {method} {path} | IP: {client_ip} | Trace: {trace_id}")

    start_time = datetime.now()
    try:
        response = await call_next(request)
        duration = (datetime.now() - start_time).total_seconds() * 1000

        # 获取响应大小
        response_size = int(response.headers.get("Content-Length", 0))

        # 记录响应信息
        request_logger.log_request(
            method=method,
            path=path,
            client_ip=client_ip,
            status_code=response.status_code,
            duration=duration,
            size=response_size,
        )

        # 添加追踪信息到响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id

        return response
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(
            f"❌ [{request_id}] {method} {path} | 耗时: {duration:.2f}ms | 错误: {str(e)}",
            exc_info=True,
        )
        raise


# ==================== 全局异常处理器 ====================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器 - 统一处理未捕获的异常"""
    request_id = str(uuid.uuid4())[:8]
    logger.critical(
        f"💥 未捕获异常: [{request_id}] {request.method} {request.url} | 错误: {str(exc)}",
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "detail": "服务器内部错误",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP 异常处理器 - 统一处理 HTTP 错误"""
    logger.warning(
        f"⚠️ HTTP 错误: {request.method} {request.url} | 状态码: {exc.status_code} | 详情: {exc.detail}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "detail": exc.detail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求验证异常处理器"""
    logger.warning(f"⚠️ 请求验证失败: {request.method} {request.url} | 错误: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "detail": "请求参数验证失败",
            "errors": exc.errors(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ==================== 核心 OpenAPI 定义路由 ====================


@app.get("/api/openapi.json", include_in_schema=False)
async def custom_openapi_json():
    """生成并返回标准的 OpenAPI 定义 JSON"""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


@app.get("/api/docs", include_in_schema=False)
async def custom_swagger_ui_html(developer=Depends(get_current_developer)):
    """自定义 Swagger UI 页面，仅开发者可访问"""
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",  # 必须与上面的路由一致
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        # 使用国内更稳定的 CDN 地址
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
    )


@app.get("/api/redoc", include_in_schema=False)
async def custom_redoc_ui_html(developer=Depends(get_current_developer)):
    """自定义 ReDoc 页面，仅开发者可访问"""
    return get_redoc_html(openapi_url="/api/openapi.json", title=app.title + " - ReDoc")


# ==================== API 版本端点 ====================
@app.get("/api/v1/")
def api_v1_root():
    """API v1 根端点 - 重定向到当前版本"""
    return {
        "version": "1.0.0",
        "message": "欢迎使用 ARD Skill Hub API v1",
        "endpoints": {
            "skills": "/api/skills",
            "search": "/api/search",
            "categories": "/api/categories",
            "stats": "/api/stats",
        },
    }


@app.get("/api/v2/")
def api_v2_root():
    """API v2 根端点 - 当前版本"""
    return {
        "version": "2.0.0",
        "message": "欢迎使用 ARD Skill Hub API v2",
        "endpoints": {
            "skills": "/api/skills",
            "search": "/api/search",
            "categories": "/api/categories",
            "stats": "/api/stats",
            "favorites": "/api/favorites",
        },
        "features": ["技能管理", "搜索功能", "分类浏览", "统计数据", "收藏功能"],
    }


# ==================== 通知 API ====================
@app.post("/api/notifications/check-updates")
def check_notifications_updates():
    """检查通知更新（公开端点）"""
    return {
        "has_update": False,
        "latest_version": "1.0.0",
        "message": "当前已是最新版本",
        "notification_count": 0,
    }


# ==================== 更新日志 API ====================

from ardc.store.changelog import ChangelogStore, ChangelogEntry

changelog_store = ChangelogStore()


class ChangelogCreate(BaseModel):
    version: str
    title: str
    description: str
    changes: List[str]
    author: Optional[str] = None
    is_major: bool = False
    affected_components: Optional[List[str]] = None


@app.get("/api/changelog")
def get_changelog(limit: int = 20, component: Optional[str] = None):
    entries = (
        changelog_store.get_entries_by_component(component)
        if component
        else changelog_store.get_all_entries(limit=limit)
    )
    return {"total": len(entries), "entries": [e.dict() for e in entries]}


@app.post("/api/changelog")
def add_changelog(entry: ChangelogCreate, developer=Depends(get_current_developer)):
    try:
        new_entry = ChangelogEntry(
            version=entry.version,
            title=entry.title,
            description=entry.description,
            changes=entry.changes,
            release_date=datetime.now().strftime("%Y-%m-%d"),
            author=entry.author or developer.username,
            is_major=entry.is_major,
            affected_components=entry.affected_components,
        )
        changelog_store.add_entry(new_entry)
        return {"message": "更新日志添加成功", "version": entry.version}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== 日志查看 API ====================
def _tail_file(filepath: str, num_lines: int = 1000) -> list:
    """高效读取文件末尾指定行数"""
    lines = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # 先尝试用 seek 定位到文件末尾附近
            buffer_size = 8192
            f.seek(0, os.SEEK_END)
            file_size = f.tell()

            if file_size == 0:
                return lines

            # 从文件末尾向前读取，直到获取足够的行数
            while len(lines) < num_lines:
                # 计算要读取的字节数
                read_size = min(buffer_size, file_size)
                if read_size == 0:
                    break

                f.seek(file_size - read_size)
                chunk = f.read(read_size)
                file_size -= read_size

                # 拆分行为列表
                chunk_lines = chunk.split("\n")

                # 如果不是文件开头，第一个元素可能不完整，需要和之前的内容合并
                if file_size > 0 and lines:
                    lines[0] = chunk_lines[-1] + lines[0]
                    chunk_lines = chunk_lines[:-1]

                # 添加到结果列表前面
                lines = chunk_lines + lines

            # 只返回需要的行数
            return lines[-num_lines:]
    except Exception as e:
        logger.error(f"读取日志文件失败: {filepath}, 错误: {e}")
        return lines


@app.get("/api/logs")
async def get_logs(
    developer=Depends(get_current_developer),
    level: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    from ardc.utils.logging import LogConfig
    import os

    all_logs = []
    log_dir = LogConfig.LOG_DIR
    if not os.path.exists(log_dir):
        return {"logs": [], "total": 0}

    # 限制一次最多读取的行数，防止内存溢出
    max_lines_per_file = 5000

    log_files = sorted(
        [f for f in os.listdir(log_dir) if f.endswith(".log") and not f.endswith("_json.log")]
    )
    for filename in log_files:
        filepath = os.path.join(log_dir, filename)
        try:
            # 高效读取文件末尾行
            lines = _tail_file(filepath, max_lines_per_file)
            for line in lines:
                parts = line.strip().split(" - ", 4)
                if len(parts) == 5:
                    all_logs.append(
                        {
                            "timestamp": parts[0],
                            "logger": parts[1],
                            "level": parts[2],
                            "source": parts[3],
                            "message": parts[4],
                        }
                    )
        except Exception as e:
            logger.error(f"处理日志文件失败: {filepath}, 错误: {e}")

    if level:
        all_logs = [log for log in all_logs if log["level"] == level.upper()]
    if keyword:
        all_logs = [log for log in all_logs if keyword.lower() in log["message"].lower()]

    all_logs.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"logs": all_logs[offset : offset + limit], "total": len(all_logs)}


# ==================== 收藏夹 API ====================
@app.get("/api/favorites")
def get_favorites(
    developer=Depends(get_current_developer), token: Optional[str] = Depends(oauth2_scheme)
):
    """获取用户收藏的技能（支持匿名访问）"""
    # 如果未登录（token无效或没有developer），返回空列表
    if not developer:
        return {
            "favorites": [],
            "message": "请登录以查看个人收藏",
            "hint": "使用 POST /api/auth/login 登录",
        }

    favorites_file = Path.home() / ".ardc" / "favorites.json"
    if favorites_file.exists():
        with open(favorites_file, "r", encoding="utf-8") as f:
            return {"favorites": json.load(f).get(developer.username, [])}
    return {"favorites": []}


@app.get("/api/favorites/public")
def get_public_favorites():
    """获取公开收藏列表（无需认证）"""
    favorites_file = Path.home() / ".ardc" / "favorites.json"
    if favorites_file.exists():
        with open(favorites_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 返回所有用户的收藏数量统计
            return {"total_users": len(data), "total_favorites": sum(len(v) for v in data.values())}
    return {"total_users": 0, "total_favorites": 0, "favorites": []}


@app.get("/api/favorites/anonymous")
def get_anonymous_favorites():
    """匿名用户获取收藏列表（返回公共收藏或空列表）"""
    return {
        "favorites": [],
        "message": "请登录以查看个人收藏",
        "hint": "使用 POST /api/auth/login 登录",
    }


@app.post("/api/favorites/{skill_id}")
def add_favorite(skill_id: str, developer=Depends(get_current_developer)):
    """添加收藏（需要开发者权限）"""
    # 先验证技能是否存在
    skill = registry.get_skill_by_version(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail=f"技能不存在: {skill_id}")

    fav_path = Path.home() / ".ardc" / "favorites.json"
    fav_path.parent.mkdir(parents=True, exist_ok=True)
    favs = {}
    if fav_path.exists():
        with open(fav_path, "r", encoding="utf-8") as f:
            favs = json.load(f)
    if developer.username not in favs:
        favs[developer.username] = []
    if skill_id not in favs[developer.username]:
        favs[developer.username].append(skill_id)
    with open(fav_path, "w", encoding="utf-8") as f:
        json.dump(favs, f, indent=2)
    return {"message": "收藏成功", "skill_id": skill_id}


@app.post("/api/favorites/{skill_id}/favorite")
def add_favorite_alt(skill_id: str, developer=Depends(get_current_developer)):
    """添加收藏（备用路径，兼容旧版客户端）"""
    return add_favorite(skill_id, developer)


# ==================== 健康检查 API ====================
@app.get("/api/health")
def health_check():
    """健康检查端点 - 用于服务状态监控"""
    return {"status": "healthy", "service": "ARD Skill Hub API", "version": "1.0.0"}


# ==================== CLI 安装脚本 API ====================
# 项目根目录: ardc/api/main.py -> ardc/ -> scripts/skillhub/
SKILLHUB_ROOT = Path(__file__).parent.parent.parent
INSTALL_SCRIPT_PATH = SKILLHUB_ROOT / "sh" / "install.sh"
CLI_SCRIPT_PATH = SKILLHUB_ROOT / "cli.py"

logger.info(f"SKILLHUB_ROOT: {SKILLHUB_ROOT}")
logger.info(f"INSTALL_SCRIPT_PATH exists: {INSTALL_SCRIPT_PATH.exists()}")


@app.get("/api/install/install.sh")
def get_install_script():
    """获取 CLI 安装脚本 (Bash)"""
    logger.info(f"安装脚本路径: {INSTALL_SCRIPT_PATH}, 存在: {INSTALL_SCRIPT_PATH.exists()}")
    if INSTALL_SCRIPT_PATH.exists():
        content = INSTALL_SCRIPT_PATH.read_text(encoding="utf-8")
        return PlainTextResponse(content=content, media_type="text/plain")
    raise HTTPException(status_code=404, detail="安装脚本不存在")


@app.get("/api/install/cli.py")
def get_cli_script():
    """获取 CLI 工具脚本"""
    logger.info(f"CLI脚本路径: {CLI_SCRIPT_PATH}, 存在: {CLI_SCRIPT_PATH.exists()}")
    if CLI_SCRIPT_PATH.exists():
        content = CLI_SCRIPT_PATH.read_text(encoding="utf-8")
        return PlainTextResponse(content=content, media_type="text/plain")
    raise HTTPException(status_code=404, detail="CLI 脚本不存在")


@app.get("/api/install/install.ps1")
def get_install_script_ps():
    """获取 CLI 安装脚本 (PowerShell)"""
    ps_script = """# ARD Skill Hub CLI 安装脚本 (PowerShell)
# 需要以管理员身份运行

$ARD_INSTALL_DIR = "$env:USERPROFILE\\.ardc"
$ARD_BIN_DIR = "$ARD_INSTALL_DIR\\bin"
$ARD_CLI_URL = "http://47.79.91.89:8888/api/install/cli.py"

Write-Host "🚀 正在安装 ARD Skill Hub CLI 工具..."

# 检查 Python
if (-not (Get-Command python3 -ErrorAction SilentlyContinue)) {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Host "❌ 错误: 未找到 Python 3"
        exit 1
    }
}

# 创建目录
New-Item -ItemType Directory -Path $ARD_BIN_DIR -Force | Out-Null

# 下载 CLI
Write-Host "📥 下载 CLI 工具..."
Invoke-WebRequest -Uri $ARD_CLI_URL -OutFile "$ARD_BIN_DIR\\ardc.py"

# 设置 PATH
$path = [Environment]::GetEnvironmentVariable("Path", "User")
if ($path -notlike "*$ARD_BIN_DIR*") {
    [Environment]::SetEnvironmentVariable("Path", "$path;$ARD_BIN_DIR", "User")
}

Write-Host "✅ 安装完成!"
Write-Host "请重新打开 PowerShell 后运行: ardc --version"
"""
    return JSONResponse(content=ps_script)


# ==================== 启动 ====================


def start_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()

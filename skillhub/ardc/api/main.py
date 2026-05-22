#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 主入口 - 生产环境优化版
提供技能仓库 RESTful API，支持请求日志、全局异常处理、版本号比对等
"""

import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from packaging.version import parse as parse_version

from ardc.store.registry import SkillRegistry
from ardc.store.index import SkillIndex
from ardc.version.manager import VersionManager
from ardc.api.auth import router as auth_router, get_current_user, get_current_developer, oauth2_scheme
from ardc.utils.logging import get_logger, get_request_logger, set_request_context

logger = get_logger(__name__)
request_logger = get_request_logger()

# 1. 初始化 FastAPI，禁用默认文档路径以避免冲突
app = FastAPI(
    title="ARD Skill Repository API",
    version="1.0.0",
    description="技能仓库 RESTful API - 提供技能管理、用户认证、技能搜索等功能",
    docs_url=None,      # 禁用默认 Swagger
    redoc_url=None,     # 禁用默认 ReDoc
    openapi_url=None    # 禁用默认 OpenAPI JSON
)

# 包含认证路由
app.include_router(auth_router)

# CORS 配置 - 生产环境必须设置 ALLOWED_ORIGINS 环境变量
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
# 过滤空字符串
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()]

# 如果没有配置允许的域名，默认允许本地开发环境
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ==================== 请求日志中间件 ====================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件 - 记录所有请求的详细信息"""
    request_id = str(uuid.uuid4())[:8]
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = str(request.url.path)
    
    # 设置请求上下文
    set_request_context(request_id=request_id, client_ip=client_ip)
    
    logger.info(f"📥 收到请求: [{request_id}] {method} {path} | IP: {client_ip}")
    
    start_time = datetime.now()
    try:
        response = await call_next(request)
        duration = (datetime.now() - start_time).total_seconds() * 1000
        
        # 记录响应信息
        request_logger.log_request(
            method=method,
            path=path,
            client_ip=client_ip,
            status_code=response.status_code,
            duration=duration
        )
        
        return response
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"❌ 请求异常: [{request_id}] {method} {path} | 耗时: {duration:.2f}ms | 错误: {str(e)}", exc_info=True)
        raise

# ==================== 全局异常处理器 ====================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器 - 统一处理未捕获的异常"""
    request_id = str(uuid.uuid4())[:8]
    logger.critical(f"💥 未捕获异常: [{request_id}] {request.method} {request.url} | 错误: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "detail": "服务器内部错误",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP 异常处理器 - 统一处理 HTTP 错误"""
    logger.warning(f"⚠️ HTTP 错误: {request.method} {request.url} | 状态码: {exc.status_code} | 详情: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "detail": exc.detail,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

registry = SkillRegistry()
index = SkillIndex()
version_manager = VersionManager()

class SkillCreate(BaseModel):
    id: str
    name: str
    version: str
    description: Optional[str] = ""
    author: str
    category: str
    entry_point: str
    tags: Optional[List[str]] = []
    release_notes: Optional[str] = ""

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
        openapi_url="/api/openapi.json", # 必须与上面的路由一致
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        # 使用国内更稳定的 CDN 地址
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/api/redoc", include_in_schema=False)
async def custom_redoc_ui_html(developer=Depends(get_current_developer)):
    """自定义 ReDoc 页面，仅开发者可访问"""
    return get_redoc_html(
        openapi_url="/api/openapi.json",
        title=app.title + " - ReDoc"
    )

# ==================== 业务 API ====================

@app.get("/api/skills")
def list_skills(category: Optional[str] = None):
    skills = index.get_by_category(category) if category else index.get_all_skills()
    return {"skills": [s.dict() for s in skills]}

@app.get("/api/skills/{skill_id}")
def get_skill(skill_id: str, version: Optional[str] = None):
    skill = registry.get_skill_by_version(skill_id, version)
    if not skill:
        raise HTTPException(status_code=404, detail="技能不存在")
    return skill.dict()

@app.post("/api/skills")
def create_skill(skill: SkillCreate):
    logger.info(f"🎯 创建技能请求: {skill.id} - {skill.name}")
    from ardc.store.metadata import SkillMetadata
    try:
        metadata = SkillMetadata(
            id=skill.id, name=skill.name, version=skill.version,
            description=skill.description, author=skill.author,
            category=skill.category, entry_point=skill.entry_point, tags=skill.tags
        )
        registry.register_skill(metadata, skill.release_notes)
        index.add_skill(metadata)
        version_manager.release_version(metadata, skill.release_notes)
        return {"message": "技能注册成功", "skill_id": skill.id}
    except Exception as e:
        logger.error(f"❌ 技能注册失败: {skill.id}, 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/skills/{skill_id}")
def delete_skill(skill_id: str):
    index.remove_skill(skill_id)
    return {"message": "技能删除成功"}

@app.get("/api/skills/{skill_id}/versions")
def get_skill_versions(skill_id: str):
    versions = version_manager.list_versions(skill_id)
    return {"versions": [v.dict() for v in versions]}

@app.get("/api/skills/{skill_id}/check-update")
def check_skill_update(skill_id: str, current_version: str = None):
    """检查技能是否有更新"""
    try:
        latest = registry.get_latest_version(skill_id)
        if not latest:
            return {"has_update": False, "latest_version": current_version or "1.0.0"}
        
        has_update = False
        if current_version:
            try:
                # 使用 packaging.version 进行标准的版本号比对
                curr_version = parse_version(current_version)
                latest_version = parse_version(latest.version)
                has_update = latest_version > curr_version
                logger.debug(f"版本比对: 当前={current_version}, 最新={latest.version}, 有更新={has_update}")
            except Exception as e:
                logger.warning(f"版本号解析失败: {current_version} 或 {latest.version}, 错误: {str(e)}")
                has_update = False
        
        return {
            "has_update": has_update,
            "current_version": current_version,
            "latest_version": latest.version,
            "changelog": latest.release_notes if hasattr(latest, 'release_notes') else ""
        }
    except Exception as e:
        logger.error(f"检查更新失败: {skill_id}, 错误: {str(e)}")
        raise HTTPException(status_code=500, detail="检查更新失败")

@app.get("/api/search")
def search_skills(keyword: str, category: Optional[str] = None, limit: int = 20):
    try:
        results = index.search(keyword, category, limit=limit)
        return {"total": len(results), "skills": [s.dict() for s in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
def get_stats():
    return index.get_statistics()

@app.get("/api/categories")
def get_categories():
    """获取所有技能分类"""
    categories = index._index.get("categories", {})
    return {"categories": [{"name": name, "count": count} for name, count in categories.items()]}

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "ARD Skill Hub API", "version": "1.0.0"}

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
    entries = changelog_store.get_entries_by_component(component) if component else changelog_store.get_all_entries(limit=limit)
    return {"total": len(entries), "entries": [e.dict() for e in entries]}

@app.post("/api/changelog")
def add_changelog(entry: ChangelogCreate, developer=Depends(get_current_developer)):
    try:
        new_entry = ChangelogEntry(
            version=entry.version, title=entry.title, description=entry.description,
            changes=entry.changes, release_date=datetime.now().strftime("%Y-%m-%d"),
            author=entry.author or developer.username, is_major=entry.is_major,
            affected_components=entry.affected_components
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
        with open(filepath, 'r', encoding='utf-8') as f:
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
                chunk_lines = chunk.split('\n')
                
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
    offset: int = 0
):
    from ardc.utils.logging import LogConfig
    import os
    all_logs = []
    log_dir = LogConfig.LOG_DIR
    if not os.path.exists(log_dir):
        return {"logs": [], "total": 0}
    
    # 限制一次最多读取的行数，防止内存溢出
    max_lines_per_file = 5000
    
    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.log') and not f.endswith('_json.log')])
    for filename in log_files:
        filepath = os.path.join(log_dir, filename)
        try:
            # 高效读取文件末尾行
            lines = _tail_file(filepath, max_lines_per_file)
            for line in lines:
                parts = line.strip().split(' - ', 4)
                if len(parts) == 5:
                    all_logs.append({
                        'timestamp': parts[0], 
                        'logger': parts[1], 
                        'level': parts[2], 
                        'source': parts[3], 
                        'message': parts[4]
                    })
        except Exception as e:
            logger.error(f"处理日志文件失败: {filepath}, 错误: {e}")
    
    if level:
        all_logs = [log for log in all_logs if log['level'] == level.upper()]
    if keyword:
        all_logs = [log for log in all_logs if keyword.lower() in log['message'].lower()]
    
    all_logs.sort(key=lambda x: x['timestamp'], reverse=True)
    return {"logs": all_logs[offset:offset + limit], "total": len(all_logs)}

# ==================== 收藏夹 API ====================
@app.get("/api/favorites")
def get_favorites(developer=Depends(get_current_developer)):
    """获取用户收藏的技能（需要开发者权限）"""
    favorites_file = Path.home() / ".ardc" / "favorites.json"
    if favorites_file.exists():
        with open(favorites_file, 'r', encoding='utf-8') as f:
            return {"favorites": json.load(f).get(developer.username, [])}
    return {"favorites": []}

@app.get("/api/favorites/public")
def get_public_favorites():
    """获取公开收藏列表（无需认证）"""
    favorites_file = Path.home() / ".ardc" / "favorites.json"
    if favorites_file.exists():
        with open(favorites_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 返回所有用户的收藏数量统计
            return {"total_users": len(data), "total_favorites": sum(len(v) for v in data.values())}
    return {"total_users": 0, "total_favorites": 0, "favorites": []}

@app.post("/api/favorites/{skill_id}")
def add_favorite(skill_id: str, developer=Depends(get_current_developer)):
    fav_path = Path.home() / ".ardc" / "favorites.json"
    fav_path.parent.mkdir(parents=True, exist_ok=True)
    favs = {}
    if fav_path.exists():
        with open(fav_path, 'r', encoding='utf-8') as f: favs = json.load(f)
    if developer.username not in favs: favs[developer.username] = []
    if skill_id not in favs[developer.username]: favs[developer.username].append(skill_id)
    with open(fav_path, 'w', encoding='utf-8') as f: json.dump(favs, f, indent=2)
    return {"message": "收藏成功"}

# ==================== CLI 安装脚本 API ====================
# 项目根目录: ardc/api/main.py -> ardc/ -> skillhub/
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
        content = INSTALL_SCRIPT_PATH.read_text(encoding='utf-8')
        return PlainTextResponse(content=content, media_type="text/plain")
    raise HTTPException(status_code=404, detail="安装脚本不存在")

@app.get("/api/install/cli.py")
def get_cli_script():
    """获取 CLI 工具脚本"""
    logger.info(f"CLI脚本路径: {CLI_SCRIPT_PATH}, 存在: {CLI_SCRIPT_PATH.exists()}")
    if CLI_SCRIPT_PATH.exists():
        content = CLI_SCRIPT_PATH.read_text(encoding='utf-8')
        return PlainTextResponse(content=content, media_type="text/plain")
    raise HTTPException(status_code=404, detail="CLI 脚本不存在")

@app.get("/api/install/install.ps1")
def get_install_script_ps():
    """获取 CLI 安装脚本 (PowerShell)"""
    ps_script = '''# ARD Skill Hub CLI 安装脚本 (PowerShell)
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
'''
    return JSONResponse(content=ps_script)

# ==================== 启动 ====================

def start_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 主入口 - 修复版
提供技能仓库 RESTful API，修复 Swagger UI 渲染异常问题
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import json
from datetime import datetime

from ardc.store.registry import SkillRegistry
from ardc.store.index import SkillIndex
from ardc.version.manager import VersionManager
from ardc.api.auth import router as auth_router, get_current_user, get_current_developer, oauth2_scheme
from ardc.utils.logging import get_logger

logger = get_logger(__name__)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    try:
        latest = registry.get_latest_version(skill_id)
        if not latest:
            return {"has_update": False, "latest_version": current_version or "1.0.0"}
        
        has_update = False
        if current_version:
            try:
                curr = [int(x) for x in current_version.split('.')]
                late = [int(x) for x in latest.version.split('.')]
                has_update = late > curr
            except: pass
        
        return {
            "has_update": has_update,
            "current_version": current_version,
            "latest_version": latest.version,
            "changelog": latest.release_notes if hasattr(latest, 'release_notes') else ""
        }
    except:
        return {"has_update": False, "current_version": current_version}

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
    
    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.log') and not f.endswith('_json.log')])
    for filename in log_files:
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' - ', 4)
                    if len(parts) == 5:
                        all_logs.append({'timestamp': parts[0], 'logger': parts[1], 'level': parts[2], 'source': parts[3], 'message': parts[4]})
        except: pass
    
    if level:
        all_logs = [log for log in all_logs if log['level'] == level.upper()]
    if keyword:
        all_logs = [log for log in all_logs if keyword.lower() in log['message'].lower()]
    
    all_logs.sort(key=lambda x: x['timestamp'], reverse=True)
    return {"logs": all_logs[offset:offset + limit], "total": len(all_logs)}

# ==================== 收藏夹 API ====================
@app.get("/api/favorites")
def get_favorites(developer=Depends(get_current_developer)):
    favorites_file = Path.home() / ".ardc" / "favorites.json"
    if favorites_file.exists():
        with open(favorites_file, 'r', encoding='utf-8') as f:
            return {"favorites": json.load(f).get(developer.username, [])}
    return {"favorites": []}

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

# ==================== 启动 ====================

def start_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
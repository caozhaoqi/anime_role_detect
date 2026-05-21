#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 主入口
提供技能仓库 RESTful API
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import json

from ardc.store.registry import SkillRegistry
from ardc.store.index import SkillIndex
from ardc.version.manager import VersionManager
from ardc.api.auth import router as auth_router, get_current_developer, oauth2_scheme
from ardc.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="ARD Skill Repository API",
    version="1.0.0",
    description="技能仓库 RESTful API - 提供技能管理、用户认证、技能搜索等功能",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

logger.info("✅ ARD Skill Hub API 初始化完成")

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
            id=skill.id,
            name=skill.name,
            version=skill.version,
            description=skill.description,
            author=skill.author,
            category=skill.category,
            entry_point=skill.entry_point,
            tags=skill.tags
        )
        
        registry.register_skill(metadata, skill.release_notes)
        index.add_skill(metadata)
        version_manager.release_version(metadata, skill.release_notes)
        
        logger.info(f"✅ 技能注册成功: {skill.id}")
        return {"message": "技能注册成功", "skill_id": skill.id}
    except Exception as e:
        logger.error(f"❌ 技能注册失败: {skill.id}, 错误: {str(e)}")
        raise

@app.delete("/api/skills/{skill_id}")
def delete_skill(skill_id: str):
    index.remove_skill(skill_id)
    return {"message": "技能删除成功"}

@app.get("/api/skills/{skill_id}/versions")
def get_versions(skill_id: str):
    versions = version_manager.list_versions(skill_id)
    return {"versions": [v.dict() for v in versions]}

@app.post("/api/skills/{skill_id}/install")
def install_skill(skill_id: str, version: Optional[str] = None):
    logger.info(f"🎯 安装技能请求: {skill_id} (版本: {version or '最新'})")
    try:
        if registry.install_skill(skill_id, version):
            logger.info(f"✅ 技能安装成功: {skill_id}")
            return {"message": "技能安装成功"}
        logger.warning(f"⚠️ 技能安装失败: {skill_id}")
        raise HTTPException(status_code=400, detail="安装失败")
    except Exception as e:
        logger.error(f"❌ 技能安装异常: {skill_id}, 错误: {str(e)}")
        raise

@app.delete("/api/skills/{skill_id}/uninstall")
def uninstall_skill(skill_id: str):
    logger.info(f"🎯 卸载技能请求: {skill_id}")
    try:
        if registry.uninstall_skill(skill_id):
            logger.info(f"✅ 技能卸载成功: {skill_id}")
            return {"message": "技能卸载成功"}
        logger.warning(f"⚠️ 技能卸载失败: {skill_id}")
        raise HTTPException(status_code=400, detail="卸载失败")
    except Exception as e:
        logger.error(f"❌ 技能卸载异常: {skill_id}, 错误: {str(e)}")
        raise

@app.get("/api/skills/{skill_id}/versions")
def get_skill_versions(skill_id: str):
    """获取技能版本历史"""
    versions = version_manager.list_versions(skill_id)
    return {"versions": [v.dict() for v in versions]}

@app.get("/api/skills/{skill_id}/check-update")
def check_skill_update(skill_id: str, current_version: str = None):
    """检查技能更新"""
    try:
        latest = registry.get_latest_version(skill_id)
        if not latest:
            return {
                "has_update": False,
                "current_version": current_version,
                "latest_version": current_version or "1.0.0",
                "changelog": ""
            }
        
        has_update = False
        if current_version:
            try:
                current_parts = [int(x) for x in current_version.split('.')]
                latest_parts = [int(x) for x in latest.version.split('.')]
                has_update = latest_parts > current_parts
            except:
                pass
        
        return {
            "has_update": has_update,
            "current_version": current_version,
            "latest_version": latest.version,
            "changelog": latest.release_notes if hasattr(latest, 'release_notes') else ""
        }
    except Exception as e:
        logger.warning(f"⚠️ 检查更新失败: {skill_id}, 错误: {str(e)}")
        return {
            "has_update": False,
            "current_version": current_version,
            "latest_version": current_version or "1.0.0",
            "changelog": ""
        }

@app.get("/api/skills/{skill_id}/rating")
def get_skill_rating(skill_id: str):
    """获取技能评分"""
    return {"rating": 4.5, "count": 10}

@app.get("/api/skills/{skill_id}/reviews")
def get_skill_reviews(skill_id: str):
    """获取技能评论"""
    return {"reviews": []}

@app.post("/api/skills/{skill_id}/review")
def submit_skill_review(
    skill_id: str,
    rating: int = Query(ge=1, le=5),
    comment: str = Query(default=""),
    token: str = Depends(oauth2_scheme)
):
    """提交技能评论"""
    logger.info(f"📝 用户提交评论: 技能={skill_id}, 评分={rating}")
    return {"success": True, "message": "评论成功"}

@app.get("/api/search")
def search_skills(keyword: str, category: Optional[str] = None, limit: int = 20):
    logger.info(f"🔍 搜索技能: 关键词='{keyword}', 分类='{category}', 限制={limit}")
    try:
        results = index.search(keyword, category, limit=limit)
        logger.info(f"🔍 搜索完成: 找到 {len(results)} 个技能")
        return {"total": len(results), "skills": [s.dict() for s in results]}
    except Exception as e:
        logger.error(f"❌ 搜索失败: 关键词='{keyword}', 错误: {str(e)}")
        raise

@app.get("/api/tags")
def get_tags():
    return index.get_tags()

@app.get("/api/categories")
def get_categories():
    return index.get_categories()

@app.get("/api/stats")
def get_stats():
    return index.get_statistics()

@app.get("/api/install/install.sh")
def get_install_script():
    install_script_path = Path(__file__).parent.parent.parent / "install.sh"
    if not install_script_path.exists():
        raise HTTPException(status_code=404, detail="安装脚本不存在")
    return FileResponse(install_script_path, media_type="text/x-shellscript")

@app.get("/api/install/install.bat")
def get_install_bat():
    install_bat_path = Path(__file__).parent.parent.parent / "install.bat"
    if not install_bat_path.exists():
        raise HTTPException(status_code=404, detail="Windows 安装脚本不存在")
    return FileResponse(install_bat_path, media_type="text/plain")

@app.get("/api/install/cli.py")
def get_cli():
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    if not cli_path.exists():
        raise HTTPException(status_code=404, detail="CLI 工具不存在")
    return FileResponse(cli_path, media_type="text/x-python")

@app.get("/docs")
async def custom_swagger_ui_html(developer=Depends(get_current_developer)):
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="ARD Skill Repository API - Swagger UI"
    )

@app.get("/redoc")
async def custom_redoc_html(developer=Depends(get_current_developer)):
    from fastapi.openapi.docs import get_redoc_html
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="ARD Skill Repository API - ReDoc"
    )

@app.get("/openapi.json")
async def custom_openapi(developer=Depends(get_current_developer)):
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ARD Skill Hub API",
        "version": "1.0.0"
    }

# ==================== 日志查看 API ====================
@app.get("/api/logs")
async def get_logs(
    developer=Depends(get_current_developer),
    level: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    获取日志（仅开发者可访问）
    
    Args:
        level: 日志级别过滤 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        keyword: 关键词过滤
        limit: 返回条数限制
        offset: 偏移量
    """
    logger.info(f"🔍 开发者 {developer.username} 查看日志: level={level}, keyword={keyword}")
    
    from ardc.utils.logging import LogConfig
    import os
    
    all_logs = []
    log_dir = LogConfig.LOG_DIR
    
    if not os.path.exists(log_dir):
        return {"logs": [], "total": 0}
    
    # 读取所有日志文件
    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.log') and not f.endswith('_json.log')])
    
    for filename in log_files:
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    # 解析日志行
                    # 格式: 2026-05-21 10:30:45,123 - logger - LEVEL - module:line - message
                    parts = line.strip().split(' - ', 4)
                    if len(parts) == 5:
                        log_entry = {
                            'timestamp': parts[0],
                            'logger': parts[1],
                            'level': parts[2],
                            'source': parts[3],
                            'message': parts[4],
                            'file': filename
                        }
                        all_logs.append(log_entry)
        except Exception as e:
            logger.error(f"❌ 读取日志文件失败: {filename}, 错误: {str(e)}")
    
    # 按级别过滤
    if level:
        all_logs = [log for log in all_logs if log['level'] == level.upper()]
    
    # 按关键词过滤
    if keyword:
        all_logs = [log for log in all_logs if keyword.lower() in log['message'].lower()]
    
    # 按时间排序（最新的在前）
    all_logs.sort(key=lambda x: x['timestamp'], reverse=True)
    
    total = len(all_logs)
    paginated_logs = all_logs[offset:offset + limit]
    
    return {
        "logs": paginated_logs,
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.get("/api/logs/stats")
async def get_log_stats(developer=Depends(get_current_developer)):
    """
    获取日志统计信息（仅开发者可访问）
    """
    logger.info(f"📊 开发者 {developer.username} 获取日志统计")
    
    from ardc.utils.logging import LogConfig
    import os
    
    stats = {
        'total': 0,
        'levels': {'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0},
        'files': []
    }
    
    log_dir = LogConfig.LOG_DIR
    
    if not os.path.exists(log_dir):
        return stats
    
    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.log') and not f.endswith('_json.log')])
    
    for filename in log_files:
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                file_stats = {
                    'filename': filename,
                    'line_count': len(lines),
                    'size': os.path.getsize(filepath)
                }
                stats['files'].append(file_stats)
                
                for line in lines:
                    parts = line.strip().split(' - ', 4)
                    if len(parts) == 5:
                        stats['total'] += 1
                        level = parts[2]
                        if level in stats['levels']:
                            stats['levels'][level] += 1
        except Exception as e:
            logger.error(f"❌ 读取日志文件失败: {filename}, 错误: {str(e)}")
    
    return stats

@app.get("/api/logs/errors")
async def get_error_logs(developer=Depends(get_current_developer), limit: int = 50):
    """
    获取错误日志（仅开发者可访问）
    """
    logger.info(f"❌ 开发者 {developer.username} 获取错误日志")
    return await get_logs(developer=developer, level="ERROR", limit=limit)

@app.get("/api/logs/warnings")
async def get_warning_logs(developer=Depends(get_current_developer), limit: int = 50):
    """
    获取警告日志（仅开发者可访问）
    """
    logger.info(f"⚠️ 开发者 {developer.username} 获取警告日志")
    return await get_logs(developer=developer, level="WARNING", limit=limit)

@app.get("/api/favorites")
def get_favorites(developer=Depends(get_current_developer)):
    """获取收藏列表（仅开发者可访问）"""
    logger.info(f"⭐ 开发者 {developer.username} 获取收藏列表")
    
    favorites_file = Path.home() / ".ardc" / "favorites.json"
    if favorites_file.exists():
        try:
            with open(favorites_file, 'r', encoding='utf-8') as f:
                favorites = json.load(f)
                return {"favorites": favorites.get(developer.username, [])}
        except Exception as e:
            logger.error(f"❌ 读取收藏文件失败: {str(e)}")
    
    return {"favorites": []}

@app.post("/api/favorites/{skill_id}")
def add_favorite(skill_id: str, developer=Depends(get_current_developer)):
    """添加收藏（仅开发者可访问）"""
    logger.info(f"⭐ 开发者 {developer.username} 添加收藏: {skill_id}")
    
    favorites_file = Path.home() / ".ardc" / "favorites.json"
    favorites = {}
    
    if favorites_file.exists():
        try:
            with open(favorites_file, 'r', encoding='utf-8') as f:
                favorites = json.load(f)
        except Exception as e:
            logger.error(f"❌ 读取收藏文件失败: {str(e)}")
    
    if developer.username not in favorites:
        favorites[developer.username] = []
    
    if skill_id not in favorites[developer.username]:
        favorites[developer.username].append(skill_id)
    
    try:
        with open(favorites_file, 'w', encoding='utf-8') as f:
            json.dump(favorites, f, ensure_ascii=False, indent=2)
        return {"message": "收藏成功"}
    except Exception as e:
        logger.error(f"❌ 保存收藏失败: {str(e)}")
        raise HTTPException(status_code=500, detail="保存失败")

@app.delete("/api/favorites/{skill_id}")
def remove_favorite(skill_id: str, developer=Depends(get_current_developer)):
    """移除收藏（仅开发者可访问）"""
    logger.info(f"⭐ 开发者 {developer.username} 移除收藏: {skill_id}")
    
    favorites_file = Path.home() / ".ardc" / "favorites.json"
    
    if favorites_file.exists():
        try:
            with open(favorites_file, 'r', encoding='utf-8') as f:
                favorites = json.load(f)
            
            if developer.username in favorites and skill_id in favorites[developer.username]:
                favorites[developer.username].remove(skill_id)
                
                with open(favorites_file, 'w', encoding='utf-8') as f:
                    json.dump(favorites, f, ensure_ascii=False, indent=2)
            
            return {"message": "取消收藏成功"}
        except Exception as e:
            logger.error(f"❌ 操作收藏失败: {str(e)}")
            raise HTTPException(status_code=500, detail="操作失败")
    
    return {"message": "取消收藏成功"}

def start_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
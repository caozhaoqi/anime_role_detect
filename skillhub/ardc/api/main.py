#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 主入口
提供技能仓库 RESTful API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

from ardc.store.registry import SkillRegistry
from ardc.store.index import SkillIndex
from ardc.version.manager import VersionManager
from ardc.api.auth import router as auth_router

app = FastAPI(title="ARD Skill Repository API", version="1.0.0")

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
    from ardc.store.metadata import SkillMetadata
    
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
    
    return {"message": "技能注册成功", "skill_id": skill.id}

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
    if registry.install_skill(skill_id, version):
        return {"message": "技能安装成功"}
    raise HTTPException(status_code=400, detail="安装失败")

@app.delete("/api/skills/{skill_id}/uninstall")
def uninstall_skill(skill_id: str):
    if registry.uninstall_skill(skill_id):
        return {"message": "技能卸载成功"}
    raise HTTPException(status_code=400, detail="卸载失败")

@app.get("/api/search")
def search_skills(keyword: str, category: Optional[str] = None, limit: int = 20):
    results = index.search(keyword, category, limit=limit)
    return {"total": len(results), "skills": [s.dict() for s in results]}

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

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ARD Skill Hub API",
        "version": "1.0.0"
    }

def start_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
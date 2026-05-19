#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARDC SkillHub 技能仓库服务
提供技能的存储、查询、下载、版本管理等功能
"""
import os
import json
import zipfile
import hashlib
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(title="ARDC SkillHub API", version="1.0.0")

# 配置
SKILL_STORAGE = Path.home() / ".ardc" / "server_skills"
SKILL_STORAGE.mkdir(parents=True, exist_ok=True)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 安全
security = HTTPBearer()

# 数据模型
class Skill(BaseModel):
    name: str
    version: str
    description: str
    author: str
    category: str
    tags: List[str] = []
    dependencies: List[str] = []
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    downloads: int = 0

class SkillVersion(BaseModel):
    version: str
    release_date: str
    changelog: str
    download_url: str

class InstallRequest(BaseModel):
    skill_name: str
    version: Optional[str] = None

# 模拟数据库
skills_db = {
    "ardc-collector": {
        "name": "ardc-collector",
        "version": "1.0.0",
        "description": "数据采集技能，支持从多个来源批量采集动漫角色图片",
        "author": "ARDC Team",
        "category": "数据采集",
        "tags": ["数据采集", "图片下载", "爬虫"],
        "dependencies": ["requests", "beautifulsoup4"],
        "created_at": "2024-01-01",
        "updated_at": "2024-01-15",
        "downloads": 1234
    },
    "ardc-cleaner": {
        "name": "ardc-cleaner",
        "version": "1.0.0",
        "description": "数据清洗技能，提供图片去重、质量检查、格式转换等功能",
        "author": "ARDC Team",
        "category": "数据处理",
        "tags": ["数据清洗", "去重", "质量检查"],
        "dependencies": ["pillow", "imagehash"],
        "created_at": "2024-01-05",
        "updated_at": "2024-01-20",
        "downloads": 856
    },
    "ardc-trainer": {
        "name": "ardc-trainer",
        "version": "1.0.0",
        "description": "模型训练技能，支持多种深度学习框架训练角色分类模型",
        "author": "ARDC Team",
        "category": "模型训练",
        "tags": ["模型训练", "深度学习", "分类"],
        "dependencies": ["pytorch", "torchvision", "scikit-learn"],
        "created_at": "2024-01-10",
        "updated_at": "2024-01-25",
        "downloads": 623
    },
    "ardc-classifier": {
        "name": "ardc-classifier",
        "version": "1.0.0",
        "description": "角色分类技能，使用预训练模型进行角色识别",
        "author": "ARDC Team",
        "category": "推理预测",
        "tags": ["分类", "推理", "角色识别"],
        "dependencies": ["pytorch", "torchvision"],
        "created_at": "2024-01-12",
        "updated_at": "2024-01-28",
        "downloads": 1056
    },
    "ardc-search": {
        "name": "ardc-search",
        "version": "1.0.0",
        "description": "以图搜图技能，支持基于图片特征的相似图片搜索",
        "author": "ARDC Team",
        "category": "检索搜索",
        "tags": ["图像检索", "以图搜图", "FAISS"],
        "dependencies": ["faiss-cpu", "pillow"],
        "created_at": "2024-01-15",
        "updated_at": "2024-02-01",
        "downloads": 445
    }
}

versions_db = {
    "ardc-collector": [
        {"version": "1.0.0", "release_date": "2024-01-15", "changelog": "初始版本"},
        {"version": "0.9.0", "release_date": "2024-01-01", "changelog": "测试版本"}
    ]
}

# 用户数据库（模拟）
# admin 的密码是 admin123
users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "password": "sha256$5f4dcc3b5aa765d61d8327deb882cf99$b118b3c8794f0bfc9a0cd38910067d77ce01f816b2e054803f1849cfb16fe460",  # admin123
        "created_at": "2024-01-01",
        "is_active": True
    }
}

# 生成 Token
def generate_token(username: str) -> str:
    """生成用户认证 Token"""
    token_data = f"{username}:{datetime.now().timestamp()}"
    return hashlib.sha256(token_data.encode()).hexdigest()

# 哈希密码
def hash_password(password: str) -> str:
    """哈希密码"""
    salt = hashlib.md5(os.urandom(16)).hexdigest()
    hashed = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
    return f"sha256${salt}${hashed}"

# 验证密码
def verify_password(password: str, hashed_password: str) -> bool:
    """验证密码"""
    parts = hashed_password.split("$")
    if len(parts) != 3 or parts[0] != "sha256":
        return False
    salt = parts[1]
    expected = parts[2]
    actual = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
    return actual == expected

# 登录请求模型
class LoginRequest(BaseModel):
    username: str
    password: str

# 注册请求模型
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

# ============================================================
# 认证端点
# ============================================================
@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """用户名密码登录"""
    # 检查用户是否存在
    if request.username not in users_db:
        return {"success": False, "message": "用户名或密码错误"}
    
    user = users_db[request.username]
    
    # 检查用户状态
    if not user.get("is_active", True):
        return {"success": False, "message": "用户已被禁用"}
    
    # 验证密码
    if not verify_password(request.password, user["password"]):
        return {"success": False, "message": "用户名或密码错误"}
    
    # 生成 Token
    token = generate_token(request.username)
    
    return {
        "success": True,
        "message": "登录成功",
        "token": token,
        "username": request.username,
        "email": user["email"]
    }

@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    """用户注册"""
    # 检查用户名是否已存在
    if request.username in users_db:
        return {"success": False, "message": "用户名已存在"}
    
    # 检查邮箱格式（简单验证）
    if "@" not in request.email:
        return {"success": False, "message": "无效的邮箱地址"}
    
    # 检查密码长度
    if len(request.password) < 6:
        return {"success": False, "message": "密码长度至少6位"}
    
    # 创建用户
    users_db[request.username] = {
        "username": request.username,
        "email": request.email,
        "password": hash_password(request.password),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_active": True
    }
    
    return {
        "success": True,
        "message": "注册成功"
    }

@app.post("/api/auth/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """登出（演示用，实际应在服务端维护 token 黑名单）"""
    return {"success": True, "message": "登出成功"}

@app.get("/api/auth/me")
async def get_user_info(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前用户信息"""
    # 简化实现：从 token 中提取用户名（实际应验证 token）
    token = credentials.credentials
    
    # 在实际应用中，应该验证 token 并获取用户信息
    # 这里简化处理，返回第一个用户
    if users_db:
        username = next(iter(users_db.keys()))
        user = users_db[username]
        return {
            "success": True,
            "data": {
                "username": user["username"],
                "email": user["email"],
                "created_at": user["created_at"]
            }
        }
    
    return {"success": False, "message": "未找到用户"}

# ============================================================
# 技能管理端点
# ============================================================
@app.get("/api/skills")
async def list_skills(
    category: Optional[str] = None,
    keyword: Optional[str] = None
):
    """列出所有技能"""
    result = list(skills_db.values())
    
    if category:
        result = [s for s in result if s["category"] == category]
    
    if keyword:
        keyword = keyword.lower()
        result = [s for s in result if 
                  keyword in s["name"].lower() or 
                  keyword in s["description"].lower()]
    
    return result

@app.get("/api/skills/{skill_name}")
async def get_skill(skill_name: str):
    """获取技能详情"""
    skill = skills_db.get(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail="技能不存在")
    return skill

@app.post("/api/skills")
async def create_skill(
    skill: Skill,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """创建新技能"""
    if skill.name in skills_db:
        raise HTTPException(status_code=400, detail="技能已存在")
    
    skill.created_at = datetime.now().isoformat()
    skill.updated_at = skill.created_at
    skills_db[skill.name] = skill.dict()
    
    # 创建技能目录
    skill_dir = SKILL_STORAGE / skill.name
    skill_dir.mkdir(parents=True, exist_ok=True)
    
    return {"success": True, "message": "技能创建成功"}

@app.put("/api/skills/{skill_name}")
async def update_skill(
    skill_name: str,
    skill: Skill,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """更新技能信息"""
    if skill_name not in skills_db:
        raise HTTPException(status_code=404, detail="技能不存在")
    
    skill.updated_at = datetime.now().isoformat()
    skills_db[skill_name] = skill.dict()
    
    return {"success": True, "message": "技能更新成功"}

@app.delete("/api/skills/{skill_name}")
async def delete_skill(
    skill_name: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """删除技能"""
    if skill_name not in skills_db:
        raise HTTPException(status_code=404, detail="技能不存在")
    
    del skills_db[skill_name]
    
    # 删除技能目录
    skill_dir = SKILL_STORAGE / skill_name
    if skill_dir.exists():
        import shutil
        shutil.rmtree(skill_dir)
    
    return {"success": True, "message": "技能删除成功"}

# ============================================================
# 版本管理端点
# ============================================================
@app.get("/api/skills/{skill_name}/versions")
async def list_versions(skill_name: str):
    """列出技能的所有版本"""
    if skill_name not in skills_db:
        raise HTTPException(status_code=404, detail="技能不存在")
    
    versions = versions_db.get(skill_name, [])
    return versions

@app.get("/api/skills/{skill_name}/versions/{version}")
async def get_version(skill_name: str, version: str):
    """获取指定版本详情"""
    if skill_name not in skills_db:
        raise HTTPException(status_code=404, detail="技能不存在")
    
    versions = versions_db.get(skill_name, [])
    version_info = next((v for v in versions if v["version"] == version), None)
    
    if not version_info:
        raise HTTPException(status_code=404, detail="版本不存在")
    
    return version_info

# ============================================================
# 下载端点
# ============================================================
@app.get("/api/skills/{skill_name}/download")
async def download_skill(skill_name: str, version: Optional[str] = None):
    """下载技能包"""
    if skill_name not in skills_db:
        raise HTTPException(status_code=404, detail="技能不存在")
    
    # 模拟下载，实际应返回文件
    import io
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        # 添加 SKILL.md
        skill_info = skills_db[skill_name]
        skill_md = f"""# {skill_info['name']}

## 基本信息
- 名称: {skill_info['name']}
- 版本: {skill_info['version']}
- 作者: {skill_info['author']}
- 分类: {skill_info['category']}

## 描述
{skill_info['description']}

## 依赖
{chr(10).join(f"- {dep}" for dep in skill_info['dependencies'])}

## 标签
{chr(10).join(f"- {tag}" for tag in skill_info['tags'])}
"""
        zf.writestr("SKILL.md", skill_md)
        
        # 添加示例脚本
        zf.writestr("scripts/__init__.py", "")
        zf.writestr("scripts/main.py", """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
{skill_name} - {skill_info['description']}
\"\"\"
def main():
    print(f"欢迎使用 {skill_info['name']} v{skill_info['version']}")

if __name__ == '__main__':
    main()
""")
    
    zip_buffer.seek(0)
    
    # 更新下载次数
    skills_db[skill_name]["downloads"] += 1
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={skill_name}.zip"}
    )

# ============================================================
# 上传端点
# ============================================================
@app.post("/api/skills/{skill_name}/upload")
async def upload_skill(
    skill_name: str,
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """上传技能包"""
    if skill_name not in skills_db:
        raise HTTPException(status_code=404, detail="技能不存在")
    
    # 保存文件
    skill_dir = SKILL_STORAGE / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = skill_dir / f"{skill_name}.zip"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # 解压
    with zipfile.ZipFile(file_path, 'r') as zf:
        zf.extractall(skill_dir)
    
    # 更新版本信息
    if skill_name not in versions_db:
        versions_db[skill_name] = []
    
    current_version = skills_db[skill_name]["version"]
    versions_db[skill_name].insert(0, {
        "version": current_version,
        "release_date": datetime.now().isoformat(),
        "changelog": "上传新版本"
    })
    
    skills_db[skill_name]["updated_at"] = datetime.now().isoformat()
    
    return {"success": True, "message": "技能上传成功"}

# ============================================================
# 搜索端点
# ============================================================
@app.get("/api/search")
async def search_skills(
    q: str,
    category: Optional[str] = None,
    limit: int = 10
):
    """搜索技能"""
    results = []
    
    for skill in skills_db.values():
        match = False
        
        # 匹配名称和描述
        if q.lower() in skill["name"].lower():
            match = True
        elif q.lower() in skill["description"].lower():
            match = True
        elif any(q.lower() in tag.lower() for tag in skill["tags"]):
            match = True
        
        # 匹配分类
        if category and skill["category"] != category:
            match = False
        
        if match:
            results.append(skill)
    
    # 按下载量排序
    results.sort(key=lambda x: x["downloads"], reverse=True)
    
    return results[:limit]

# ============================================================
# 统计端点
# ============================================================
@app.get("/api/stats")
async def get_stats():
    """获取统计信息"""
    total_skills = len(skills_db)
    total_downloads = sum(skill["downloads"] for skill in skills_db.values())
    
    category_stats = {}
    for skill in skills_db.values():
        category = skill["category"]
        category_stats[category] = category_stats.get(category, 0) + 1
    
    return {
        "total_skills": total_skills,
        "total_downloads": total_downloads,
        "category_stats": category_stats,
        "top_downloads": sorted(
            skills_db.values(),
            key=lambda x: x["downloads"],
            reverse=True
        )[:5]
    }

# ============================================================
# 健康检查
# ============================================================
@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

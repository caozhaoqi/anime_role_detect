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
        "category": "collector",
        "status": "stable",
        "tags": ["数据采集", "图片下载", "爬虫"],
        "dependencies": ["requests", "beautifulsoup4"],
        "created_at": "2024-01-01",
        "updated_at": "2024-01-15",
        "downloads": 1234,
        "installed": False,
        "memory_mb": 256,
        "cpu_cores": 1,
        "runtime": "Python 3.8+"
    },
    "ardc-cleaner": {
        "name": "ardc-cleaner",
        "version": "1.0.0",
        "description": "数据清洗技能，提供图片去重、质量检查、格式转换等功能",
        "author": "ARDC Team",
        "category": "cleaner",
        "status": "stable",
        "tags": ["数据清洗", "去重", "质量检查"],
        "dependencies": ["pillow", "imagehash"],
        "created_at": "2024-01-05",
        "updated_at": "2024-01-20",
        "downloads": 856,
        "installed": False,
        "memory_mb": 512,
        "cpu_cores": 2,
        "runtime": "Python 3.8+"
    },
    "ardc-trainer": {
        "name": "ardc-trainer",
        "version": "1.0.0",
        "description": "模型训练技能，支持多种深度学习框架训练角色分类模型",
        "author": "ARDC Team",
        "category": "trainer",
        "status": "stable",
        "tags": ["模型训练", "深度学习", "分类"],
        "dependencies": ["pytorch", "torchvision", "scikit-learn"],
        "created_at": "2024-01-10",
        "updated_at": "2024-01-25",
        "downloads": 623,
        "installed": False,
        "memory_mb": 2048,
        "cpu_cores": 4,
        "runtime": "Python 3.8+"
    },
    "ardc-classifier": {
        "name": "ardc-classifier",
        "version": "1.0.0",
        "description": "角色分类技能，使用预训练模型进行角色识别",
        "author": "ARDC Team",
        "category": "classifier",
        "status": "stable",
        "tags": ["分类", "推理", "角色识别"],
        "dependencies": ["pytorch", "torchvision"],
        "created_at": "2024-01-12",
        "updated_at": "2024-01-28",
        "downloads": 1056,
        "installed": False,
        "memory_mb": 1024,
        "cpu_cores": 2,
        "runtime": "Python 3.8+"
    },
    "ardc-search": {
        "name": "ardc-search",
        "version": "1.0.0",
        "description": "以图搜图技能，支持基于图片特征的相似图片搜索",
        "author": "ARDC Team",
        "category": "search",
        "status": "stable",
        "tags": ["图像检索", "以图搜图", "FAISS"],
        "dependencies": ["faiss-cpu", "pillow"],
        "created_at": "2024-01-15",
        "updated_at": "2024-02-01",
        "downloads": 445,
        "installed": False,
        "memory_mb": 512,
        "cpu_cores": 1,
        "runtime": "Python 3.8+"
    }
}

versions_db = {
    "ardc-collector": [
        {"version": "1.0.0", "release_date": "2024-01-15", "changelog": "初始版本，支持多源数据采集"},
        {"version": "0.9.0", "release_date": "2024-01-01", "changelog": "测试版本"}
    ],
    "ardc-cleaner": [
        {"version": "1.0.0", "release_date": "2024-01-20", "changelog": "初始版本，支持图片去重和格式转换"},
        {"version": "0.9.5", "release_date": "2024-01-10", "changelog": "添加质量检查功能"},
        {"version": "0.9.0", "release_date": "2024-01-05", "changelog": "测试版本"}
    ],
    "ardc-trainer": [
        {"version": "1.0.0", "release_date": "2024-01-25", "changelog": "初始版本，支持PyTorch训练"},
        {"version": "0.9.0", "release_date": "2024-01-10", "changelog": "测试版本"}
    ],
    "ardc-classifier": [
        {"version": "1.0.0", "release_date": "2024-01-28", "changelog": "初始版本，支持角色分类"},
        {"version": "0.9.0", "release_date": "2024-01-12", "changelog": "测试版本"}
    ],
    "ardc-search": [
        {"version": "1.0.0", "release_date": "2024-02-01", "changelog": "初始版本，支持以图搜图"},
        {"version": "0.9.0", "release_date": "2024-01-15", "changelog": "测试版本"}
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
    
    return {"skills": result}

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

@app.post("/api/skills/{skill_name}/versions")
async def create_version(
    skill_name: str,
    version: str,
    changelog: Optional[str] = "",
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """发布新版本（开发者接口）"""
    if skill_name not in skills_db:
        raise HTTPException(status_code=404, detail="技能不存在")
    
    versions = versions_db.get(skill_name, [])
    
    # 检查版本是否已存在
    if any(v["version"] == version for v in versions):
        raise HTTPException(status_code=400, detail="版本已存在")
    
    # 添加新版本
    new_version = {
        "version": version,
        "release_date": datetime.now().strftime("%Y-%m-%d"),
        "changelog": changelog or "无更新说明"
    }
    
    versions.insert(0, new_version)
    versions_db[skill_name] = versions
    
    # 更新技能的当前版本
    skills_db[skill_name]["version"] = version
    skills_db[skill_name]["updated_at"] = datetime.now().strftime("%Y-%m-%d")
    
    return {"success": True, "message": f"版本 {version} 发布成功"}

@app.get("/api/skills/{skill_name}/check-update")
async def check_update(skill_name: str, current_version: Optional[str] = None):
    """检查是否有新版本"""
    if skill_name not in skills_db:
        raise HTTPException(status_code=404, detail="技能不存在")
    
    versions = versions_db.get(skill_name, [])
    if not versions:
        return {"has_update": False, "latest_version": skills_db[skill_name]["version"]}
    
    latest_version = versions[0]["version"]
    has_update = False
    
    if current_version and current_version != latest_version:
        has_update = True
    
    return {
        "has_update": has_update,
        "latest_version": latest_version,
        "current_version": current_version,
        "changelog": versions[0]["changelog"]
    }

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
        zf.writestr("scripts/main.py", f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
{skill_name} - {skill_info['description']}
\"\"\"
def main():
    print(f"欢迎使用 {skill_info['name']} v{skill_info['version']}")

if __name__ == '__main__':
    main()
""")
        
        # 添加配置文件
        zf.writestr("config.json", f"""{{
    "name": "{skill_info['name']}",
    "version": "{skill_info['version']}",
    "author": "{skill_info['author']}",
    "description": "{skill_info['description']}",
    "dependencies": {skill_info['dependencies']},
    "memory_mb": {skill_info['memory_mb']},
    "cpu_cores": {skill_info['cpu_cores']},
    "runtime": "{skill_info['runtime']}"
}}""")
        
        # 添加入口文件
        zf.writestr("__init__.py", f"""\"\"\"
{skill_name} - {skill_info['description']}
Version: {skill_info['version']}
Author: {skill_info['author']}
\"\"\"
from .scripts.main import main

__version__ = "{skill_info['version']}"
__author__ = "{skill_info['author']}"
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
# 安装端点
# ============================================================
@app.post("/api/skills/{skill_name}/install")
async def install_skill(skill_name: str):
    """安装技能"""
    if skill_name not in skills_db:
        raise HTTPException(status_code=404, detail="技能不存在")
    
    # 标记为已安装
    skills_db[skill_name]["installed"] = True
    
    return {
        "success": True,
        "message": f"技能 {skill_name} 安装成功",
        "skill": skills_db[skill_name]
    }

@app.delete("/api/skills/{skill_name}/uninstall")
async def uninstall_skill(skill_name: str):
    """卸载技能"""
    if skill_name not in skills_db:
        raise HTTPException(status_code=404, detail="技能不存在")
    
    # 标记为未安装
    skills_db[skill_name]["installed"] = False
    
    return {
        "success": True,
        "message": f"技能 {skill_name} 卸载成功",
        "skill": skills_db[skill_name]
    }

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
    
    return {"skills": results[:limit]}

# ============================================================
# 分类端点
# ============================================================
@app.get("/api/categories")
async def get_categories():
    """获取所有分类"""
    categories = {}
    for skill in skills_db.values():
        category = skill["category"]
        categories[category] = categories.get(category, 0) + 1
    return categories

@app.get("/api/tags")
async def get_tags():
    """获取所有标签"""
    tags = set()
    for skill in skills_db.values():
        tags.update(skill["tags"])
    return list(tags)

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
# 安装脚本端点
# ============================================================
@app.get("/api/install/install.sh")
async def get_install_script_sh():
    """获取 macOS/Linux 安装脚本"""
    script = """#!/bin/bash
set -e

echo "========================================"
echo "  ARDC SkillHub CLI 安装脚本"
echo "========================================"
echo ""

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

# 检查 Python 版本
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "错误: Python 版本需要 3.8+，当前版本: $PYTHON_VERSION"
    exit 1
fi

echo "Python 版本检查通过: $PYTHON_VERSION"
echo ""

# 创建安装目录
INSTALL_DIR="$HOME/.ardc"
BIN_DIR="$INSTALL_DIR/bin"
mkdir -p "$BIN_DIR"

echo "创建安装目录: $INSTALL_DIR"
echo ""

# 下载技能同步工具
echo "下载 ARDC SkillHub CLI..."
curl -s -L "http://47.79.91.89:8888/api/scripts/ardc-skill-sync.py" -o "$BIN_DIR/ardc-skill-sync"

if [ $? -ne 0 ]; then
    echo "错误: 下载失败"
    exit 1
fi

chmod +x "$BIN_DIR/ardc-skill-sync"

# 创建配置文件
echo "创建配置文件..."
cat > "$INSTALL_DIR/config.json" << EOF
{
    "skill_hub_url": "http://47.79.91.89:8888",
    "skills_dir": "$INSTALL_DIR/skills",
    "version": "1.0.0"
}
EOF

# 添加到 PATH
if [ -f "$HOME/.bashrc" ]; then
    echo "export PATH=\$PATH:$BIN_DIR" >> "$HOME/.bashrc"
fi

if [ -f "$HOME/.zshrc" ]; then
    echo "export PATH=\$PATH:$BIN_DIR" >> "$HOME/.zshrc"
fi

echo ""
echo "========================================"
echo "  安装完成!"
echo "========================================"
echo ""
echo "请重启终端或执行以下命令使配置生效:"
echo "  source ~/.bashrc  # 或 source ~/.zshrc"
echo ""
echo "使用方法:"
echo "  ardc-skill-sync login     # 登录认证"
echo "  ardc-skill-sync list      # 查看技能列表"
echo "  ardc-skill-sync install <skill-name>  # 安装技能"
echo "  ardc-skill-sync help      # 查看帮助"
echo ""
"""
    return Response(script, media_type="text/plain")

@app.get("/api/install/install.ps1")
async def get_install_script_ps1():
    """获取 Windows PowerShell 安装脚本"""
    script = """<#
.SYNOPSIS
    ARDC SkillHub CLI 安装脚本
#>

Write-Host "========================================"
Write-Host "  ARDC SkillHub CLI 安装脚本"
Write-Host "========================================"
Write-Host ""

# 检查 Python 是否安装
if (-not (Get-Command python3 -ErrorAction SilentlyContinue)) {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Error "未找到 Python，请先安装 Python 3.8+"
        exit 1
    }
    $pythonCmd = "python"
} else {
    $pythonCmd = "python3"
}

# 检查 Python 版本
$pythonVersion = & $pythonCmd --version 2>&1 | Select-Object -First 1
$versionMatch = [regex]::Match($pythonVersion, '(\d+\.\d+)')
if (-not $versionMatch.Success) {
    Write-Error "无法检测 Python 版本"
    exit 1
}

$currentVersion = [version]$versionMatch.Groups[1].Value
$requiredVersion = [version]"3.8"

if ($currentVersion -lt $requiredVersion) {
    Write-Error "Python 版本需要 3.8+，当前版本: $currentVersion"
    exit 1
}

Write-Host "Python 版本检查通过: $currentVersion"
Write-Host ""

# 创建安装目录
$installDir = "$env:USERPROFILE\.ardc"
$binDir = "$installDir\bin"
New-Item -ItemType Directory -Path $binDir -Force | Out-Null

Write-Host "创建安装目录: $installDir"
Write-Host ""

# 下载技能同步工具
Write-Host "下载 ARDC SkillHub CLI..."
$url = "http://47.79.91.89:8888/api/scripts/ardc-skill-sync.py"
$outputPath = "$binDir\ardc-skill-sync.py"

try {
    Invoke-WebRequest -Uri $url -OutFile $outputPath -UseBasicParsing
} catch {
    Write-Error "下载失败: $_"
    exit 1
}

# 创建批处理包装器
@"
@echo off
python "%binDir%\ardc-skill-sync.py" %*
"@ | Out-File -FilePath "$binDir\ardc-skill-sync.bat" -Encoding utf8

# 创建配置文件
$configContent = @"
{
    "skill_hub_url": "http://47.79.91.89:8888",
    "skills_dir": "$installDir\\skills",
    "version": "1.0.0"
}
"@
$configContent | Out-File -FilePath "$installDir\config.json" -Encoding utf8

# 添加到 PATH
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if (-not $currentPath.Contains($binDir)) {
    [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$binDir", "User")
    Write-Host "已将 $binDir 添加到用户 PATH"
}

Write-Host ""
Write-Host "========================================"
Write-Host "  安装完成!"
Write-Host "========================================"
Write-Host ""
Write-Host "请重启终端使配置生效"
Write-Host ""
Write-Host "使用方法:"
Write-Host "  ardc-skill-sync login     # 登录认证"
Write-Host "  ardc-skill-sync list      # 查看技能列表"
Write-Host "  ardc-skill-sync install <skill-name>  # 安装技能"
Write-Host "  ardc-skill-sync help      # 查看帮助"
Write-Host ""
"""
    return Response(script, media_type="text/plain")

@app.get("/api/scripts/ardc-skill-sync.py")
async def get_skill_sync_script():
    """获取技能同步工具脚本"""
    script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARD SkillHub CLI - 技能同步工具
"""

import os
import sys
import json
import hashlib
import datetime
import argparse
import subprocess
from pathlib import Path

# 配置文件路径
CONFIG_FILE = Path.home() / ".ardc" / "config.json"
TOKEN_FILE = Path.home() / ".ardc" / "token.txt"

def print_success(msg):
    print(f"\\033[32m✓ {msg}\\033[0m")

def print_error(msg):
    print(f"\\033[31m✗ {msg}\\033[0m")

def print_info(msg):
    print(f"\\033[34mℹ {msg}\\033[0m")

def load_config():
    """加载配置文件"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_config(config):
    """保存配置文件"""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

def get_skill_hub_url():
    """获取技能仓库地址"""
    config = load_config()
    return config.get("skill_hub_url", "http://47.79.91.89:8888")

def get_token():
    """获取 Token"""
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE, "r") as f:
            return f.read().strip()
    return None

def set_token(token):
    """设置 Token"""
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_FILE, "w") as f:
        f.write(token)

def login():
    """用户名密码登录认证"""
    print("=" * 60)
    print("          ARDC SkillHub 登录认证")
    print("=" * 60)
    print()
    
    skill_hub_url = get_skill_hub_url()
    
    try:
        import requests
    except ImportError:
        print_error("请先安装 requests 库: pip install requests")
        return
    
    # 获取用户名和密码
    import getpass
    username = input("请输入用户名: ").strip()
    password = getpass.getpass("请输入密码: ").strip()
    
    if not username or not password:
        print_error("用户名和密码不能为空")
        return
    
    try:
        print("\\n正在登录...")
        
        response = requests.post(
            f"{skill_hub_url}/api/auth/login",
            json={
                "username": username,
                "password": password
            },
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            token = data.get("token")
            set_token(token)
            print_success("登录成功！")
            print(f"Token 已保存到: {TOKEN_FILE}")
            
            # 保存用户信息
            config = load_config()
            config["username"] = username
            save_config(config)
            
            print_info(f"欢迎回来, {username}!")
        else:
            print_error(f"登录失败: {data.get('message', '未知错误')}")
            
    except requests.exceptions.RequestException as e:
        print_error(f"登录失败: {e}")
        print_info("正在使用离线模式...")
        # 生成临时 token（演示用）
        temp_token = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()
        set_token(temp_token)
        print_success("已进入离线模式")

def register():
    """用户注册"""
    print("=" * 60)
    print("          ARDC SkillHub 用户注册")
    print("=" * 60)
    print()
    
    skill_hub_url = get_skill_hub_url()
    
    try:
        import requests
    except ImportError:
        print_error("请先安装 requests 库: pip install requests")
        return
    
    # 获取注册信息
    import getpass
    username = input("请输入用户名: ").strip()
    email = input("请输入邮箱: ").strip()
    password = getpass.getpass("请输入密码: ").strip()
    confirm_password = getpass.getpass("请确认密码: ").strip()
    
    # 验证输入
    if not username:
        print_error("用户名不能为空")
        return
    if not email:
        print_error("邮箱不能为空")
        return
    if not password:
        print_error("密码不能为空")
        return
    if password != confirm_password:
        print_error("两次输入的密码不一致")
        return
    
    try:
        print("\\n正在注册...")
        
        response = requests.post(
            f"{skill_hub_url}/api/auth/register",
            json={
                "username": username,
                "email": email,
                "password": password
            },
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            print_success("注册成功！")
            print_info("请使用用户名密码登录")
        else:
            print_error(f"注册失败: {data.get('message', '未知错误')}")
            
    except requests.exceptions.RequestException as e:
        print_error(f"注册失败: {e}")

def status():
    """显示本地配置"""
    print("=" * 60)
    print("          ARDC SkillHub 本地状态")
    print("=" * 60)
    print()
    
    config = load_config()
    token = get_token()
    
    print(f"技能仓库地址: {config.get('skill_hub_url', '未配置')}")
    print(f"技能存储目录: {config.get('skills_dir', '未配置')}")
    print(f"已登录: {'是' if token else '否'}")
    print(f"用户名: {config.get('username', '未登录')}")
    print()
    
    # 检查技能目录
    skills_dir = Path(config.get("skills_dir", Path.home() / ".ardc" / "skills"))
    if skills_dir.exists():
        skills = list(skills_dir.glob("*"))
        print(f"已安装技能 ({len(skills)} 个):")
        for skill in skills:
            if skill.is_dir():
                print(f"  - {skill.name}")
    else:
        print("已安装技能: 0 个")

def list_skills():
    """列出所有技能"""
    print("=" * 60)
    print("          ARDC SkillHub 技能列表")
    print("=" * 60)
    print()
    
    skill_hub_url = get_skill_hub_url()
    
    try:
        import requests
    except ImportError:
        print_error("请先安装 requests 库: pip install requests")
        return
    
    try:
        response = requests.get(f"{skill_hub_url}/api/skills", timeout=30)
        response.raise_for_status()
        data = response.json()
        
        skills = data.get("skills", [])
        
        if not skills:
            print_info("暂无可用技能")
            return
        
        print(f"共 {len(skills)} 个技能:")
        print("-" * 60)
        
        for skill in skills:
            status_icon = "✓" if skill.get("installed") else " "
            print(f"{status_icon} {skill['name']}")
            print(f"     版本: {skill['version']}")
            print(f"     分类: {skill['category']}")
            print(f"     描述: {skill['description']}")
            print(f"     下载量: {skill['downloads']}")
            print()
            
    except requests.exceptions.RequestException as e:
        print_error(f"获取技能列表失败: {e}")

def install_skill(skill_name):
    """安装技能"""
    print("=" * 60)
    print(f"          安装技能: {skill_name}")
    print("=" * 60)
    print()
    
    skill_hub_url = get_skill_hub_url()
    
    try:
        import requests
    except ImportError:
        print_error("请先安装 requests 库: pip install requests")
        return
    
    try:
        print(f"正在下载 {skill_name}...")
        
        response = requests.get(
            f"{skill_hub_url}/api/skills/{skill_name}/download",
            timeout=60
        )
        response.raise_for_status()
        
        # 保存技能包
        config = load_config()
        skills_dir = Path(config.get("skills_dir", Path.home() / ".ardc" / "skills"))
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        skill_path = skills_dir / f"{skill_name}.zip"
        with open(skill_path, "wb") as f:
            f.write(response.content)
        
        # 解压
        print("正在解压技能包...")
        import zipfile
        with zipfile.ZipFile(skill_path, "r") as zf:
            zf.extractall(skills_dir / skill_name)
        
        # 删除压缩包
        skill_path.unlink()
        
        print_success(f"技能 {skill_name} 安装成功！")
        print(f"技能已安装到: {skills_dir / skill_name}")
        
    except requests.exceptions.RequestException as e:
        print_error(f"安装失败: {e}")

def uninstall_skill(skill_name):
    """卸载技能"""
    print("=" * 60)
    print(f"          卸载技能: {skill_name}")
    print("=" * 60)
    print()
    
    config = load_config()
    skills_dir = Path(config.get("skills_dir", Path.home() / ".ardc" / "skills"))
    skill_path = skills_dir / skill_name
    
    if not skill_path.exists():
        print_error(f"技能 {skill_name} 未安装")
        return
    
    import shutil
    try:
        shutil.rmtree(skill_path)
        print_success(f"技能 {skill_name} 卸载成功！")
    except Exception as e:
        print_error(f"卸载失败: {e}")

def check_updates():
    """检查更新"""
    print("=" * 60)
    print("          检查技能更新")
    print("=" * 60)
    print()
    
    print_info("正在检查已安装技能的更新...")
    print_info("此功能正在开发中")

def sync_skills():
    """同步更新技能"""
    print("=" * 60)
    print("          同步更新技能")
    print("=" * 60)
    print()
    
    print_info("正在同步更新所有技能...")
    print_info("此功能正在开发中")

def show_help():
    """显示帮助信息"""
    help_text = """
ARD SkillHub CLI - 技能同步工具

用法: ardc-skill-sync <command> [options]

命令:
  login               用户名密码登录认证
  register            注册新账户
  status              显示本地配置与检测到的技能目录
  check               检查已安装技能的更新情况
  sync                同步更新所有技能
  list                查询 SkillHub 上所有已发布的技能
  install <skill>     安装指定技能
  uninstall <skill>   卸载指定技能
  help                显示此帮助信息

示例:
  ardc-skill-sync login
  ardc-skill-sync list
  ardc-skill-sync install ardc-collector
  ardc-skill-sync uninstall ardc-collector

配置文件: ~/.ardc/config.json
Token 文件: ~/.ardc/token.txt
技能目录: ~/.ardc/skills
"""
    print(help_text)

def main():
    parser = argparse.ArgumentParser(
        prog="ardc-skill-sync",
        description="ARD SkillHub CLI - 技能同步工具"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["login", "register", "status", "check", "sync", "list", "install", "uninstall", "help"],
        default="help",
        help="要执行的命令"
    )
    parser.add_argument(
        "skill",
        nargs="?",
        help="技能名称（用于 install/uninstall 命令）"
    )
    
    args = parser.parse_args()
    
    commands = {
        "login": login,
        "register": register,
        "status": status,
        "check": check_updates,
        "sync": sync_skills,
        "list": list_skills,
        "install": lambda: install_skill(args.skill) if args.skill else (print_error("请指定技能名称"), show_help()),
        "uninstall": lambda: uninstall_skill(args.skill) if args.skill else (print_error("请指定技能名称"), show_help()),
        "help": show_help
    }
    
    command_func = commands.get(args.command, show_help)
    command_func()

if __name__ == "__main__":
    main()
'''
    return Response(script, media_type="text/plain")

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

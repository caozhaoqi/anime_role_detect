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

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# 导入数据库模块（支持 uvicorn 运行）
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import func
from database import init_db, get_db, SessionLocal
from database import User, Skill, SkillVersion, Review, Favorite, Screenshot, InstallHistory, Notification

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

# 认证
security = HTTPBearer(auto_error=False)

# ==================== 初始化数据库 ====================
init_db()
db = SessionLocal()
try:
    from database import insert_initial_data
    insert_initial_data(db)
finally:
    db.close()

# ==================== 请求模型 ====================
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = ""

class SkillRequest(BaseModel):
    id: str
    name: str
    description: str
    category: str
    tags: List[str] = []
    dependencies: List[str] = []
    config_schema: List[Dict] = []
    version: str = "1.0.0"
    author: str = "ARD Team"
    status: str = "stable"
    entry_point: str = ""
    runtime: str = "python"

# ==================== 工具函数 ====================
def hash_password(password: str) -> str:
    """哈希密码"""
    salt = hashlib.md5(os.urandom(16)).hexdigest()
    hashed = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
    return f"sha256${salt}${hashed}"

def verify_password(password: str, hashed_password: str) -> bool:
    """验证密码"""
    if not hashed_password.startswith("sha256$"):
        return False
    parts = hashed_password.split("$")
    if len(parts) != 3:
        return False
    salt = parts[1]
    expected_hash = parts[2]
    actual_hash = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
    return actual_hash == expected_hash

def generate_token(username: str) -> str:
    """生成用户认证 Token"""
    token_data = f"{username}:{datetime.now().timestamp()}"
    return hashlib.sha256(token_data.encode()).hexdigest()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前用户"""
    if not credentials:
        raise HTTPException(status_code=401, detail="未授权")
    
    token = credentials.credentials
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.token == token).first()
        if not user:
            raise HTTPException(status_code=401, detail="无效的 Token")
        return user
    finally:
        db.close()

def skill_to_dict(skill: Skill) -> dict:
    """将技能对象转换为字典"""
    return {
        "id": skill.skill_id,
        "name": skill.name,
        "description": skill.description,
        "category": skill.category,
        "status": skill.status,
        "version": skill.version,
        "author": skill.author,
        "downloads": skill.downloads,
        "rating": skill.rating,
        "review_count": skill.review_count,
        "installed": skill.installed,
        "has_update": skill.has_update,
        "changelog": skill.changelog,
        "dependencies": json.loads(skill.dependencies) if skill.dependencies else [],
        "tags": json.loads(skill.tags) if skill.tags else [],
        "config_schema": json.loads(skill.config_schema) if skill.config_schema else [],
        "entry_point": skill.entry_point,
        "runtime": skill.runtime,
        "created_at": skill.created_at.strftime("%Y-%m-%d %H:%M:%S") if skill.created_at else None,
        "updated_at": skill.updated_at.strftime("%Y-%m-%d %H:%M:%S") if skill.updated_at else None
    }

# ==================== 认证端点 ====================
@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """用户名密码登录"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == request.username).first()
        
        if not user:
            return {"success": False, "message": "用户名或密码错误"}
        
        if not user.is_active:
            return {"success": False, "message": "用户已被禁用"}
        
        if not verify_password(request.password, user.password):
            return {"success": False, "message": "用户名或密码错误"}
        
        token = generate_token(request.username)
        user.token = token
        db.commit()
        
        return {
            "success": True,
            "message": "登录成功",
            "token": token,
            "username": request.username,
            "email": user.email,
            "role": user.role
        }
    finally:
        db.close()

@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    """用户注册"""
    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == request.username).first():
            return {"success": False, "message": "用户名已存在"}
        
        hashed_password = hash_password(request.password)
        user = User(
            username=request.username,
            password=hashed_password,
            email=request.email,
            role="user"
        )
        db.add(user)
        db.commit()
        
        return {"success": True, "message": "注册成功"}
    finally:
        db.close()

@app.post("/api/auth/logout")
async def logout(user: User = Depends(get_current_user)):
    """用户登出"""
    db = SessionLocal()
    try:
        user_db = db.query(User).filter(User.id == user.id).first()
        if user_db:
            user_db.token = None
            db.commit()
        return {"success": True, "message": "登出成功"}
    finally:
        db.close()

# ==================== 技能端点 ====================
@app.get("/api/skills")
async def get_skills(
    category: Optional[str] = None,
    status: Optional[str] = None,
    installed: Optional[bool] = None,
    limit: int = 20,
    offset: int = 0
):
    """获取技能列表"""
    db = SessionLocal()
    try:
        query = db.query(Skill)
        
        if category:
            query = query.filter(Skill.category == category)
        if status:
            query = query.filter(Skill.status == status)
        if installed is not None:
            query = query.filter(Skill.installed == installed)
        
        skills = query.offset(offset).limit(limit).all()
        return {"skills": [skill_to_dict(s) for s in skills]}
    finally:
        db.close()

@app.get("/api/skills/{skill_id}")
async def get_skill(skill_id: str):
    """获取技能详情"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        return skill_to_dict(skill)
    finally:
        db.close()

@app.get("/api/skills/{skill_id}/versions")
async def get_skill_versions(skill_id: str):
    """获取技能版本列表"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        versions = db.query(SkillVersion).filter(SkillVersion.skill_id == skill.id).all()
        return {
            "versions": [
                {
                    "version": v.version,
                    "release_date": v.release_date.strftime("%Y-%m-%d") if v.release_date else None,
                    "changelog": v.changelog
                }
                for v in versions
            ]
        }
    finally:
        db.close()

@app.get("/api/skills/{skill_id}/check-update")
async def check_update(skill_id: str, current_version: str):
    """检查更新"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        # 简单版本比较
        current_parts = list(map(int, current_version.split('.')))
        latest_parts = list(map(int, skill.version.split('.')))
        
        has_update = latest_parts > current_parts
        
        return {
            "has_update": has_update,
            "latest_version": skill.version,
            "changelog": skill.changelog
        }
    finally:
        db.close()

@app.post("/api/skills")
async def create_skill(request: SkillRequest):
    """创建技能（支持批量注册，无需认证）"""
    db = SessionLocal()
    try:
        # 检查名称是否已存在
        if db.query(Skill).filter(Skill.name == request.name).first():
            return {"success": False, "message": "技能名称已存在"}
        
        # 使用请求中的 id 作为 skill_id
        skill_id = request.id
        
        # 检查 skill_id 是否已存在
        if db.query(Skill).filter(Skill.skill_id == skill_id).first():
            return {"success": False, "message": "技能 ID 已存在"}
        
        skill = Skill(
            name=request.name,
            skill_id=skill_id,
            description=request.description,
            category=request.category,
            version=request.version,
            author=request.author,
            status=request.status,
            entry_point=request.entry_point,
            runtime=request.runtime,
            tags=json.dumps(request.tags),
            dependencies=json.dumps(request.dependencies),
            config_schema=json.dumps(request.config_schema)
        )
        db.add(skill)
        db.commit()
        
        return {"success": True, "message": "技能创建成功"}
    finally:
        db.close()

@app.put("/api/skills/{skill_id}", dependencies=[Depends(get_current_user)])
async def update_skill(skill_id: str, request: SkillRequest):
    """更新技能"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        skill.name = request.name
        skill.description = request.description
        skill.category = request.category
        skill.version = request.version
        skill.author = request.author
        skill.status = request.status
        skill.entry_point = request.entry_point
        skill.runtime = request.runtime
        skill.tags = json.dumps(request.tags)
        skill.dependencies = json.dumps(request.dependencies)
        skill.config_schema = json.dumps(request.config_schema)
        db.commit()
        
        return {"success": True, "message": "技能更新成功"}
    finally:
        db.close()

@app.delete("/api/skills/{skill_id}", dependencies=[Depends(get_current_user)])
async def delete_skill(skill_id: str):
    """删除技能"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        db.delete(skill)
        db.commit()
        
        return {"success": True, "message": "技能删除成功"}
    finally:
        db.close()

@app.post("/api/skills/{skill_id}/upload", dependencies=[Depends(get_current_user)])
async def upload_skill(skill_id: str, file: UploadFile = File(...)):
    """上传技能包"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        # 保存文件
        file_path = SKILL_STORAGE / f"{skill_id}.zip"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        return {"success": True, "message": "上传成功"}
    finally:
        db.close()

@app.get("/api/skills/{skill_id}/download")
async def download_skill(skill_id: str):
    """下载技能包"""
    file_path = SKILL_STORAGE / f"{skill_id}.zip"
    if not file_path.exists():
        # 创建一个示例 zip 文件
        with zipfile.ZipFile(file_path, 'w') as zf:
            zf.writestr("skill.json", json.dumps({"name": skill_id, "version": "1.0.0"}))
    
    return FileResponse(file_path, filename=f"{skill_id}.zip")

@app.post("/api/skills/{skill_id}/install")
async def install_skill(skill_id: str, user: User = Depends(get_current_user)):
    """安装技能"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        skill.installed = True
        skill.downloads = (skill.downloads or 0) + 1
        db.commit()
        
        return {"success": True, "message": "安装成功"}
    finally:
        db.close()

@app.delete("/api/skills/{skill_id}/install")
async def uninstall_skill(skill_id: str, user: User = Depends(get_current_user)):
    """卸载技能"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        skill.installed = False
        db.commit()
        
        return {"success": True, "message": "卸载成功"}
    finally:
        db.close()

# ==================== 评分和评论端点 ====================
@app.post("/api/skills/{skill_id}/review")
async def add_review(
    skill_id: str,
    rating: int = Query(..., ge=1, le=5),
    comment: Optional[str] = "",
    user: User = Depends(get_current_user)
):
    """添加评分和评论"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        if rating < 1 or rating > 5:
            raise HTTPException(status_code=400, detail="评分必须在 1-5 之间")
        
        # 检查是否已评论
        existing_review = db.query(Review).filter(
            Review.skill_id == skill.id,
            Review.user_id == user.id
        ).first()
        if existing_review:
            raise HTTPException(status_code=400, detail="您已对此技能进行过评论")
        
        review = Review(
            skill_id=skill.id,
            user_id=user.id,
            rating=rating,
            comment=comment
        )
        db.add(review)
        
        # 更新技能评分
        reviews = db.query(Review).filter(Review.skill_id == skill.id).all()
        if reviews:
            avg_rating = sum(r.rating for r in reviews) / len(reviews)
            skill.rating = round(avg_rating, 1)
            skill.review_count = len(reviews)
        
        db.commit()
        
        return {"success": True, "message": "评论成功"}
    finally:
        db.close()

@app.get("/api/skills/{skill_id}/reviews")
async def get_reviews(skill_id: str, limit: int = 10):
    """获取技能评论列表"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        reviews = db.query(Review).filter(Review.skill_id == skill.id).order_by(
            Review.created_at.desc()
        ).limit(limit).all()
        
        result = []
        for review in reviews:
            user = db.query(User).filter(User.id == review.user_id).first()
            result.append({
                "id": review.id,
                "username": user.username if user else "unknown",
                "rating": review.rating,
                "comment": review.comment,
                "created_at": review.created_at.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return {"reviews": result}
    finally:
        db.close()

@app.get("/api/skills/{skill_id}/rating")
async def get_rating(skill_id: str):
    """获取技能评分"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        return {"rating": skill.rating or 0, "count": skill.review_count or 0}
    finally:
        db.close()

# ==================== 收藏端点 ====================
@app.post("/api/skills/{skill_id}/favorite")
async def add_favorite(skill_id: str, user: User = Depends(get_current_user)):
    """添加收藏"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        existing = db.query(Favorite).filter(
            Favorite.skill_id == skill.id,
            Favorite.user_id == user.id
        ).first()
        if existing:
            return {"success": False, "message": "已收藏此技能"}
        
        favorite = Favorite(skill_id=skill.id, user_id=user.id)
        db.add(favorite)
        db.commit()
        
        return {"success": True, "message": "收藏成功"}
    finally:
        db.close()

@app.delete("/api/skills/{skill_id}/favorite")
async def remove_favorite(skill_id: str, user: User = Depends(get_current_user)):
    """取消收藏"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        favorite = db.query(Favorite).filter(
            Favorite.skill_id == skill.id,
            Favorite.user_id == user.id
        ).first()
        if not favorite:
            return {"success": False, "message": "未收藏此技能"}
        
        db.delete(favorite)
        db.commit()
        
        return {"success": True, "message": "取消收藏成功"}
    finally:
        db.close()

@app.get("/api/favorites")
async def get_favorites(user: User = Depends(get_current_user)):
    """获取用户收藏列表"""
    db = SessionLocal()
    try:
        favorites = db.query(Favorite).filter(Favorite.user_id == user.id).all()
        skills = []
        for fav in favorites:
            skill = db.query(Skill).filter(Skill.id == fav.skill_id).first()
            if skill:
                skills.append(skill_to_dict(skill))
        return {"skills": skills}
    finally:
        db.close()

# ==================== 搜索端点 ====================
@app.get("/api/search")
async def search_skills(
    q: str = "",
    category: Optional[str] = None,
    status: Optional[str] = None,
    min_rating: Optional[float] = None,
    sort_by: Optional[str] = "downloads",
    sort_order: Optional[str] = "desc",
    limit: int = 20,
    offset: int = 0
):
    """搜索技能"""
    db = SessionLocal()
    try:
        query = db.query(Skill)
        
        if q:
            q_lower = q.lower()
            query = query.filter(
                (Skill.name.ilike(f"%{q_lower}%")) |
                (Skill.description.ilike(f"%{q_lower}%")) |
                (Skill.author.ilike(f"%{q_lower}%"))
            )
        
        if category:
            query = query.filter(Skill.category == category)
        
        if status:
            query = query.filter(Skill.status == status)
        
        if min_rating:
            query = query.filter(Skill.rating >= min_rating)
        
        # 排序
        sort_column = {
            "downloads": Skill.downloads,
            "rating": Skill.rating,
            "review_count": Skill.review_count,
            "updated_at": Skill.updated_at
        }.get(sort_by, Skill.downloads)
        
        if sort_order == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column)
        
        total = query.count()
        skills = query.offset(offset).limit(limit).all()
        
        return {
            "skills": [skill_to_dict(s) for s in skills],
            "total": total,
            "limit": limit,
            "offset": offset
        }
    finally:
        db.close()

# ==================== 排行榜端点 ====================
@app.get("/api/rankings")
async def get_rankings(type: str = "downloads", limit: int = 10):
    """获取技能排行榜"""
    db = SessionLocal()
    try:
        query = db.query(Skill)
        
        if type == "rating":
            query = query.filter(Skill.rating > 0).order_by(Skill.rating.desc())
        elif type == "reviews":
            query = query.order_by(Skill.review_count.desc())
        else:
            query = query.order_by(Skill.downloads.desc())
        
        skills = query.limit(limit).all()
        
        return {
            "type": type,
            "skills": [skill_to_dict(s) for s in skills],
            "total": len(skills)
        }
    finally:
        db.close()

@app.get("/api/rankings/all")
async def get_all_rankings(limit: int = 5):
    """获取所有排行榜数据"""
    db = SessionLocal()
    try:
        download_ranking = db.query(Skill).order_by(Skill.downloads.desc()).limit(limit).all()
        rating_ranking = db.query(Skill).filter(Skill.rating > 0).order_by(Skill.rating.desc()).limit(limit).all()
        review_ranking = db.query(Skill).order_by(Skill.review_count.desc()).limit(limit).all()
        
        return {
            "downloads": [skill_to_dict(s) for s in download_ranking],
            "rating": [skill_to_dict(s) for s in rating_ranking],
            "reviews": [skill_to_dict(s) for s in review_ranking]
        }
    finally:
        db.close()

# ==================== 分类端点 ====================
@app.get("/api/categories")
async def get_categories():
    """获取所有分类"""
    db = SessionLocal()
    try:
        categories = db.query(Skill.category, func.count(Skill.id)).group_by(Skill.category).all()
        return {cat[0]: cat[1] for cat in categories}
    finally:
        db.close()

# ==================== 统计端点 ====================
@app.get("/api/stats")
async def get_stats():
    """获取统计数据"""
    db = SessionLocal()
    try:
        total_skills = db.query(Skill).count()
        total_categories = db.query(Skill.category).distinct().count()
        total_downloads = db.query(func.sum(Skill.downloads)).scalar() or 0
        total_users = db.query(User).count()
        
        return {
            "total_skills": total_skills,
            "total_categories": total_categories,
            "total_downloads": total_downloads,
            "total_users": total_users
        }
    finally:
        db.close()

# ==================== 截图管理端点 ====================
@app.get("/api/skills/{skill_id}/screenshots")
async def get_screenshots(skill_id: str):
    """获取技能截图列表"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        screenshots = db.query(Screenshot).filter(Screenshot.skill_id == skill.id).all()
        return {
            "screenshots": [
                {
                    "url": s.url,
                    "caption": s.caption,
                    "added_at": s.added_at.strftime("%Y-%m-%d %H:%M:%S")
                }
                for s in screenshots
            ]
        }
    finally:
        db.close()

@app.post("/api/skills/{skill_id}/screenshots", dependencies=[Depends(get_current_user)])
async def add_screenshot(skill_id: str, url: str = Query(...), caption: Optional[str] = ""):
    """添加技能截图"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        screenshot = Screenshot(skill_id=skill.id, url=url, caption=caption)
        db.add(screenshot)
        db.commit()
        
        return {"success": True, "message": "截图添加成功"}
    finally:
        db.close()

# ==================== 版本回滚端点 ====================
@app.post("/api/skills/{skill_id}/rollback")
async def rollback_version(
    skill_id: str,
    target_version: str = Query(...),
    user: User = Depends(get_current_user)
):
    """回滚到指定版本"""
    db = SessionLocal()
    try:
        skill = db.query(Skill).filter(Skill.skill_id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="技能不存在")
        
        # 检查目标版本是否存在
        version = db.query(SkillVersion).filter(
            SkillVersion.skill_id == skill.id,
            SkillVersion.version == target_version
        ).first()
        if not version:
            raise HTTPException(status_code=400, detail="目标版本不存在")
        
        # 添加安装历史
        history = InstallHistory(
            user_id=user.id,
            skill_id=skill.id,
            version=skill.version,
            action="rollback_from"
        )
        db.add(history)
        
        # 回滚版本
        skill.version = target_version
        db.commit()
        
        return {"success": True, "message": f"已回滚到版本 {target_version}"}
    finally:
        db.close()

# ==================== 启动服务 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
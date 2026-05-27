#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARD SkillHub 数据库模型
使用 SQLite + SQLAlchemy 实现数据持久化
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# 创建数据库引擎
SQLALCHEMY_DATABASE_URL = "sqlite:///./skillhub.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基类
Base = declarative_base()

# ==================== 模型定义 ====================

class User(Base):
    """用户模型"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    email = Column(String)
    role = Column(String, default="user")
    token = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 关联关系
    reviews = relationship("Review", back_populates="user")
    favorites = relationship("Favorite", back_populates="user")

class Skill(Base):
    """技能模型"""
    __tablename__ = "skills"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    skill_id = Column(String, unique=True, index=True, nullable=False)  # 唯一标识
    description = Column(Text)
    category = Column(String)
    status = Column(String, default="stable")
    version = Column(String, default="1.0.0")
    author = Column(String)
    downloads = Column(Integer, default=0)
    rating = Column(Float, default=0.0)
    review_count = Column(Integer, default=0)
    installed = Column(Boolean, default=False)
    has_update = Column(Boolean, default=False)
    changelog = Column(Text)
    dependencies = Column(Text)  # JSON 格式存储
    tags = Column(Text)  # JSON 格式存储
    config_schema = Column(Text)  # JSON 格式存储
    entry_point = Column(String)  # 技能入口文件路径
    runtime = Column(String, default="python")  # 运行时环境
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 关联关系
    versions = relationship("SkillVersion", back_populates="skill")
    reviews = relationship("Review", back_populates="skill")
    favorites = relationship("Favorite", back_populates="skill")
    screenshots = relationship("Screenshot", back_populates="skill")

class SkillVersion(Base):
    """技能版本模型"""
    __tablename__ = "skill_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    skill_id = Column(Integer, ForeignKey("skills.id"))
    version = Column(String, nullable=False)
    release_date = Column(DateTime)
    changelog = Column(Text)
    file_path = Column(String)
    
    # 关联关系
    skill = relationship("Skill", back_populates="versions")

class Review(Base):
    """评论模型"""
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    skill_id = Column(Integer, ForeignKey("skills.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    rating = Column(Integer, nullable=False)
    comment = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    
    # 关联关系
    skill = relationship("Skill", back_populates="reviews")
    user = relationship("User", back_populates="reviews")

class Favorite(Base):
    """收藏模型"""
    __tablename__ = "favorites"
    
    id = Column(Integer, primary_key=True, index=True)
    skill_id = Column(Integer, ForeignKey("skills.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.now)
    
    # 关联关系
    skill = relationship("Skill", back_populates="favorites")
    user = relationship("User", back_populates="favorites")

class Screenshot(Base):
    """截图模型"""
    __tablename__ = "screenshots"
    
    id = Column(Integer, primary_key=True, index=True)
    skill_id = Column(Integer, ForeignKey("skills.id"))
    url = Column(String, nullable=False)
    caption = Column(String)
    added_at = Column(DateTime, default=datetime.now)
    
    # 关联关系
    skill = relationship("Skill", back_populates="screenshots")

class InstallHistory(Base):
    """安装历史模型"""
    __tablename__ = "install_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    skill_id = Column(Integer, ForeignKey("skills.id"))
    version = Column(String)
    action = Column(String)  # install, update, rollback
    installed_at = Column(DateTime, default=datetime.now)

class Notification(Base):
    """通知模型"""
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    skill_id = Column(Integer, ForeignKey("skills.id"))
    type = Column(String)  # update, new_skill, etc.
    message = Column(String)
    read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)

# ==================== 初始化函数 ====================

def init_db():
    """初始化数据库表"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================== 初始数据 ====================

INITIAL_SKILLS = [
    {
        "name": "ARD Collector",
        "skill_id": "ardc-collector",
        "description": "Anime Role Detect Collector - 数据采集技能，用于从多个数据源采集动漫角色数据",
        "category": "collector",
        "status": "stable",
        "version": "1.0.0",
        "author": "admin",
        "downloads": 128,
        "rating": 4.5,
        "review_count": 15,
        "tags": ["anime", "data", "collector"],
        "dependencies": ["requests", "beautifulsoup4"]
    },
    {
        "name": "ARD Cleaner",
        "skill_id": "ardc-cleaner",
        "description": "Anime Role Detect Cleaner - 数据清洗技能，用于清洗和标准化采集到的数据",
        "category": "cleaner",
        "status": "stable",
        "version": "1.0.0",
        "author": "admin",
        "downloads": 96,
        "rating": 4.3,
        "review_count": 12,
        "tags": ["data", "cleaning", "preprocessing"],
        "dependencies": ["pandas", "numpy"]
    },
    {
        "name": "ARD Classifier",
        "skill_id": "ardc-classifier",
        "description": "Anime Role Detect Classifier - 角色分类技能，使用机器学习模型对角色进行分类",
        "category": "classifier",
        "status": "stable",
        "version": "1.1.0",
        "author": "admin",
        "downloads": 203,
        "rating": 4.8,
        "review_count": 28,
        "tags": ["machine-learning", "classification", "ai"],
        "dependencies": ["scikit-learn", "tensorflow"]
    },
    {
        "name": "ARD Trainer",
        "skill_id": "ardc-trainer",
        "description": "Anime Role Detect Trainer - 模型训练技能，用于训练和优化角色检测模型",
        "category": "trainer",
        "status": "testing",
        "version": "0.9.0",
        "author": "developer",
        "downloads": 64,
        "rating": 4.2,
        "review_count": 8,
        "tags": ["training", "machine-learning", "model"],
        "dependencies": ["tensorflow", "pytorch"]
    },
    {
        "name": "ARD Search",
        "skill_id": "ardc-search",
        "description": "Anime Role Detect Search - 搜索检索技能，提供高效的角色数据搜索功能",
        "category": "search",
        "status": "stable",
        "version": "1.0.0",
        "author": "admin",
        "downloads": 156,
        "rating": 4.6,
        "review_count": 20,
        "tags": ["search", "elasticsearch", "fulltext"],
        "dependencies": ["elasticsearch", "whoosh"]
    },
    {
        "name": "ARD Analyzer",
        "skill_id": "ardc-analyzer",
        "description": "Anime Role Detect Analyzer - 数据分析技能，提供角色数据的深度分析报告",
        "category": "analyzer",
        "status": "stable",
        "version": "1.0.0",
        "author": "admin",
        "downloads": 89,
        "rating": 4.4,
        "review_count": 11,
        "tags": ["analysis", "visualization", "report"],
        "dependencies": ["matplotlib", "seaborn", "plotly"]
    },
    {
        "name": "ARD Utility",
        "skill_id": "ardc-utility",
        "description": "Anime Role Detect Utility - 工具辅助技能，提供各种实用工具函数",
        "category": "utility",
        "status": "stable",
        "version": "1.2.0",
        "author": "developer",
        "downloads": 178,
        "rating": 4.7,
        "review_count": 22,
        "tags": ["utility", "tools", "helpers"],
        "dependencies": []
    }
]

INITIAL_USERS = [
    {
        "username": "admin",
        "plain_password": "admin123",
        "email": "admin@example.com",
        "role": "admin",
        "is_active": True
    },
    {
        "username": "developer",
        "plain_password": "developer123",
        "email": "dev@example.com",
        "role": "developer",
        "is_active": True
    },
    {
        "username": "user",
        "plain_password": "user123",
        "email": "user@example.com",
        "role": "user",
        "is_active": True
    }
]

def insert_initial_data(db):
    """插入初始数据"""
    import json
    
    # 检查是否已有数据
    if db.query(Skill).first():
        return  # 已有数据，跳过初始化
    
    # 插入用户
    for user_data in INITIAL_USERS:
        plain_password = user_data.pop("plain_password", None)
        if plain_password:
            import hashlib
            import os
            salt = hashlib.md5(os.urandom(16)).hexdigest()
            hashed = hashlib.sha256(f"{plain_password}{salt}".encode()).hexdigest()
            user_data["password"] = f"sha256${salt}${hashed}"
        user = User(**user_data)
        db.add(user)
    
    # 插入技能
    for skill_data in INITIAL_SKILLS:
        skill = Skill(
            name=skill_data["name"],
            skill_id=skill_data["skill_id"],
            description=skill_data["description"],
            category=skill_data["category"],
            status=skill_data["status"],
            version=skill_data["version"],
            author=skill_data["author"],
            downloads=skill_data["downloads"],
            rating=skill_data["rating"],
            review_count=skill_data["review_count"],
            tags=json.dumps(skill_data["tags"]),
            dependencies=json.dumps(skill_data["dependencies"])
        )
        db.add(skill)
    
    db.commit()

if __name__ == "__main__":
    # 创建表
    init_db()
    
    # 插入初始数据
    db = next(get_db())
    insert_initial_data(db)
    print("数据库初始化完成！")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模块
使用 SQLAlchemy + SQLite 实现并发安全的用户数据存储
"""

from sqlalchemy import create_engine, Column, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone
from typing import Optional

from ardc.utils.logging import get_logger

logger = get_logger(__name__)

# SQLite 数据库路径
DATABASE_URL = "sqlite:///./data/ardc.db"

# 创建引擎（SQLite 需要 check_same_thread=False）
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 基类
Base = declarative_base()


class DBUser(Base):
    """用户数据库模型"""
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_developer = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class TokenBlacklist(Base):
    """Token 黑名单模型"""
    __tablename__ = "token_blacklist"

    id = Column(String, primary_key=True, index=True)
    jti = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# 创建所有表
def init_db():
    """初始化数据库表"""
    import os
    # 确保 data 目录存在
    os.makedirs(os.path.dirname(DATABASE_URL.replace("sqlite:///", "")), exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info("数据库初始化完成")


def get_db() -> Session:
    """获取数据库会话（依赖注入）"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

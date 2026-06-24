#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模块
使用 SQLAlchemy + SQLite/PostgreSQL 实现并发安全的用户数据存储
支持连接池配置，提升并发性能
"""

from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Integer, Float, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from ardc.utils.logging import get_logger
from ardc.config import settings

logger = get_logger(__name__)


def create_db_engine(database_url: str) -> object:
    """创建数据库引擎，根据数据库类型配置不同参数"""
    connect_args = {}
    db_settings = settings.database

    # SQLite 特殊配置
    if database_url.startswith("sqlite://"):
        connect_args["check_same_thread"] = False
        # SQLite 不需要连接池（文件锁限制）
        return create_engine(database_url, connect_args=connect_args, echo=db_settings.echo_sql)

    # PostgreSQL/MySQL 配置连接池
    return create_engine(
        database_url,
        pool_size=db_settings.pool_size,
        max_overflow=db_settings.max_overflow,
        pool_timeout=db_settings.pool_timeout,
        pool_recycle=db_settings.pool_recycle,
        pool_pre_ping=True,  # 连接前检查有效性
        echo=db_settings.echo_sql,
    )


# 创建引擎
engine = create_db_engine(settings.database.url)

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, expire_on_commit=False  # 提高性能，避免重复查询
)

# 基类
Base = declarative_base()

logger.info(f"数据库引擎初始化完成: {settings.database.url}")
logger.info(
    f"连接池配置: pool_size={settings.database.pool_size}, max_overflow={settings.database.max_overflow}, timeout={settings.database.pool_timeout}s"
)


class DBUser(Base):
    """用户数据库模型"""

    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_developer = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class TokenBlacklist(Base):
    """Token 黑名单模型"""

    __tablename__ = "token_blacklist"

    id = Column(String, primary_key=True, index=True)
    jti = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SkillReview(Base):
    """技能评价模型"""

    __tablename__ = "skill_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    skill_id = Column(String, index=True, nullable=False)
    username = Column(String, nullable=False, default="anonymous")
    rating = Column(Integer, nullable=False)  # 1-5
    comment = Column(Text, nullable=True, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("skill_id", "username", name="uq_skill_user_review"),
    )


# 创建所有表
def init_db():
    """初始化数据库表"""
    # 确保 data 目录存在
    db_path = settings.database.url.replace("sqlite:///", "")
    if db_path:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info("数据库初始化完成")


def get_db() -> Session:
    """获取数据库会话（依赖注入）"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

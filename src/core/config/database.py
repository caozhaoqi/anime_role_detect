#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库配置
提供数据库连接和会话管理
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
from src.core.logging.global_logger import get_logger

logger = get_logger("database")

DATABASE_URL = os.environ.get(
    'DATABASE_URL',
    'sqlite+aiosqlite:///./recognition.db'
)

USE_SQLITE = DATABASE_URL.startswith('sqlite')

engine = None
SessionLocal = None
Base = declarative_base()

def get_database_url() -> str:
    """获取数据库URL"""
    return DATABASE_URL

def init_database():
    """初始化数据库引擎"""
    global engine, SessionLocal

    if USE_SQLITE:
        engine = create_engine(
            DATABASE_URL.replace('+aiosqlite', ''),
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
        logger.info(f"SQLite数据库初始化: {DATABASE_URL}")
    else:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20
        )
        logger.info(f"数据库引擎初始化: {DATABASE_URL}")

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """获取数据库会话"""
    global SessionLocal

    if SessionLocal is None:
        init_database()

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """创建所有表"""
    global engine, Base

    if engine is None:
        init_database()

    from src.models.database_models import RecognitionRecordModel
    Base.metadata.create_all(bind=engine)
    logger.info("数据库表创建完成")

def drop_tables():
    """删除所有表"""
    global engine, Base

    if engine is None:
        init_database()

    Base.metadata.drop_all(bind=engine)
    logger.info("数据库表删除完成")

def reset_database():
    """重置数据库"""
    drop_tables()
    create_tables()
    logger.info("数据库重置完成")
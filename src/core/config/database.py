#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库配置
支持双数据库：SQLite（本地） + MySQL（远程）
"""

import os
from typing import Generator, Optional, Any
from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("database")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("dotenv 模块不可用")

try:
    from sqlalchemy import create_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import QueuePool
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    logger.warning("sqlalchemy 模块不可用，数据库功能将不可用")

try:
    from src.core.base_model import EnhancedBase
    HAS_ENHANCED_BASE = True
except ImportError:
    HAS_ENHANCED_BASE = False
    logger.warning("EnhancedBase 不可用，使用标准 Base")

if HAS_SQLALCHEMY:
    _declarative_base = declarative_base()
    
    if HAS_ENHANCED_BASE:
        class Base(_declarative_base, EnhancedBase):
            __abstract__ = True
    else:
        Base = _declarative_base
else:
    Base = None
    Session = Any

SQLITE_URL = os.environ.get(
    "SQLITE_URL",
    f"sqlite:///{os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data', 'recognition.db')}",
)

if HAS_SQLALCHEMY:
    from urllib.parse import quote_plus

    MYSQL_URL = os.environ.get("MYSQL_URL")

    if not MYSQL_URL:
        mysql_host = os.environ.get("MYSQL_HOST", "")
        mysql_port = os.environ.get("MYSQL_PORT", "3306")
        mysql_user = os.environ.get("MYSQL_USER", "")
        mysql_password = os.environ.get("MYSQL_PASSWORD", "")
        mysql_db = os.environ.get("MYSQL_DB", "")

        if mysql_host and mysql_user and mysql_password:
            encoded_password = quote_plus(mysql_password)
            MYSQL_URL = f"mysql+pymysql://{mysql_user}:{encoded_password}@{mysql_host}:{mysql_port}/{mysql_db}?charset=utf8mb4"
else:
    MYSQL_URL = None

DATABASE_MODE = os.environ.get("DATABASE_MODE", "sqlite").lower()

LOCAL_DB_URL = SQLITE_URL
REMOTE_DB_URL = MYSQL_URL

_local_engine = None
_remote_engine = None
_local_session = None
_remote_session = None


def get_local_db_url() -> str:
    """获取本地数据库URL"""
    return LOCAL_DB_URL


def get_remote_db_url() -> str:
    """获取远程数据库URL"""
    return REMOTE_DB_URL or ""


def is_remote_available() -> bool:
    """检查远程数据库是否可用"""
    return bool(REMOTE_DB_URL)


def get_database_mode() -> str:
    """获取数据库模式"""
    return DATABASE_MODE


def init_local_database():
    """初始化本地SQLite数据库"""
    global _local_engine, _local_session

    if not HAS_SQLALCHEMY:
        logger.warning("sqlalchemy 不可用，无法初始化数据库")
        return

    db_path = LOCAL_DB_URL
    if db_path.startswith("sqlite:///"):
        file_path = db_path[len("sqlite:///"):]
        db_dir = os.path.dirname(file_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        logger.info(f"SQLite数据库路径: {file_path}")

    # P1-3: 从 ServiceConfig 读取连接池参数
    try:
        from src.core.config.service_config import get_service_config
        _svc_config = get_service_config()
        _db_pool_size = _svc_config.DB_POOL_SIZE
        _db_max_overflow = _svc_config.DB_MAX_OVERFLOW
    except Exception:
        _db_pool_size = 5
        _db_max_overflow = 10

    # P1-3: SQLite 使用 QueuePool 替代 StaticPool，支持连接池化
    _local_engine = create_engine(
        LOCAL_DB_URL,
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=QueuePool,
        pool_size=_db_pool_size,
        max_overflow=_db_max_overflow,
        pool_pre_ping=True,
        echo=False,
    )
    logger.info(f"本地SQLite数据库初始化: {LOCAL_DB_URL}")

    _local_session = sessionmaker(autocommit=False, autoflush=False, bind=_local_engine)


def init_remote_database():
    """初始化远程MySQL数据库"""
    global _remote_engine, _remote_session

    if not HAS_SQLALCHEMY:
        logger.warning("sqlalchemy 不可用，无法初始化远程数据库")
        return

    if not REMOTE_DB_URL:
        logger.warning("远程MySQL数据库配置未设置，跳过初始化")
        return

    try:
        from src.core.config.service_config import get_service_config
        _svc_config = get_service_config()
        _mysql_pool_size = _svc_config.DB_POOL_SIZE
        _mysql_max_overflow = _svc_config.DB_MAX_OVERFLOW
    except Exception:
        _mysql_pool_size = 5
        _mysql_max_overflow = 10

    try:
        _engine = create_engine(
            REMOTE_DB_URL,
            pool_pre_ping=True,
            pool_size=_mysql_pool_size,
            max_overflow=_mysql_max_overflow,
            connect_args={"connect_timeout": 10},
            echo=False,
        )

        # 真实连接探测：create_engine 是懒连接，必须实际连一次确认可用，
        # 否则后续所有 DB 操作都会抛 OperationalError，且无法触发本地降级
        try:
            from sqlalchemy import text as _sa_text
            with _engine.connect() as _conn:
                _conn.execute(_sa_text("SELECT 1"))
        except Exception as e:
            logger.error(f"远程MySQL连接探测失败，降级使用本地SQLite: {e}")
            _remote_engine = None
            _remote_session = None
            return False

        _remote_engine = _engine
        logger.info(f"远程MySQL数据库初始化: {REMOTE_DB_URL}")

        _remote_session = sessionmaker(autocommit=False, autoflush=False, bind=_remote_engine)
        return True
    except Exception as e:
        logger.error(f"远程MySQL数据库初始化失败: {e}")
        return False


def init_database(mode: Optional[str] = None):
    """初始化数据库引擎"""
    if not HAS_SQLALCHEMY:
        logger.warning("sqlalchemy 不可用，跳过数据库初始化")
        return

    current_mode = mode or DATABASE_MODE

    logger.info(f"初始化数据库，模式: {current_mode}")

    init_local_database()

    if current_mode in ["remote", "dual"]:
        init_remote_database()


def get_local_db() -> Generator[Session, None, None]:
    """获取本地数据库会话"""
    if not HAS_SQLALCHEMY:
        logger.warning("sqlalchemy 不可用，无法获取数据库会话")
        yield None
        return

    global _local_session

    if _local_session is None:
        init_local_database()

    db = _local_session()
    try:
        yield db
    finally:
        db.close()


def get_remote_db() -> Generator[Session, None, None]:
    """获取远程数据库会话"""
    if not HAS_SQLALCHEMY:
        logger.warning("sqlalchemy 不可用，无法获取远程数据库会话")
        yield None
        return

    global _remote_session

    if _remote_session is None:
        init_remote_database()

    if _remote_session is None:
        logger.warning("远程数据库不可用，回退到本地数据库")
        yield from get_local_db()
        return

    db = _remote_session()
    try:
        yield db
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    """获取数据库会话（根据模式选择）"""
    if not HAS_SQLALCHEMY:
        logger.warning("sqlalchemy 不可用，无法获取数据库会话")
        yield None
        return

    mode = DATABASE_MODE

    if mode == "remote":
        yield from get_remote_db()
    elif mode == "dual":
        yield from get_remote_db()
    else:
        yield from get_local_db()


def get_db_session(mode: str = "auto") -> Session:
    """获取数据库会话（直接返回）"""
    if not HAS_SQLALCHEMY:
        logger.warning("sqlalchemy 不可用，无法获取数据库会话")
        return None

    if mode == "auto":
        mode = DATABASE_MODE

    if mode == "remote":
        if _remote_session is None:
            init_remote_database()
        if _remote_session:
            return _remote_session()
        logger.warning("远程数据库不可用，使用本地数据库")
        return get_local_db_session()

    elif mode == "local":
        return get_local_db_session()

    elif mode == "dual":
        if _remote_session is None:
            init_remote_database()
        if _remote_session:
            return _remote_session()
        return get_local_db_session()

    return get_local_db_session()


def get_local_db_session() -> Session:
    """获取本地数据库会话（直接返回）"""
    if not HAS_SQLALCHEMY:
        logger.warning("sqlalchemy 不可用，无法获取本地数据库会话")
        return None

    if _local_session is None:
        init_local_database()
    return _local_session()


def get_remote_db_session() -> Optional[Session]:
    """获取远程数据库会话（直接返回）"""
    if not HAS_SQLALCHEMY:
        logger.warning("sqlalchemy 不可用，无法获取远程数据库会话")
        return None

    if _remote_session is None:
        init_remote_database()
    return _remote_session() if _remote_session else None


def create_tables(engine=None):
    """创建所有表"""
    if not HAS_SQLALCHEMY or Base is None:
        logger.warning("sqlalchemy 不可用，无法创建数据库表")
        return

    target_engine = engine or _local_engine

    if target_engine is None:
        init_local_database()
        target_engine = _local_engine

    from src.models.database_models import (
        UserModel,
        RecognitionRecordModel,
        ApiKeyModel,
        SystemConfigModel,
        CleaningRecordModel,
        UserFeedbackModel,
    )

    Base.metadata.create_all(bind=target_engine)
    logger.info(f"数据库表创建完成，引擎: {target_engine.url}")


def create_local_tables():
    """创建本地表"""
    create_tables(_local_engine)


def create_remote_tables():
    """创建远程表"""
    if _remote_engine:
        create_tables(_remote_engine)


def drop_tables(engine=None):
    """删除所有表"""
    if not HAS_SQLALCHEMY or Base is None:
        logger.warning("sqlalchemy 不可用，无法删除数据库表")
        return

    target_engine = engine or _local_engine

    if target_engine is None:
        init_local_database()
        target_engine = _local_engine

    Base.metadata.drop_all(bind=target_engine)
    logger.info(f"数据库表删除完成，引擎: {target_engine.url}")


def reset_database():
    """重置数据库"""
    drop_tables()
    create_tables()
    logger.info("数据库重置完成")


def sync_local_to_remote():
    """同步本地数据到远程数据库"""
    if not HAS_SQLALCHEMY:
        logger.warning("sqlalchemy 不可用，无法同步数据")
        return False

    if not is_remote_available() or _remote_session is None:
        logger.warning("远程数据库不可用，无法同步")
        return False

    try:
        local_db = get_local_db_session()
        remote_db = get_remote_db_session()

        if remote_db is None:
            local_db.close()
            return False

        from src.models.database_models import (
            UserModel,
            RecognitionRecordModel,
            ApiKeyModel,
            SystemConfigModel,
            CleaningRecordModel,
            UserFeedbackModel,
        )

        create_remote_tables()

        models_to_sync = [
            UserModel,
            SystemConfigModel,
            ApiKeyModel,
            RecognitionRecordModel,
            CleaningRecordModel,
            UserFeedbackModel,
        ]

        synced_count = 0
        for model in models_to_sync:
            local_records = local_db.query(model).all()
            for record in local_records:
                existing = remote_db.query(model).filter_by(id=record.id).first()
                if not existing:
                    new_record = model()
                    for col in model.__table__.columns:
                        col_name = col.name
                        if hasattr(record, col_name):
                            setattr(new_record, col_name, getattr(record, col_name))
                    remote_db.add(new_record)
                    synced_count += 1

        remote_db.commit()
        local_db.close()
        remote_db.close()

        logger.info(f"本地数据同步到远程数据库完成，共同步 {synced_count} 条记录")
        return True

    except Exception as e:
        logger.error(f"数据同步失败: {e}")
        return False


def close_all_sessions():
    """关闭所有数据库会话"""
    global _local_session, _remote_session
    _local_session = None
    _remote_session = None
    logger.info("所有数据库会话已关闭")


def get_engine_info() -> dict:
    """获取数据库引擎信息"""
    info = {
        "mode": DATABASE_MODE,
        "local_url": LOCAL_DB_URL,
        "remote_url": REMOTE_DB_URL or "未配置",
        "remote_available": is_remote_available(),
        "local_engine_exists": _local_engine is not None,
        "remote_engine_exists": _remote_engine is not None,
        "has_sqlalchemy": HAS_SQLALCHEMY,
    }
    return info

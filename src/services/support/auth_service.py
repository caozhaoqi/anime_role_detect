#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认证服务 - 支持 MySQL → SQLite → 内存 三层降级存储用户信息
"""

import os
import hashlib
import time
import secrets
import json
import sqlite3
import threading
import concurrent.futures
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("auth_service")

def _get_default_password(username: str) -> str:
    """从环境变量获取默认密码，未设置时生成随机密码"""
    env_var = f"{username.upper()}_PASSWORD"
    pwd = os.environ.get(env_var)
    if pwd:
        return pwd
    random_pwd = secrets.token_urlsafe(12)
    logger.warning(f"⚠️  {env_var} 未设置，为用户 {username} 生成随机密码: {random_pwd}")
    return random_pwd

try:
    import jwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False
    logger.warning("jwt 模块不可用，令牌功能将不可用")

try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False
    logger.warning("bcrypt 模块不可用，回退到 SHA-256 加盐哈希")

try:
    from src.core.config.database import get_db, init_database, create_tables, get_database_mode, is_remote_connected
    from src.models.database_models import UserModel
    HAS_DATABASE = True
except ImportError as e:
    HAS_DATABASE = False
    logger.warning(f"数据库模块不可用: {e}")

try:
    from sqlalchemy.exc import OperationalError, InterfaceError
except ImportError:
    # SQLAlchemy 不可用时退化为通用异常，避免导入错误
    OperationalError = InterfaceError = Exception


def _is_mysql_connection_failure(exc: Exception) -> bool:
    """判断异常是否代表 MySQL 不可达（连接/驱动/超时/鉴权等），应触发降级。

    注意：不能用固定字符串匹配（如仅 'Lost connection'/'Can't connect'），
    否则在驱动缺失（No module named 'pymysql'）、DNS 失败、连接超时、鉴权失败等
    场景下无法触发降级，导致「写入走 SQLite 降级、读取仍查 MySQL 而返回 None」
    的认证不一致（注册成功但登录失败）。
    """
    if isinstance(exc, (OperationalError, InterfaceError)):
        return True
    msg = str(exc)
    keywords = (
        "Lost connection", "Can't connect", "Can not connect", "could not connect",
        "OperationalError", "InterfaceError", "2003", "2006", "2013",
        "pymysql", "MySQLdb", "No module named", "ModuleNotFoundError",
        "timed out", "timeout", "refused", "Unknown MySQL server",
        "Access denied",
    )
    return any(k in msg for k in keywords)


class SQLiteUserStore:
    """SQLite 本地用户存储 — 作为 MySQL 不可用时的降级方案"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # 默认存储到项目 data 目录
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            data_dir = os.path.join(project_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "auth.db")
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self):
        """初始化 SQLite 表结构"""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT DEFAULT 'user',
                        email TEXT,
                        is_active INTEGER DEFAULT 1,
                        is_superuser INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT (datetime('now')),
                        updated_at TEXT DEFAULT (datetime('now')),
                        last_login_at TEXT,
                        login_count INTEGER DEFAULT 0,
                        failed_login_count INTEGER DEFAULT 0,
                        locked_until TEXT
                    )
                """)
                conn.commit()
            logger.info(f"SQLite 用户存储初始化完成: {self.db_path}")
        except Exception as e:
            logger.error(f"SQLite 初始化失败: {e}")
            raise

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """获取用户信息（不按 is_active 过滤，锁定用户也可查到）"""
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM users WHERE username=?",
                    (username,)
                ).fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"SQLite 查询用户失败: {e}")
            return None

    def create_user(self, username: str, password_hash: str, role: str = "user",
                    email: str = None, is_superuser: bool = False) -> Optional[Dict[str, Any]]:
        """创建新用户"""
        with self._lock:
            try:
                with self._get_conn() as conn:
                    # 检查用户名
                    existing = conn.execute(
                        "SELECT id FROM users WHERE username=?", (username,)
                    ).fetchone()
                    if existing:
                        logger.warning(f"SQLite 用户名已存在: {username}")
                        return None

                    cursor = conn.execute(
                        """INSERT INTO users (username, password_hash, role, email, is_active, is_superuser)
                           VALUES (?, ?, ?, ?, 1, ?)""",
                        (username, password_hash, role, email, 1 if is_superuser else 0)
                    )
                    conn.commit()
                    user_id = cursor.lastrowid
                    logger.info(f"SQLite 创建用户成功: {username} (id={user_id})")
                    return {
                        "id": user_id,
                        "username": username,
                        "role": role,
                        "email": email,
                        "is_active": True,
                        "is_superuser": is_superuser,
                    }
            except Exception as e:
                logger.error(f"SQLite 创建用户失败: {e}")
                return None

    def verify_password(self, username: str, plain_password: str, verify_fn) -> Optional[Dict[str, Any]]:
        """验证密码"""
        user = self.get_user(username)
        if not user:
            return None

        # 检查锁定
        if user.get("locked_until"):
            try:
                locked_until = datetime.fromisoformat(user["locked_until"])
                if locked_until.tzinfo is None:
                    locked_until = locked_until.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                if locked_until > now:
                    remaining = (locked_until - now).total_seconds() // 60
                    logger.warning(f"SQLite 用户 {username} 已被锁定，剩余 {remaining} 分钟")
                    return {"user": user, "status": "locked"}
            except (ValueError, TypeError):
                pass

        if verify_fn(plain_password, user.get("password_hash", "")):
            # 更新登录信息
            self._update_login_info(user.get("id"), success=True)
            return {"user": user, "status": "ok"}

        self._update_login_info(user.get("id"), success=False)
        return None

    def _update_login_info(self, user_id: int, success: bool = True):
        """更新登录统计"""
        try:
            with self._get_conn() as conn:
                if success:
                    conn.execute(
                        """UPDATE users SET last_login_at=datetime('now'),
                           login_count=login_count+1, failed_login_count=0, locked_until=NULL
                           WHERE id=?""", (user_id,)
                    )
                else:
                    conn.execute(
                        """UPDATE users SET failed_login_count=failed_login_count+1
                           WHERE id=?""", (user_id,)
                    )
                    # 5次失败锁定
                    row = conn.execute(
                        "SELECT failed_login_count FROM users WHERE id=?", (user_id,)
                    ).fetchone()
                    if row and row["failed_login_count"] >= 5:
                        lock_until = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
                        conn.execute(
                            "UPDATE users SET locked_until=? WHERE id=?",
                            (lock_until, user_id)
                        )
                conn.commit()
        except Exception as e:
            logger.error(f"SQLite 更新登录信息失败: {e}")

    def list_users(self) -> List[Dict[str, Any]]:
        """列出所有用户"""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT username, role, email, is_active, is_superuser, created_at FROM users ORDER BY created_at DESC"
                ).fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"SQLite 列出用户失败: {e}")
            return []


class AuthService:
    """认证服务 - 支持数据库存储"""

    _instance: Optional["AuthService"] = None

    # ========== 密码哈希 ==========
    @staticmethod
    def _hash_password(password: str) -> str:
        """哈希密码，优先 bcrypt，回退 SHA-256 加盐"""
        if HAS_BCRYPT:
            return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        salt = secrets.token_hex(16)
        return f"sha256${salt}${hashlib.sha256((password + salt).encode()).hexdigest()}"

    @staticmethod
    def _verify_password(plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        if hashed_password.startswith("sha256$"):
            parts = hashed_password.split("$")
            if len(parts) != 3:
                return False
            _, salt, stored_hash = parts
            return hashlib.sha256((plain_password + salt).encode()).hexdigest() == stored_hash
        if HAS_BCRYPT:
            return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
        return False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    STORAGE_MODE_MYSQL = "mysql"
    STORAGE_MODE_SQLITE = "sqlite"
    STORAGE_MODE_MEMORY = "memory"

    def __init__(self):
        if hasattr(self, "initialized") and self.initialized:
            return

        self.initialized = True

        # 配置
        self.SECRET_KEY = os.environ.get(
            "SECRET_KEY",
            secrets.token_hex(32)  # 每次启动生成临时密钥，生产环境必须通过环境变量注入
        )
        self.ALGORITHM = os.environ.get("ALGORITHM", "HS256")
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        self.REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

        # 存储层级：MySQL → SQLite → Memory
        self.storage_mode = self.STORAGE_MODE_MEMORY  # 默认
        self.users = {}  # 内存兜底，始终初始化
        self.sqlite_store: Optional[SQLiteUserStore] = None

        # ---- 第 1 层：尝试 MySQL ----
        mysql_ok = False
        if HAS_DATABASE:
            try:
                init_database()
                create_tables()
                # 修复（storage_mode 上报不一致，2026-08-10）：
                # 仅当 DATABASE_MODE 为 remote/dual 且 MySQL 操作成功才判定 MySQL 模式。
                # 原实现只看"初始化+建表+建默认用户成功"——在 SQLite 引擎上同样会成功，
                # 导致 sqlite 降级时误报 MySQL（k8s 里 api-service 降级 sqlite 仍报 mysql）。
                db_mode = get_database_mode()
                if db_mode in ("remote", "dual") and is_remote_connected():
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(self._ensure_default_users)
                            future.result(timeout=10)
                        self.storage_mode = self.STORAGE_MODE_MYSQL
                        mysql_ok = True
                        logger.info(f"认证服务初始化完成（MySQL 模式，DATABASE_MODE={db_mode}）")
                    except concurrent.futures.TimeoutError:
                        logger.warning("[DEGRADE] MySQL _ensure_default_users 超时(10s)")
                    except Exception as e:
                        logger.warning(f"[DEGRADE] MySQL 创建默认用户失败: {e}")
                else:
                    logger.info(
                        f"数据库模式 {db_mode}（远程未连接），认证存储走 SQLite/内存层"
                        if db_mode in ("remote", "dual")
                        else f"数据库模式 {db_mode}，认证存储走 SQLite/内存层"
                    )
            except Exception as e:
                logger.warning(f"[DEGRADE] 数据库初始化失败: {e}")

        # ---- 第 2 层：尝试 SQLite ----
        if not mysql_ok:
            try:
                self.sqlite_store = SQLiteUserStore()
                # 确保默认用户存在于 SQLite
                self._ensure_sqlite_default_users()
                self.storage_mode = self.STORAGE_MODE_SQLITE
                logger.info("认证服务初始化完成（SQLite 模式）")
            except Exception as e:
                logger.warning(f"[DEGRADE] SQLite 初始化失败，回退到内存模式: {e}")
                self.sqlite_store = None

        # ---- 第 3 层：内存兜底 ----
        if self.storage_mode == self.STORAGE_MODE_MEMORY:
            self.users = {
                "admin": {"password": self._hash_password(_get_default_password("admin")), "role": "admin"},
                "user": {"password": self._hash_password(_get_default_password("user")), "role": "user"},
            }
            logger.info("认证服务初始化完成（内存模式，密码已哈希）")

        logger.info(f"当前存储模式: {self.storage_mode}")

    def _ensure_default_users(self):
        """确保默认用户存在（MySQL）"""
        if not HAS_DATABASE or not HAS_BCRYPT:
            return

        default_users = [
            {"username": "admin", "password": _get_default_password("admin"), "role": "admin", "is_superuser": True},
            {"username": "user", "password": _get_default_password("user"), "role": "user", "is_superuser": False},
        ]

        db = next(get_db())
        try:
            for user_data in default_users:
                existing_user = db.query(UserModel).filter_by(username=user_data["username"]).first()
                if not existing_user:
                    new_user = UserModel(
                        username=user_data["username"],
                        role=user_data["role"],
                        is_superuser=user_data["is_superuser"],
                        is_active=True
                    )
                    new_user.set_password(user_data["password"])
                    db.add(new_user)
                    logger.info(f"创建默认用户: {user_data['username']}")

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"创建默认用户失败: {e}")
        finally:
            db.close()

    def _ensure_sqlite_default_users(self):
        """确保默认用户存在于 SQLite"""
        if not self.sqlite_store:
            return
        default_users = [
            {"username": "admin", "password": _get_default_password("admin"), "role": "admin", "is_superuser": True},
            {"username": "user", "password": _get_default_password("user"), "role": "user", "is_superuser": False},
        ]
        for u in default_users:
            existing = self.sqlite_store.get_user(u["username"])
            if not existing:
                pwd_hash = self._hash_password(u["password"])
                self.sqlite_store.create_user(
                    username=u["username"],
                    password_hash=pwd_hash,
                    role=u["role"],
                    is_superuser=u["is_superuser"],
                )
                logger.info(f"SQLite 创建默认用户: {u['username']}")

    def _get_user_from_db(self, username: str) -> Optional[Dict[str, Any]]:
        """从存储获取用户 — 按优先级: MySQL → SQLite → Memory"""
        # 第 1 层: MySQL
        if self.storage_mode == self.STORAGE_MODE_MYSQL:
            try:
                db = next(get_db())
                user = db.query(UserModel).filter_by(username=username).first()
                db.close()
                if user:
                    return {
                        "id": user.id,
                        "username": user.username,
                        "role": user.role,
                        "email": getattr(user, "email", None),
                        "is_active": True,
                        "is_superuser": user.is_superuser,
                        "_mysql_obj": user,  # 保留原始对象用于 update_login_info
                    }
            except Exception as e:
                logger.error(f"MySQL 查询用户失败: {e}")
                if _is_mysql_connection_failure(e):
                    logger.warning("[DEGRADE] MySQL 不可达，尝试降级到 SQLite")
                    # 尝试降级到 SQLite
                    if self._try_downgrade_to_sqlite():
                        return self._get_user_from_sqlite(username)
                    else:
                        self._downgrade_to_memory()
                        return None

        # 第 2 层: SQLite
        if self.storage_mode == self.STORAGE_MODE_SQLITE:
            return self._get_user_from_sqlite(username)

        # 第 3 层: Memory — 不在这里处理，由 authenticate_user 调用 self.users
        return None

    def _get_user_from_sqlite(self, username: str) -> Optional[Dict[str, Any]]:
        """从 SQLite 获取用户"""
        if not self.sqlite_store:
            return None
        try:
            user = self.sqlite_store.get_user(username)
            return user
        except Exception as e:
            logger.error(f"SQLite 查询用户失败: {e}")
            self._downgrade_to_memory()
            return None

    def _try_downgrade_to_sqlite(self) -> bool:
        """尝试从 MySQL 降级到 SQLite"""
        try:
            self.sqlite_store = SQLiteUserStore()
            self._ensure_sqlite_default_users()
            self.storage_mode = self.STORAGE_MODE_SQLITE
            logger.info("[DEGRADE] 已降级到 SQLite 模式")
            return True
        except Exception as e:
            logger.error(f"[DEGRADE] SQLite 降级失败: {e}")
            return False

    def _downgrade_to_memory(self):
        """降级到内存模式"""
        self.storage_mode = self.STORAGE_MODE_MEMORY
        if not self.users:
            self.users = {
                "admin": {"password": self._hash_password(_get_default_password("admin")), "role": "admin"},
                "user": {"password": self._hash_password(_get_default_password("user")), "role": "user"},
            }
        logger.warning("[DEGRADE] 已降级到内存认证模式")

    def _update_user_login_info(self, user_data: Dict[str, Any], success: bool = True):
        """更新用户登录信息"""
        # MySQL 模式
        mysql_obj = user_data.get("_mysql_obj") if isinstance(user_data, dict) else None
        if mysql_obj and self.storage_mode == self.STORAGE_MODE_MYSQL:
            try:
                db = next(get_db())
                db_user = db.query(UserModel).filter_by(id=mysql_obj.id).first()
                if db_user:
                    if success:
                        db_user.last_login_at = datetime.now(timezone.utc)
                        db_user.login_count = (db_user.login_count or 0) + 1
                        db_user.failed_login_count = 0
                        db_user.locked_until = None
                    else:
                        db_user.failed_login_count = (db_user.failed_login_count or 0) + 1
                        if db_user.failed_login_count >= 5:
                            db_user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=10)
                    db.commit()
            except Exception as e:
                logger.error(f"更新用户登录信息失败: {e}")
            finally:
                db.close()
            return

        # SQLite 模式
        if self.storage_mode == self.STORAGE_MODE_SQLITE and self.sqlite_store:
            uid = user_data.get("id") if isinstance(user_data, dict) else None
            if uid:
                self.sqlite_store._update_login_info(int(uid), success=success)

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问令牌"""
        if not HAS_JWT:
            logger.warning("jwt 模块不可用，返回简单令牌")
            return f"simple_token_{secrets.token_hex(16)}"

        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
        return encoded_jwt

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """创建刷新令牌"""
        if not HAS_JWT:
            logger.warning("jwt 模块不可用，返回简单令牌")
            return f"simple_refresh_{secrets.token_hex(16)}"

        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
        return encoded_jwt

    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """使用刷新令牌获取新的访问令牌"""
        try:
            payload = jwt.decode(refresh_token, self.SECRET_KEY, algorithms=[self.ALGORITHM])

            if payload.get("type") != "refresh":
                logger.warning("无效的刷新令牌类型")
                return None

            username = payload.get("sub")
            if not username:
                logger.warning("刷新令牌缺少用户信息")
                return None

            # 查找用户（跨存储层）
            user_data = self._get_user_from_db(username)
            if user_data:
                user_role = user_data.get("role", "user")
            else:
                mem_user = self.users.get(username)
                if mem_user:
                    user_role = mem_user.get("role", "user")
                else:
                    logger.warning(f"刷新令牌用户不存在: {username}")
                    return None

            new_access_token = self.create_access_token({"sub": username, "role": user_role})

            return {
                "access_token": new_access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": self.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            }

        except jwt.ExpiredSignatureError:
            logger.warning("刷新令牌已过期")
            return None
        except jwt.PyJWTError as e:
            logger.error(f"刷新令牌验证失败: {e}")
            return None

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证令牌"""
        try:
            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            return payload
        except jwt.PyJWTError as e:
            logger.error(f"令牌验证失败: {e}")
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """验证用户 — 按优先级: MySQL → SQLite → Memory"""
        # 尝试查询用户
        user_data = self._get_user_from_db(username)

        if user_data:
            # MySQL 或 SQLite 用户
            mysql_obj = user_data.get("_mysql_obj")

            # 检查锁定
            if mysql_obj and mysql_obj.locked_until:
                locked_until = mysql_obj.locked_until
                if locked_until.tzinfo is None:
                    locked_until = locked_until.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                if locked_until > now:
                    remaining = (locked_until - now).total_seconds() // 60
                    logger.warning(f"用户 {username} 已被锁定，剩余 {remaining} 分钟")
                    return None

            if mysql_obj and HAS_BCRYPT:
                # MySQL 用户用 bcrypt 验证
                if mysql_obj.verify_password(password):
                    self._update_user_login_info(user_data, success=True)
                    return {
                        "id": mysql_obj.id,
                        "username": mysql_obj.username,
                        "role": mysql_obj.role,
                        "is_superuser": mysql_obj.is_superuser,
                    }
                else:
                    self._update_user_login_info(user_data, success=False)
                    return None
            else:
                # SQLite 用户 或 dual 模式下记录实际落在 SQLite（sqlite_store 可能未实例化）
                if self.sqlite_store:
                    result = self.sqlite_store.verify_password(
                        username, password, self._verify_password
                    )
                    if result and result.get("status") == "ok":
                        return {
                            "id": user_data.get("id"),
                            "username": username,
                            "role": user_data.get("role"),
                            "is_superuser": user_data.get("is_superuser", False),
                        }
                    elif result and result.get("status") == "locked":
                        return None
                    else:
                        return None
                elif mysql_obj is not None:
                    # dual 模式：记录实际写入 SQLite（get_db 降级），但 storage_mode 仍为 mysql 且
                    # sqlite_store 未实例化。直接调用 UserModel.verify_password 校验——该方法已正确
                    # 处理 bcrypt 与明文降级两种哈希方案，避免因读取错误字段（password_hash vs password）导致校验失败。
                    if mysql_obj.verify_password(password):
                        self._update_user_login_info(user_data, success=True)
                        return {
                            "id": getattr(mysql_obj, "id", user_data.get("id")),
                            "username": username,
                            "role": user_data.get("role"),
                            "is_superuser": user_data.get("is_superuser", False),
                        }
                    else:
                        self._update_user_login_info(user_data, success=False)
                        return None
                else:
                    return None

        # 回退到内存模式
        mem_user = self.users.get(username)
        if mem_user and self._verify_password(password, mem_user.get("password", "")):
            return {
                "id": username,
                "username": username,
                "role": mem_user.get("role"),
                "is_superuser": mem_user.get("role") == "admin",
            }

        return None

    def get_user_role(self, username: str) -> Optional[str]:
        """获取用户角色"""
        user_data = self._get_user_from_db(username)
        if user_data:
            return user_data.get("role")

        mem_user = self.users.get(username)
        return mem_user.get("role") if mem_user else None

    def is_admin(self, username: str) -> bool:
        """检查是否为管理员"""
        role = self.get_user_role(username)
        return role == "admin"

    def is_user(self, username: str) -> bool:
        """检查是否为普通用户"""
        role = self.get_user_role(username)
        return role == "user"

    def create_user(self, username: str, password: str, role: str = "user", email: str = None) -> Optional[Dict[str, Any]]:
        """创建新用户 — 按当前存储模式写入"""
        if not username or not password:
            logger.warning("用户名和密码不能为空")
            return None
        if len(username) < 2 or len(username) > 32:
            logger.warning(f"用户名长度不合法: {username}")
            return None
        if len(password) < 4:
            logger.warning("密码长度至少 4 位")
            return None

        password_hash = self._hash_password(password)
        is_superuser = (role == "admin")

        # MySQL 模式
        if self.storage_mode == self.STORAGE_MODE_MYSQL:
            try:
                db = next(get_db())
                existing = db.query(UserModel).filter_by(username=username).first()
                if existing:
                    logger.warning(f"用户名已存在: {username}")
                    return None

                new_user = UserModel(
                    username=username,
                    role=role,
                    email=email,
                    is_active=True,
                    is_superuser=is_superuser
                )
                new_user.set_password(password)
                db.add(new_user)
                db.commit()
                db.refresh(new_user)
                logger.info(f"MySQL 创建用户成功: {username}")
                return new_user.to_dict()
            except Exception as e:
                db.rollback()
                logger.error(f"MySQL 创建用户失败: {e}")
                # 降级到 SQLite
                if _is_mysql_connection_failure(e):
                    logger.warning("[DEGRADE] MySQL 不可达，降级到 SQLite 创建用户")
                    if self._try_downgrade_to_sqlite():
                        return self._create_user_in_sqlite(username, password_hash, role, email, is_superuser)
                return None
            finally:
                db.close()

        # SQLite 模式
        if self.storage_mode == self.STORAGE_MODE_SQLITE:
            return self._create_user_in_sqlite(username, password_hash, role, email, is_superuser)

        # 内存模式
        if username in self.users:
            logger.warning(f"用户名已存在: {username}")
            return None
        self.users[username] = {
            "password": password_hash,
            "role": role,
            "email": email,
        }
        logger.info(f"内存模式创建用户成功: {username}")
        return {"id": username, "username": username, "role": role, "email": email, "is_superuser": is_superuser}

    def _create_user_in_sqlite(self, username: str, password_hash: str, role: str,
                                email: str, is_superuser: bool) -> Optional[Dict[str, Any]]:
        """在 SQLite 中创建用户"""
        if not self.sqlite_store:
            logger.error("SQLite 存储不可用")
            return None
        try:
            return self.sqlite_store.create_user(
                username=username,
                password_hash=password_hash,
                role=role,
                email=email,
                is_superuser=is_superuser,
            )
        except Exception as e:
            logger.error(f"SQLite 创建用户失败: {e}")
            # 最终回退到内存
            self._downgrade_to_memory()
            if username not in self.users:
                self.users[username] = {
                    "password": password_hash,
                    "role": role,
                    "email": email,
                }
                logger.info(f"[DEGRADE->memory] 创建用户: {username}")
                return {"id": username, "username": username, "role": role, "email": email, "is_superuser": is_superuser}
            return None

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        user_data = self._get_user_from_db(username)
        if user_data:
            # 清洗掉内部字段
            return {k: v for k, v in user_data.items() if not k.startswith("_")}

        mem_user = self.users.get(username)
        if mem_user:
            return {**mem_user, "username": username}
        return None

    def list_users(self, page: int = 1, page_size: int = 10) -> List[Dict[str, Any]]:
        """列出用户"""
        # MySQL 模式
        if self.storage_mode == self.STORAGE_MODE_MYSQL:
            try:
                db = next(get_db())
                offset = (page - 1) * page_size
                users = db.query(UserModel).order_by(UserModel.created_at.desc()).offset(offset).limit(page_size).all()
                return [user.to_dict() for user in users]
            except Exception as e:
                logger.error(f"MySQL 列出用户失败: {e}")
                return []
            finally:
                db.close()

        # SQLite 模式
        if self.storage_mode == self.STORAGE_MODE_SQLITE and self.sqlite_store:
            return self.sqlite_store.list_users()

        # 内存模式
        return [{"username": k, **v} for k, v in self.users.items()]

    def update_user(self, username: str, **kwargs) -> Optional[Dict[str, Any]]:
        """更新用户信息"""
        # MySQL 模式
        if self.storage_mode == self.STORAGE_MODE_MYSQL:
            try:
                db = next(get_db())
                user = db.query(UserModel).filter_by(username=username).first()
                if not user:
                    return None
                if "password" in kwargs:
                    user.set_password(kwargs.pop("password"))
                if "role" in kwargs:
                    user.role = kwargs["role"]
                    user.is_superuser = (kwargs["role"] == "admin")
                for key, value in kwargs.items():
                    if hasattr(user, key):
                        setattr(user, key, value)
                user.updated_at = datetime.now(timezone.utc)
                db.commit()
                db.refresh(user)
                return user.to_dict()
            except Exception as e:
                db.rollback()
                logger.error(f"更新用户失败: {e}")
                return None
            finally:
                db.close()

        # SQLite 模式 — 不支持复杂更新，仅支持密码和角色
        if self.storage_mode == self.STORAGE_MODE_SQLITE and self.sqlite_store:
            logger.warning("SQLite 模式暂不支持 update_user")

        # 内存模式
        if username in self.users:
            if "password" in kwargs:
                self.users[username]["password"] = self._hash_password(kwargs["password"])
            if "role" in kwargs:
                self.users[username]["role"] = kwargs["role"]
            return {"username": username, **self.users[username]}
        return None

    def delete_user(self, username: str) -> bool:
        """删除用户"""
        # MySQL 模式
        if self.storage_mode == self.STORAGE_MODE_MYSQL:
            try:
                db = next(get_db())
                user = db.query(UserModel).filter_by(username=username).first()
                if not user:
                    return False
                db.delete(user)
                db.commit()
                logger.info(f"删除用户成功: {username}")
                return True
            except Exception as e:
                db.rollback()
                logger.error(f"删除用户失败: {e}")
                return False
            finally:
                db.close()

        # SQLite 模式
        if self.storage_mode == self.STORAGE_MODE_SQLITE and self.sqlite_store:
            logger.warning("SQLite 模式暂不支持 delete_user")

        # 内存模式
        if username in self.users:
            del self.users[username]
            logger.info(f"内存模式删除用户: {username}")
            return True
        return False


# 全局认证服务实例
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """获取认证服务实例"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service


def init_auth_service():
    """初始化认证服务"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
        logger.info("认证服务初始化完成")
    return _auth_service


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌"""
    return get_auth_service().create_access_token(data, expires_delta)


def create_refresh_token(data: Dict[str, Any]) -> str:
    """创建刷新令牌"""
    return get_auth_service().create_refresh_token(data)


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """验证令牌"""
    return get_auth_service().verify_token(token)


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """验证用户"""
    return get_auth_service().authenticate_user(username, password)


def get_user_role(username: str) -> Optional[str]:
    """获取用户角色"""
    return get_auth_service().get_user_role(username)


def is_admin(username: str) -> bool:
    """检查是否为管理员"""
    return get_auth_service().is_admin(username)


def is_user(username: str) -> bool:
    """检查是否为普通用户"""
    return get_auth_service().is_user(username)


def create_user(username: str, password: str, role: str = "user", email: str = None) -> Optional[Dict[str, Any]]:
    """创建新用户"""
    return get_auth_service().create_user(username, password, role, email)


def get_user(username: str) -> Optional[Dict[str, Any]]:
    """获取用户信息"""
    return get_auth_service().get_user(username)


def update_user(username: str, **kwargs) -> Optional[Dict[str, Any]]:
    """更新用户信息"""
    return get_auth_service().update_user(username, **kwargs)


def delete_user(username: str) -> bool:
    """删除用户"""
    return get_auth_service().delete_user(username)


def list_users(page: int = 1, page_size: int = 10) -> List[Dict[str, Any]]:
    """列出用户"""
    return get_auth_service().list_users(page, page_size)
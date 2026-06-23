#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认证服务 - 支持数据库存储用户信息
"""

import os
import time
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from datetime import datetime

from src.core.logging.global_logger import get_logger

logger = get_logger("auth_service")

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
    logger.warning("bcrypt 模块不可用，使用简单密码验证")

try:
    from src.core.config.database import get_db, init_database, create_tables
    from src.models.database_models import UserModel
    HAS_DATABASE = True
except ImportError as e:
    HAS_DATABASE = False
    logger.warning(f"数据库模块不可用: {e}")


class AuthService:
    """认证服务 - 支持数据库存储"""

    _instance: Optional["AuthService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "initialized") and self.initialized:
            return

        self.initialized = True

        # 配置
        self.SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-here-for-development")
        self.ALGORITHM = os.environ.get("ALGORITHM", "HS256")
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        self.REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

        # 数据库模式标志
        self.use_database = HAS_DATABASE

        # 初始化数据库
        if self.use_database:
            try:
                init_database()
                create_tables()
                self._ensure_default_users()
                logger.info("认证服务初始化完成（数据库模式）")
            except Exception as e:
                logger.warning(f"数据库初始化失败，回退到内存模式: {e}")
                self.use_database = False

        # 如果不使用数据库，初始化内存用户
        if not self.use_database:
            # 使用简单密码存储（不使用bcrypt）
            self.users = {
                "admin": {"password": "admin123", "role": "admin"},
                "user": {"password": "user123", "role": "user"},
            }
            logger.info("认证服务初始化完成（内存模式）")

    def _ensure_default_users(self):
        """确保默认用户存在"""
        if not HAS_DATABASE or not HAS_BCRYPT:
            return

        default_users = [
            {"username": "admin", "password": "admin123", "role": "admin", "is_superuser": True},
            {"username": "user", "password": "user123", "role": "user", "is_superuser": False},
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

    def _get_user_from_db(self, username: str) -> Optional[UserModel]:
        """从数据库获取用户"""
        if not self.use_database:
            return None

        try:
            db = next(get_db())
            user = db.query(UserModel).filter_by(username=username, is_active=True).first()
            db.close()
            return user
        except Exception as e:
            logger.error(f"从数据库获取用户失败: {e}")
            return None

    def _update_user_login_info(self, user: UserModel, success: bool = True):
        """更新用户登录信息"""
        if not self.use_database or not user:
            return

        try:
            db = next(get_db())
            db_user = db.query(UserModel).filter_by(id=user.id).first()
            if db_user:
                if success:
                    db_user.last_login_at = datetime.utcnow()
                    db_user.login_count = (db_user.login_count or 0) + 1
                    db_user.failed_login_count = 0
                    db_user.locked_until = None
                else:
                    db_user.failed_login_count = (db_user.failed_login_count or 0) + 1
                    # 连续失败5次锁定10分钟
                    if db_user.failed_login_count >= 5:
                        db_user.locked_until = datetime.utcnow() + timedelta(minutes=10)
                        db_user.is_active = False
                db.commit()
        except Exception as e:
            logger.error(f"更新用户登录信息失败: {e}")
        finally:
            db.close()

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问令牌"""
        if not HAS_JWT:
            logger.warning("jwt 模块不可用，返回简单令牌")
            return f"simple_token_{secrets.token_hex(16)}"

        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
        return encoded_jwt

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """创建刷新令牌"""
        if not HAS_JWT:
            logger.warning("jwt 模块不可用，返回简单令牌")
            return f"simple_refresh_{secrets.token_hex(16)}"

        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)
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

            user = self._get_user_from_db(username)
            if not user:
                # 回退到内存模式
                if not self.use_database:
                    user_data = self.users.get(username)
                    if not user_data:
                        logger.warning(f"用户不存在: {username}")
                        return None
                    user_role = user_data.get("role")
                else:
                    logger.warning(f"用户不存在: {username}")
                    return None
            else:
                user_role = user.role

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
        """验证用户"""
        # 先尝试数据库模式
        if self.use_database and HAS_BCRYPT:
            user = self._get_user_from_db(username)
            if user:
                # 检查用户是否被锁定
                if user.locked_until and user.locked_until > datetime.utcnow():
                    remaining = (user.locked_until - datetime.utcnow()).total_seconds() // 60
                    logger.warning(f"用户 {username} 已被锁定，剩余 {remaining} 分钟")
                    return None

                if user.verify_password(password):
                    self._update_user_login_info(user, success=True)
                    return {
                        "id": user.id,
                        "username": user.username,
                        "role": user.role,
                        "is_superuser": user.is_superuser,
                    }
                else:
                    self._update_user_login_info(user, success=False)
                    return None

        # 回退到内存模式（简单密码验证）
        user_data = self.users.get(username)
        if not user_data:
            return None

        # 简单密码比对（不使用bcrypt）
        stored_password = user_data.get("password")
        if stored_password == password:
            return {
                "id": username,
                "username": username,
                "role": user_data.get("role"),
                "is_superuser": user_data.get("role") == "admin",
            }

        return None

    def get_user_role(self, username: str) -> Optional[str]:
        """获取用户角色"""
        if self.use_database:
            user = self._get_user_from_db(username)
            return user.role if user else None

        user_data = self.users.get(username)
        return user_data.get("role") if user_data else None

    def is_admin(self, username: str) -> bool:
        """检查是否为管理员"""
        role = self.get_user_role(username)
        return role == "admin"

    def is_user(self, username: str) -> bool:
        """检查是否为普通用户"""
        role = self.get_user_role(username)
        return role == "user"

    def create_user(self, username: str, password: str, role: str = "user", email: str = None) -> Optional[Dict[str, Any]]:
        """创建新用户"""
        if not self.use_database:
            logger.warning("当前为内存模式，无法创建用户")
            return None

        try:
            db = next(get_db())
            # 检查用户名是否已存在
            existing_user = db.query(UserModel).filter_by(username=username).first()
            if existing_user:
                logger.warning(f"用户名已存在: {username}")
                return None

            new_user = UserModel(
                username=username,
                role=role,
                email=email,
                is_active=True,
                is_superuser=(role == "admin")
            )
            new_user.set_password(password)
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            logger.info(f"创建用户成功: {username}")
            return new_user.to_dict()
        except Exception as e:
            db.rollback()
            logger.error(f"创建用户失败: {e}")
            return None
        finally:
            db.close()

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        if self.use_database:
            user = self._get_user_from_db(username)
            return user.to_dict() if user else None

        user_data = self.users.get(username)
        return user_data if user_data else None

    def update_user(self, username: str, **kwargs) -> Optional[Dict[str, Any]]:
        """更新用户信息"""
        if not self.use_database:
            logger.warning("当前为内存模式，无法更新用户")
            return None

        try:
            db = next(get_db())
            user = db.query(UserModel).filter_by(username=username).first()
            if not user:
                return None

            # 更新字段
            if "password" in kwargs:
                user.set_password(kwargs.pop("password"))
            if "role" in kwargs:
                user.role = kwargs["role"]
                user.is_superuser = (kwargs["role"] == "admin")
            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)

            user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(user)
            return user.to_dict()
        except Exception as e:
            db.rollback()
            logger.error(f"更新用户失败: {e}")
            return None
        finally:
            db.close()

    def delete_user(self, username: str) -> bool:
        """删除用户"""
        if not self.use_database:
            logger.warning("当前为内存模式，无法删除用户")
            return False

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

    def list_users(self, page: int = 1, page_size: int = 10) -> List[Dict[str, Any]]:
        """列出用户"""
        if not self.use_database:
            return [{"username": k, **v} for k, v in self.users.items()]

        try:
            db = next(get_db())
            offset = (page - 1) * page_size
            users = db.query(UserModel).order_by(UserModel.created_at.desc()).offset(offset).limit(page_size).all()
            return [user.to_dict() for user in users]
        except Exception as e:
            logger.error(f"获取用户列表失败: {e}")
            return []
        finally:
            db.close()


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
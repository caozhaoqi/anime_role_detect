import os
import jwt
import time
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from src.core.logging.global_logger import get_logger

logger = get_logger("auth_service")


class AuthService:
    """认证服务"""

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

        # 模拟用户数据库
        self.users = {
            "admin": {"password": "admin123", "role": "admin"},  # 实际应用中应该使用哈希密码
            "user": {"password": "user123", "role": "user"},
        }

        logger.info("认证服务初始化完成")

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问令牌"""
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
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
        return encoded_jwt

    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """
        使用刷新令牌获取新的访问令牌

        Args:
            refresh_token: 刷新令牌

        Returns:
            Optional[Dict[str, Any]]: 包含新访问令牌的字典，或None
        """
        try:
            payload = jwt.decode(refresh_token, self.SECRET_KEY, algorithms=[self.ALGORITHM])

            if payload.get("type") != "refresh":
                logger.warning("无效的刷新令牌类型")
                return None

            username = payload.get("sub")
            if not username:
                logger.warning("刷新令牌缺少用户信息")
                return None

            user = self.users.get(username)
            if not user:
                logger.warning(f"用户不存在: {username}")
                return None

            new_access_token = self.create_access_token({"sub": username, "role": user.get("role")})

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
        user = self.users.get(username)
        if not user:
            return None
        if user.get("password") != password:
            return None
        return user

    def get_user_role(self, username: str) -> Optional[str]:
        """获取用户角色"""
        user = self.users.get(username)
        if not user:
            return None
        return user.get("role")

    def is_admin(self, username: str) -> bool:
        """检查是否为管理员"""
        role = self.get_user_role(username)
        return role == "admin"

    def is_user(self, username: str) -> bool:
        """检查是否为普通用户"""
        role = self.get_user_role(username)
        return role == "user"


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

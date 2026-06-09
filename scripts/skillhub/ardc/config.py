#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理模块
使用 pydantic-settings 实现类型安全的配置管理
支持环境变量、.env 文件和命令行参数
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
from pathlib import Path


class DatabaseSettings(BaseSettings):
    """数据库配置"""

    url: str = "sqlite:///./data/ardc.db"
    pool_size: int = 20
    max_overflow: int = 50
    pool_timeout: int = 30
    pool_recycle: int = 1800  # 30分钟
    echo_sql: bool = False


class JWTSettings(BaseSettings):
    """JWT 配置"""

    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # 密钥轮换支持 - 允许使用多个密钥进行验证
    additional_secret_keys: List[str] = []


class CorsSettings(BaseSettings):
    """CORS 配置"""

    allowed_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    allow_credentials: bool = True
    allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers: List[str] = ["*"]


class LogSettings(BaseSettings):
    """日志配置"""

    level: str = "INFO"
    dir: str = "logs"
    format: str = (
        "%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - [Req:%(request_id)s] %(message)s"
    )
    json_format: bool = False
    max_file_size_mb: int = 100
    backup_count: int = 5


class RedisSettings(BaseSettings):
    """Redis 配置"""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    prefix: str = "ardc:"
    cache_ttl_seconds: int = 3600  # 默认缓存1小时


class SecuritySettings(BaseSettings):
    """安全配置"""

    cookie_secure: bool = False  # 生产环境应设为 True
    cookie_samesite: str = "lax"
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60


class ServerSettings(BaseSettings):
    """服务器配置"""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1


class SkillSettings(BaseSettings):
    """技能配置"""

    registry_path: str = str(Path.home() / ".ardc" / "registry.json")
    skills_dir: str = str(Path.home() / ".ardc" / "skills")
    index_path: str = str(Path.home() / ".ardc" / "skill_index.json")
    versions_path: str = str(Path.home() / ".ardc" / "versions")


class Settings(BaseSettings):
    """综合配置类"""

    # 子配置
    database: DatabaseSettings = DatabaseSettings()
    jwt: JWTSettings
    cors: CorsSettings = CorsSettings()
    log: LogSettings = LogSettings()
    redis: RedisSettings = RedisSettings()
    security: SecuritySettings = SecuritySettings()
    server: ServerSettings = ServerSettings()
    skill: SkillSettings = SkillSettings()

    # API 配置
    api_title: str = "ARD Skill Repository API"
    api_version: str = "2.0.0"
    api_description: str = "技能仓库 RESTful API - 提供技能管理、用户认证、技能搜索等功能"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__"
    )


# 创建全局配置实例
settings = Settings()


# 验证关键配置
def validate_settings():
    """验证配置的有效性"""
    if not settings.jwt.secret_key:
        raise RuntimeError("JWT_SECRET_KEY 环境变量必须设置")

    # 验证数据库 URL
    if not settings.database.url:
        raise RuntimeError("DATABASE_URL 环境变量必须设置")

    # 如果使用 HTTPS，强制使用安全 Cookie
    if settings.security.cookie_secure:
        # 可以添加更多安全检查

        pass

    return True


# 预验证配置
validate_settings()

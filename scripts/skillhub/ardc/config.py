#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理模块
使用 pydantic-settings 实现类型安全的配置管理
支持环境变量、.env 文件和命令行参数
"""

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
from pathlib import Path


class DatabaseSettings(BaseModel):
    """数据库配置"""

    url: str = "sqlite:///./data/ardc.db"
    pool_size: int = 20
    max_overflow: int = 50
    pool_timeout: int = 30
    pool_recycle: int = 1800  # 30分钟
    echo_sql: bool = False


class JWTSettings(BaseModel):
    """JWT 配置"""

    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # 密钥轮换支持 - 允许使用多个密钥进行验证
    additional_secret_keys: List[str] = []


class CorsSettings(BaseModel):
    """CORS 配置"""

    allowed_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    allow_credentials: bool = True
    allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers: List[str] = ["*"]


class LogSettings(BaseModel):
    """日志配置"""

    level: str = "INFO"
    dir: str = "logs"
    format: str = (
        "%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - [Req:%(request_id)s] %(message)s"
    )
    json_format: bool = False
    max_file_size_mb: int = 100
    backup_count: int = 5


class RedisSettings(BaseModel):
    """Redis 配置"""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    prefix: str = "ardc:"
    cache_ttl_seconds: int = 3600  # 默认缓存1小时


class SecuritySettings(BaseModel):
    """安全配置"""

    cookie_secure: bool = False  # 生产环境应设为 True
    cookie_samesite: str = "lax"
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60


class ServerSettings(BaseModel):
    """服务器配置"""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1


class SkillSettings(BaseModel):
    """技能配置"""

    registry_path: str = str(Path.home() / ".ardc" / "registry.json")
    skills_dir: str = str(Path.home() / ".ardc" / "skills")
    index_path: str = str(Path.home() / ".ardc" / "skill_index.json")
    versions_path: str = str(Path.home() / ".ardc" / "versions")


class Settings(BaseSettings):
    """综合配置类"""

    # API 配置
    api_title: str = "ARD Skill Repository API"
    api_version: str = "2.0.0"
    api_description: str = "技能仓库 RESTful API - 提供技能管理、用户认证、技能搜索等功能"

    # 数据库配置
    database_url: str = "sqlite:///./data/ardc.db"
    database_pool_size: int = 20
    database_max_overflow: int = 50
    database_pool_timeout: int = 30
    database_pool_recycle: int = 1800
    database_echo_sql: bool = False

    # JWT 配置
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # CORS 配置
    cors_allowed_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: List[str] = ["*"]

    # 日志配置
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_json_format: bool = False
    log_max_file_size_mb: int = 100
    log_backup_count: int = 5

    # Redis 配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_prefix: str = "ardc:"
    redis_cache_ttl_seconds: int = 3600

    # 安全配置
    security_cookie_secure: bool = False
    security_cookie_samesite: str = "lax"
    security_rate_limit_requests: int = 100
    security_rate_limit_window_seconds: int = 60

    # 服务器配置
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    server_reload: bool = False
    server_workers: int = 1

    # 技能配置
    skill_registry_path: str = str(Path.home() / ".ardc" / "registry.json")
    skill_skills_dir: str = str(Path.home() / ".ardc" / "skills")
    skill_index_path: str = str(Path.home() / ".ardc" / "skill_index.json")
    skill_versions_path: str = str(Path.home() / ".ardc" / "versions")

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__"
    )

    @property
    def database(self) -> DatabaseSettings:
        return DatabaseSettings(
            url=self.database_url,
            pool_size=self.database_pool_size,
            max_overflow=self.database_max_overflow,
            pool_timeout=self.database_pool_timeout,
            pool_recycle=self.database_pool_recycle,
            echo_sql=self.database_echo_sql
        )

    @property
    def jwt(self) -> JWTSettings:
        return JWTSettings(
            secret_key=self.jwt_secret_key,
            algorithm=self.jwt_algorithm,
            access_token_expire_minutes=self.jwt_access_token_expire_minutes,
            refresh_token_expire_days=self.jwt_refresh_token_expire_days
        )

    @property
    def cors(self) -> CorsSettings:
        return CorsSettings(
            allowed_origins=self.cors_allowed_origins,
            allow_credentials=self.cors_allow_credentials,
            allow_methods=self.cors_allow_methods,
            allow_headers=self.cors_allow_headers
        )

    @property
    def log(self) -> LogSettings:
        return LogSettings(
            level=self.log_level,
            dir=self.log_dir,
            json_format=self.log_json_format,
            max_file_size_mb=self.log_max_file_size_mb,
            backup_count=self.log_backup_count
        )

    @property
    def redis(self) -> RedisSettings:
        return RedisSettings(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            ssl=self.redis_ssl,
            prefix=self.redis_prefix,
            cache_ttl_seconds=self.redis_cache_ttl_seconds
        )

    @property
    def security(self) -> SecuritySettings:
        return SecuritySettings(
            cookie_secure=self.security_cookie_secure,
            cookie_samesite=self.security_cookie_samesite,
            rate_limit_requests=self.security_rate_limit_requests,
            rate_limit_window_seconds=self.security_rate_limit_window_seconds
        )

    @property
    def server(self) -> ServerSettings:
        return ServerSettings(
            host=self.server_host,
            port=self.server_port,
            reload=self.server_reload,
            workers=self.server_workers
        )

    @property
    def skill(self) -> SkillSettings:
        return SkillSettings(
            registry_path=self.skill_registry_path,
            skills_dir=self.skill_skills_dir,
            index_path=self.skill_index_path,
            versions_path=self.skill_versions_path
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

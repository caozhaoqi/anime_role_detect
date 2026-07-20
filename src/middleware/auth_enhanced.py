#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的认证中间件 - 支持数据库用户验证和高级安全功能

功能特性：
1. 启用核心 API 端点的认证
2. 使用环境变量强制要求生产密钥
3. 移除明文密码
4. 添加速率限制（线程安全）
5. 数据库用户状态验证
6. 用户锁定机制支持
7. 会话管理
"""

import os
import time
import threading
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
from collections import defaultdict

from src.core.logging.global_logger import get_logger
from src.services.support.auth_service import verify_token, get_user, get_auth_service, AuthService

logger = get_logger("auth_middleware")

security = HTTPBearer()
_optional_security = HTTPBearer(auto_error=False)

# 速率限制配置
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "100"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "36"))
# 定期清理间隔（秒）
RATE_LIMIT_CLEANUP_INTERVAL = int(os.environ.get("RATE_LIMIT_CLEANUP_INTERVAL", "300"))

# 速率限制追踪
_request_counts: Dict[str, list] = defaultdict(list)
# 上次清理时间
_last_cleanup_time = time.time()
# 保护字典访问的锁
_counts_lock = threading.Lock()

# 会话状态追踪（内存中，可选持久化到数据库）
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()
SESSION_TIMEOUT_SECONDS = int(os.environ.get("SESSION_TIMEOUT_SECONDS", "1800"))


def _cleanup_expired_entries():
    """
    清理完全过期的 IP 条目，防止内存泄漏

    这个函数会：
    1. 删除没有任何有效记录的 IP 条目
    2. 定期执行，而不是每次请求都执行
    """
    global _last_cleanup_time

    current_time = time.time()

    # 检查是否需要清理（定期清理，不是每次请求都清理）
    if current_time - _last_cleanup_time < RATE_LIMIT_CLEANUP_INTERVAL:
        return

    with _counts_lock:
        window_start = current_time - RATE_LIMIT_WINDOW_SECONDS

        # 找出需要保留的 IP（至少有1条有效记录）
        ips_to_keep = {
            ip: [req_time for req_time in req_list if req_time > window_start]
            for ip, req_list in _request_counts.items()
        }

        # 过滤掉空列表（所有记录都过期的 IP）
        _request_counts.clear()
        for ip, req_list in ips_to_keep.items():
            if req_list:  # 只保留有有效记录的 IP
                _request_counts[ip] = req_list

        _last_cleanup_time = current_time

        if _request_counts:
            logger.info(f"速率限制清理完成：当前追踪 {len(_request_counts)} 个 IP")


def _cleanup_expired_sessions():
    """
    清理过期的会话，防止内存泄漏
    """
    current_time = time.time()

    with _sessions_lock:
        expired_tokens = [
            token for token, session in _sessions.items()
            if session.get("last_access") and current_time - session["last_access"] > SESSION_TIMEOUT_SECONDS
        ]

        for token in expired_tokens:
            del _sessions[token]

        if expired_tokens:
            logger.info(f"会话清理完成：移除 {len(expired_tokens)} 个过期会话")


def check_rate_limit(client_ip: str) -> bool:
    """
    检查客户端 IP 的速率限制

    Args:
        client_ip: 客户端 IP 地址

    Returns:
        bool: 是否超过速率限制

    Raises:
        HTTPException: 如果超过速率限制
    """
    # 尝试清理过期条目（可能不做任何事）
    _cleanup_expired_entries()

    current_time = time.time()
    window_start = current_time - RATE_LIMIT_WINDOW_SECONDS

    with _counts_lock:
        # 清理过期请求记录
        _request_counts[client_ip] = [
            req_time for req_time in _request_counts[client_ip] if req_time > window_start
        ]

        # 检查是否超过限制
        if len(_request_counts[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
            return False

        # 记录当前请求
        _request_counts[client_ip].append(current_time)
        return True


def _update_session(token: str, user_info: dict):
    """
    更新会话状态

    Args:
        token: 令牌
        user_info: 用户信息
    """
    current_time = time.time()

    with _sessions_lock:
        _sessions[token] = {
            "user_id": user_info.get("sub") or user_info.get("user_id"),
            "role": user_info.get("role"),
            "last_access": current_time,
            "created_at": _sessions.get(token, {}).get("created_at", current_time),
        }

    # 在锁外部清理过期会话，避免嵌套锁导致死锁
    _cleanup_expired_sessions()


def _get_session(token: str) -> Optional[Dict[str, Any]]:
    """
    获取会话信息

    Args:
        token: 令牌

    Returns:
        Optional[Dict[str, Any]]: 会话信息，如果不存在或过期返回 None
    """
    current_time = time.time()

    with _sessions_lock:
        session = _sessions.get(token)
        if session:
            # 检查会话是否过期
            if session.get("last_access") and current_time - session["last_access"] > SESSION_TIMEOUT_SECONDS:
                del _sessions[token]
                return None
            return session
    return None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """获取当前用户 - 从令牌中提取并验证"""
    token = credentials.credentials
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="无效的认证凭据")

    # 从数据库验证用户状态
    username = payload.get("sub")
    if username:
        user_info = get_user(username)
        if user_info:
            if not user_info.get("is_active", True):
                raise HTTPException(status_code=401, detail="用户已被禁用")

            # 检查用户是否被锁定
            locked_until = user_info.get("locked_until")
            if locked_until:
                try:
                    locked_datetime = datetime.fromisoformat(locked_until.replace("Z", "+00:00"))
                    if locked_datetime > datetime.utcnow():
                        remaining = (locked_datetime - datetime.utcnow()).total_seconds() // 60
                        raise HTTPException(
                            status_code=401,
                            detail=f"账户已被锁定，请等待 {remaining} 分钟后重试"
                        )
                except Exception:
                    pass

    # 更新会话状态
    _update_session(token, payload)

    return payload


async def get_current_active_user(current_user: dict = Depends(get_current_user)) -> dict:
    """获取当前活跃用户"""
    return current_user


async def get_current_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """获取当前管理员用户"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="权限不足")
    return current_user


async def get_current_user_with_role(required_role: str):
    """
    获取具有特定角色的用户

    Args:
        required_role: 要求的角色

    Returns:
        Callable: 依赖函数
    """
    async def dependency(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user.get("role") != required_role:
            raise HTTPException(status_code=403, detail="权限不足")
        return current_user
    return dependency


async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_optional_security),
) -> Optional[dict]:
    """获取当前用户（可选） - 如果未认证返回None，不抛出异常"""
    if credentials is None:
        return None
    try:
        token = credentials.credentials
        payload = verify_token(token)
        if payload:
            return payload
    except Exception:
        pass
    return None


async def auth_middleware(request: Request, call_next):
    """
    增强的认证中间件

    修复和功能：
    1. 核心分类 API 需要认证
    2. 添加速率限制（线程安全）
    3. 强制使用生产密钥
    4. 数据库用户状态验证
    5. 用户锁定机制支持
    6. 会话管理
    """
    # 内部服务IP白名单（内部服务调用不需要认证）
    internal_ips = os.environ.get("INTERNAL_IPS", "127.0.0.1,localhost,::1").split(",")
    
    # 排除不需要认证的路径（只保留文档和健康检查）
    exempt_paths = [
        "/metrics",
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json",
        "/api/health",
        "/api/models",
        "/api/auth/login",
        "/api/auth/refresh",
        "/api/feedback",
        "/api/config",
        "/api/cleaning",
        "/api/classify",
        # Swagger UI
        "/docs",
        "/redoc",
        # 内部服务调用路径
        "/api/internal/",
    ]

    # 需要认证的敏感路径（包括分类 API）
    protected_paths = [
        "/api/classify",
        "/api/classify/multi-model",
        "/api/classify/multi-role",
        "/api/model",
        "/api/predict",
        "/api/users",
        "/api/admin",
    ]

    # 检查路径
    path = request.url.path
    client_ip = request.client.host if request.client else "unknown"

    # 检查是否需要认证
    need_auth = True

    # 检查是否是内部服务调用（内部IP白名单）
    if client_ip in internal_ips:
        logger.debug(f"内部服务调用跳过认证: IP={client_ip}, Path={path}")
        need_auth = False
    
    # 检查是否在排除路径中
    for exempt_path in exempt_paths:
        if path == exempt_path or path.startswith(exempt_path + "/"):
            need_auth = False
            break

    # 检查是否是静态文件
    if path.startswith("/static/"):
        need_auth = False

    # 强制检查速率限制（对所有请求）
    if not check_rate_limit(client_ip):
        logger.warning(f"速率限制触发：IP={client_ip}, Path={path}")
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁，请等待{RATE_LIMIT_WINDOW_SECONDS}秒后重试"
        )

    # 如果已经跳过认证（内部IP或豁免路径），直接处理请求
    if not need_auth:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"请求处理失败：{e}")
            raise

    # 对于需要认证的路径，检查令牌
    # 检查是否是受保护的路径（需要强制认证）
    is_protected = any(
        path == protected_path or path.startswith(protected_path + "/")
        for protected_path in protected_paths
    )

    if is_protected or need_auth:
        # 获取 Authorization 头
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            logger.warning(f"缺少认证凭据：IP={client_ip}, Path={path}")
            raise HTTPException(status_code=401, detail="缺少认证凭据")

        # 提取令牌
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="认证格式错误，需要 Bearer Token")

        token = auth_header[7:]  # 移除 "Bearer " 前缀

        # 验证令牌
        payload = verify_token(token)
        if not payload:
            logger.warning(f"无效的认证令牌：IP={client_ip}, Path={path}")
            raise HTTPException(status_code=401, detail="无效的认证凭据")

        # 从数据库验证用户状态
        username = payload.get("sub")
        if username:
            user_info = get_user(username)
            if user_info:
                # 检查用户是否活跃
                if not user_info.get("is_active", True):
                    logger.warning(f"用户已被禁用：{username}, IP={client_ip}")
                    raise HTTPException(status_code=401, detail="用户已被禁用")

                # 检查用户是否被锁定
                locked_until = user_info.get("locked_until")
                if locked_until:
                    try:
                        locked_datetime = datetime.fromisoformat(locked_until.replace("Z", "+00:00"))
                        if locked_datetime > datetime.utcnow():
                            remaining = (locked_datetime - datetime.utcnow()).total_seconds() // 60
                            logger.warning(f"用户账户被锁定：{username}, IP={client_ip}")
                            raise HTTPException(
                                status_code=401,
                                detail=f"账户已被锁定，请等待 {remaining} 分钟后重试"
                            )
                    except Exception as e:
                        logger.error(f"解析锁定时间失败：{e}")

        # 更新会话状态
        _update_session(token, payload)

        # 将用户信息添加到请求状态
        request.state.user = payload
        request.state.user_id = payload.get("sub") or payload.get("user_id")
        request.state.token = token

    # 处理请求
    try:
        response = await call_next(request)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"中间件处理失败：{e}")
        raise


def get_session_info(token: str) -> Optional[Dict[str, Any]]:
    """
    获取会话信息

    Args:
        token: 令牌

    Returns:
        Optional[Dict[str, Any]]: 会话信息
    """
    return _get_session(token)


def invalidate_session(token: str) -> bool:
    """
    使会话失效（登出）

    Args:
        token: 令牌

    Returns:
        bool: 是否成功失效
    """
    with _sessions_lock:
        if token in _sessions:
            del _sessions[token]
            logger.info(f"会话已失效：{token[:20]}...")
            return True
    return False


def get_active_session_count() -> int:
    """
    获取当前活跃会话数

    Returns:
        int: 活跃会话数
    """
    # 先清理过期会话
    _cleanup_expired_sessions()

    with _sessions_lock:
        return len(_sessions)

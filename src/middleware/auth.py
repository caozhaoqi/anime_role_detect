import os
import time
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict
from collections import defaultdict

from src.core.logging.global_logger import get_logger
from src.services.auth_service import verify_token

logger = get_logger("auth_middleware")

security = HTTPBearer()

# 速率限制配置
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "100"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "3600"))

# 速率限制追踪
_request_counts: Dict[str, list] = defaultdict(list)


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
    current_time = time.time()
    window_start = current_time - RATE_LIMIT_WINDOW_SECONDS

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


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """获取当前用户"""
    token = credentials.credentials
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="无效的认证凭据")
    return payload


async def get_current_active_user(current_user: dict = Depends(get_current_user)) -> dict:
    """获取当前活跃用户"""
    return current_user


async def get_current_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """获取当前管理员用户"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="权限不足")
    return current_user


async def auth_middleware(request: Request, call_next):
    """
    增强的认证中间件

    修复：
    1. 核心分类 API 需要认证
    2. 添加速率限制
    3. 强制使用生产密钥
    """
    # 排除不需要认证的路径（只保留文档和健康检查）
    exempt_paths = [
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json",
        "/api/health",
        "/api/models",
        "/api/auth/login",
        "/api/auth/refresh",
        "/api/feedback",
        "/api/config",
        "/api/history",
        # ONNX 推理 API（如果有独立的认证）
        "/api/v1/onnx",
        # Swagger UI
        "/docs",
        "/redoc",
    ]

    # 需要认证的敏感路径（包括分类 API）
    protected_paths = [
        "/api/classify",
        "/api/classify/multi-model",
        "/api/classify/multi-role",
        "/api/model",
        "/api/predict",
    ]

    # 检查路径
    path = request.url.path
    client_ip = request.client.host if request.client else "unknown"

    # 检查是否需要认证
    need_auth = True

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
        logger.warning(f"速率限制触发：IP={client_ip}")
        raise HTTPException(
            status_code=429, detail=f"请求过于频繁，请等待{RATE_LIMIT_WINDOW_SECONDS}秒后重试"
        )

    # 如果是受保护的路径，强制要求认证
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
            logger.warning(f"无效的认证令牌：IP={client_ip}")
            raise HTTPException(status_code=401, detail="无效的认证凭据")

        # 将用户信息添加到请求状态
        request.state.user = payload
        request.state.user_id = payload.get("sub") or payload.get("user_id")

    # 处理请求
    try:
        response = await call_next(request)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"中间件处理失败：{e}")
        raise

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from src.core.logging.global_logger import get_logger
from src.services.auth_service import verify_token

logger = get_logger("auth_middleware")

security = HTTPBearer()

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
    """认证中间件"""
    # 排除不需要认证的路径
    exempt_paths = [
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json",
        "/api/health",
        "/api/models",
        "/api/auth/login",
        "/api/auth/refresh",
        # "/api/classify",
        # "/api/classify/multi-model",
        "/api/feedback",
        "/api/config",
        "/api/search",
        "/api/roles",
        "/api/history",
        # ONNX 推理 API
        "/api/v1/onnx",
        # Swagger UI
        "/docs",
        "/redoc"
    ]
    
    # 检查是否需要认证
    need_auth = True
    
    # 检查是否在排除路径中
    for path in exempt_paths:
        if request.url.path == path or request.url.path.startswith(path + "/"):
            need_auth = False
            break
    
    # 检查是否是静态文件
    if request.url.path.startswith("/static/"):
        need_auth = False
    
    if need_auth:
        # 获取Authorization头
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="缺少认证凭据")
        
        # 提取令牌
        token = auth_header.replace("Bearer ", "")
        payload = verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="无效的认证凭据")
        
        # 将用户信息添加到请求状态
        request.state.user = payload
    
    # 处理请求
    response = await call_next(request)
    return response

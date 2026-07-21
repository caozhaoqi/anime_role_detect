"""
认证路由
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

from fastapi import APIRouter, Form, Depends

from src.core.logging.global_logger import get_logger
from src.services.support.auth_service import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
)
from src.middleware.auth_enhanced import get_current_user, get_current_admin
from src.services.support.auth_service import get_auth_service

logger = get_logger("api.routes.auth")

router = APIRouter()


@router.post("/api/auth/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """用户登录"""
    try:
        user = authenticate_user(username, password)
        if not user:
            return {"success": False, "message": "用户名或密码错误"}
        access_token = create_access_token(data={"sub": username, "role": user.get("role")})
        refresh_token = create_refresh_token(data={"sub": username})
        return {
            "success": True,
            "message": "登录成功",
            "data": {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "username": username,
                "role": user.get("role"),
            },
        }
    except Exception as e:
        logger.error(f"登录失败: {e}")
        return {"success": False, "message": "登录失败，请稍后重试"}


@router.post("/api/auth/register")
async def register(username: str = Form(...), password: str = Form(...), email: str = Form(None)):
    """用户注册"""
    try:
        # 输入校验
        if not username or not password:
            return {"success": False, "message": "用户名和密码不能为空"}
        if len(username) < 2 or len(username) > 32:
            return {"success": False, "message": "用户名长度需要 2-32 个字符"}
        if len(password) < 4:
            return {"success": False, "message": "密码长度至少 4 位"}

        auth_service = get_auth_service()
        user = auth_service.create_user(username=username, password=password, email=email)

        if not user:
            return {"success": False, "message": "用户名已存在或注册失败"}

        # 注册成功后自动生成 token
        access_token = create_access_token(data={"sub": username, "role": user.get("role", "user")})
        refresh_token = create_refresh_token(data={"sub": username})

        logger.info(f"用户注册成功: {username}, 存储模式: {auth_service.storage_mode}")
        return {
            "success": True,
            "message": "注册成功",
            "data": {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "username": username,
                "role": user.get("role", "user"),
                "storage_mode": auth_service.storage_mode,
            },
        }
    except Exception as e:
        logger.error(f"注册失败: {e}")
        return {"success": False, "message": f"注册失败: {str(e)}"}


@router.post("/api/auth/refresh")
async def refresh_token(refresh_token: str = Form(...)):
    """刷新访问令牌"""
    try:
        from src.services.support.auth_service import get_auth_service
        auth_service = get_auth_service()
        result = auth_service.refresh_access_token(refresh_token)
        if result:
            return {"success": True, "message": "令牌刷新成功", "data": result}
        else:
            return {"success": False, "message": "无效或已过期的刷新令牌"}
    except Exception as e:
        logger.error(f"刷新令牌失败: {e}")
        return {"success": False, "message": "刷新令牌失败，请稍后重试"}


@router.get("/api/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """获取当前用户信息"""
    try:
        return {
            "success": True,
            "message": "获取用户信息成功",
            "data": {"username": current_user.get("sub"), "role": current_user.get("role")},
        }
    except Exception as e:
        logger.error(f"获取用户信息失败: {e}")
        return {"success": False, "message": "获取用户信息失败，请稍后重试"}


@router.get("/api/admin/test")
async def admin_test(current_admin: dict = Depends(get_current_admin)):
    """管理员测试端点"""
    try:
        return {
            "success": True,
            "message": "管理员访问成功",
            "data": {"username": current_admin.get("sub"), "role": current_admin.get("role")},
        }
    except Exception as e:
        logger.error(f"管理员测试失败: {e}")
        return {"success": False, "message": "管理员测试失败，请稍后重试"}

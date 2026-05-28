#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户认证模块
提供登录、注册、登出等功能
支持密钥轮换和安全 Cookie 配置
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List

from fastapi import APIRouter, HTTPException, Depends, Form, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from ardc.api.database import get_db, DBUser, TokenBlacklist, init_db
from ardc.utils.logging import get_logger
from ardc.config import settings

logger = get_logger(__name__)

# JWT 配置 - 从统一配置读取
JWT_SECRET_KEY = settings.jwt.secret_key
JWT_ALGORITHM = settings.jwt.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.jwt.access_token_expire_minutes

# 所有有效的密钥（支持密钥轮换）
ALL_SECRET_KEYS: List[str] = [JWT_SECRET_KEY] + settings.jwt.additional_secret_keys

# 密码上下文 - 使用 sha256_crypt（跨平台兼容性好，无密码长度限制）
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# OAuth2 Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# 路由
router = APIRouter(prefix="/api/auth")


def decode_jwt_token(token: str) -> Optional[Dict]:
    """解码 JWT Token，支持密钥轮换"""
    for secret_key in ALL_SECRET_KEYS:
        try:
            payload = jwt.decode(token, secret_key, algorithms=[JWT_ALGORITHM])
            return payload
        except JWTError:
            continue
    return None


class Token(BaseModel):
    access_token: str
    token_type: str


class User(BaseModel):
    id: str
    username: str
    email: str
    hashed_password: str
    is_developer: bool
    created_at: str
    updated_at: str

    class Config:
        orm_mode = True


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """生成密码哈希值 - 处理 bcrypt 的 72 字节限制"""
    # bcrypt 限制密码长度为 72 字节
    if len(password.encode('utf-8')) > 72:
        logger.warning(f"密码长度超过 72 字节，已自动截断")
        password = password[:72]
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "jti": str(uuid.uuid4())})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


async def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """获取当前用户 - 支持从 header 或 cookie 获取 token，支持密钥轮换"""
    credentials_exception = HTTPException(
        status_code=401,
        detail="无法验证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # 优先从 cookie 获取 token
    token = request.cookies.get("access_token")
    
    # 如果 cookie 中没有，从 Authorization header 获取
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
    
    if not token:
        raise credentials_exception
    
    # 使用支持密钥轮换的解码函数
    payload = decode_jwt_token(token)
    if payload is None:
        raise credentials_exception
    
    username: str = payload.get("sub")
    jti: str = payload.get("jti")
    if username is None:
        raise credentials_exception
    
    # 检查 token 是否在黑名单中
    blacklisted_token = db.query(TokenBlacklist).filter(TokenBlacklist.jti == jti).first()
    if blacklisted_token:
        raise HTTPException(
            status_code=401,
            detail="令牌已失效",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 查询用户（用户名大小写不敏感）
    user = db.query(DBUser).filter(DBUser.username.ilike(username)).first()
    if user is None:
        raise credentials_exception
    return user


async def get_current_developer(current_user: User = Depends(get_current_user)) -> User:
    """获取当前开发者用户（需要开发者权限）"""
    if not current_user.is_developer:
        raise HTTPException(
            status_code=403,
            detail="需要开发者权限",
        )
    return current_user


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


@router.post("/register")
async def register(
    request: Request,
    db: Session = Depends(get_db)
):
    """用户注册 - 支持表单格式和JSON格式"""
    username = None
    email = None
    password = None
    
    # 尝试从JSON请求体获取数据
    try:
        json_data = await request.json()
        if isinstance(json_data, dict):
            username = json_data.get("username")
            email = json_data.get("email")
            password = json_data.get("password")
    except:
        pass
    
    # 如果JSON解析失败，尝试从表单获取
    if username is None:
        form_data = await request.form()
        username = form_data.get("username")
        email = form_data.get("email")
        password = form_data.get("password")
    
    # 用户名和邮箱归一化（转为小写并去除首尾空格）
    username = (username or "").lower().strip()
    email = (email or "").lower().strip()
    
    # 验证参数
    if not username or not email or not password:
        raise HTTPException(
            status_code=422,
            detail="需要提供用户名、邮箱和密码",
        )
    
    # 检查用户是否已存在
    existing_user = db.query(DBUser).filter(DBUser.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    existing_email = db.query(DBUser).filter(DBUser.email == email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="邮箱已被注册")
    
    # 创建用户
    try:
        hashed_password = get_password_hash(password)
        new_user = DBUser(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            hashed_password=hashed_password,
            is_developer=False,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"用户注册成功: {username}")
        
        return {
            "success": True,
            "message": "注册成功",
            "user": {
                "username": new_user.username,
                "email": new_user.email,
                "is_developer": new_user.is_developer
            }
        }
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="用户名或邮箱已存在")


@router.post("/login")
async def login(
    request: Request,
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """用户登录 - 支持表单格式和JSON格式"""
    # 尝试从JSON请求体获取数据
    try:
        json_data = await request.json()
        if isinstance(json_data, dict):
            username = json_data.get("username") or username
            password = json_data.get("password") or password
    except:
        pass
    
    # 验证参数
    if not username or not password:
        raise HTTPException(
            status_code=422,
            detail="需要提供用户名和密码",
        )
    
    # 用户名归一化
    username = username.lower().strip()
    
    # 查询用户
    user = db.query(DBUser).filter(DBUser.username == username).first()
    
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    logger.info(f"用户登录成功: {user.username}, 开发者: {user.is_developer}")
    
    response = JSONResponse({
        "success": True,
        "token": access_token,
        "username": user.username,
        "email": user.email,
        "role": "developer" if user.is_developer else "user",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "username": user.username,
            "email": user.email,
            "is_developer": user.is_developer
        }
    })
    
    # 设置 cookie - 使用安全配置
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=settings.security.cookie_secure,  # 从配置读取
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite=settings.security.cookie_samesite
    )
    
    return response


@router.post("/logout")
async def logout(
    request: Request,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """用户登出 - 将 Token 加入黑名单，支持密钥轮换"""
    # 优先从 cookie 获取 token
    token_from_cookie = request.cookies.get("access_token")
    if token_from_cookie:
        token = token_from_cookie
    
    # 使用支持密钥轮换的解码函数
    payload = decode_jwt_token(token)
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="无效的令牌",
        )
    
    jti = payload.get("jti")
    expires_at = datetime.fromtimestamp(payload.get("exp"), timezone.utc)
    
    # 将 token 加入黑名单
    blacklisted_token = TokenBlacklist(
        id=str(uuid.uuid4()),
        jti=jti,
        expires_at=expires_at
    )
    db.add(blacklisted_token)
    db.commit()
    
    logger.info(f"用户登出成功")
    
    # 清除 cookie
    response = JSONResponse({"success": True, "message": "登出成功"})
    response.delete_cookie("access_token", secure=settings.security.cookie_secure)
    return response


@router.get("/me")
async def get_profile(current_user: User = Depends(get_current_user)):
    """获取当前用户信息"""
    return {
        "success": True,
        "user": {
            "username": current_user.username,
            "email": current_user.email,
            "is_developer": current_user.is_developer,
            "created_at": current_user.created_at,
            "updated_at": current_user.updated_at
        }
    }


@router.post("/promote")
async def promote_user(
    username: str,
    current_user: User = Depends(get_current_developer),
    db: Session = Depends(get_db)
):
    """提升用户为开发者（需要开发者权限）"""
    # 用户名归一化
    username = username.lower().strip()
    
    user = db.query(DBUser).filter(DBUser.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    user.is_developer = True
    user.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)
    
    logger.info(f"用户提升为开发者: {username}")
    
    return {
        "success": True,
        "message": f"用户 {username} 已提升为开发者",
        "user": {
            "username": user.username,
            "email": user.email,
            "is_developer": user.is_developer
        }
    }


@router.get("/users")
async def list_users(
    current_user: User = Depends(get_current_developer),
    db: Session = Depends(get_db)
):
    """列出所有用户（需要开发者权限）"""
    users = db.query(DBUser).all()
    return {
        "success": True,
        "users": [
            {
                "username": user.username,
                "email": user.email,
                "is_developer": user.is_developer,
                "created_at": user.created_at
            } for user in users
        ]
    }


# 初始化数据库
init_db()

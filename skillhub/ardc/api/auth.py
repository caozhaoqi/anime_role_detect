#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户认证模块
提供登录、注册、登出等功能
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import Optional, Dict
import json
from pathlib import Path

# JWT 配置
SECRET_KEY = "ard-skill-hub-secret-key-2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 密码上下文 - 使用 sha256_crypt 避免 bcrypt 的平台依赖问题
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# OAuth2 Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# 路由
router = APIRouter(prefix="/api/auth")

# 用户数据存储
USER_DATA_PATH = Path.home() / ".ardc" / "users.json"

class User(BaseModel):
    id: str
    username: str
    email: str
    hashed_password: str
    is_developer: bool = False
    created_at: str
    updated_at: str

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

def load_users() -> Dict[str, User]:
    """加载用户数据"""
    if USER_DATA_PATH.exists():
        try:
            with open(USER_DATA_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {k: User(**v) for k, v in data.items()}
        except Exception as e:
            print(f"加载用户数据失败: {e}")
    return {}

def save_users(users: Dict[str, User]):
    """保存用户数据"""
    USER_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump({k: v.dict() for k, v in users.items()}, f, indent=2, ensure_ascii=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """生成密码哈希"""
    # bcrypt 限制密码长度不能超过 72 字节
    password = password[:72]
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=401,
        detail="无法验证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    users = load_users()
    user = users.get(username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_developer(current_user: User = Depends(get_current_user)) -> User:
    """获取当前开发者用户（需要开发者权限）"""
    if not current_user.is_developer:
        raise HTTPException(
            status_code=403,
            detail="需要开发者权限"
        )
    return current_user

@router.post("/register", response_model=dict)
def register(user: UserCreate):
    """用户注册"""
    users = load_users()
    
    # 检查用户名是否已存在
    if user.username in users:
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    # 检查邮箱是否已存在
    for existing_user in users.values():
        if existing_user.email == user.email:
            raise HTTPException(status_code=400, detail="邮箱已被注册")
    
    # 创建新用户
    now = datetime.now().isoformat()
    new_user = User(
        id=user.username,
        username=user.username,
        email=user.email,
        hashed_password=get_password_hash(user.password),
        is_developer=False,
        created_at=now,
        updated_at=now
    )
    
    users[user.username] = new_user
    save_users(users)
    
    return {
        "message": "注册成功",
        "user": {
            "username": user.username,
            "email": user.email,
            "is_developer": False
        }
    }

@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """用户登录"""
    users = load_users()
    
    user = users.get(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
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
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "username": user.username,
            "email": user.email,
            "is_developer": user.is_developer
        }
    }

@router.post("/logout")
def logout():
    """用户登出"""
    return {"message": "登出成功"}

@router.get("/me")
async def get_profile(current_user: User = Depends(get_current_user)):
    """获取当前用户信息"""
    return {
        "username": current_user.username,
        "email": current_user.email,
        "is_developer": current_user.is_developer,
        "created_at": current_user.created_at
    }

@router.post("/users/{username}/promote")
async def promote_user(username: str, current_user: User = Depends(get_current_developer)):
    """提升用户为开发者（需要开发者权限）"""
    users = load_users()
    
    if username not in users:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    users[username].is_developer = True
    users[username].updated_at = datetime.now().isoformat()
    save_users(users)
    
    return {"message": f"用户 {username} 已提升为开发者"}

@router.get("/users")
async def list_users(current_user: User = Depends(get_current_developer)):
    """列出所有用户（需要开发者权限）"""
    users = load_users()
    return {
        "users": [
            {
                "username": u.username,
                "email": u.email,
                "is_developer": u.is_developer,
                "created_at": u.created_at
            }
            for u in users.values()
        ]
    }

@router.get("/users/{username}")
async def get_user(username: str, current_user: User = Depends(get_current_user)):
    """查看用户信息（自己或开发者可以查看所有）"""
    users = load_users()
    
    if username not in users:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    # 普通用户只能查看自己的信息
    if not current_user.is_developer and current_user.username != username:
        raise HTTPException(status_code=403, detail="无权限查看此用户信息")
    
    user = users[username]
    return {
        "username": user.username,
        "email": user.email,
        "is_developer": user.is_developer,
        "created_at": user.created_at,
        "updated_at": user.updated_at
    }
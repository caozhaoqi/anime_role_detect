#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API v1 版本路由入口
"""

from fastapi import APIRouter

from .skills import router as skills_router
from .search import router as search_router
from .metadata import router as metadata_router

# 创建 v1 版本路由
router = APIRouter(prefix="/v1")

# 注册子路由
router.include_router(skills_router)
router.include_router(search_router)
router.include_router(metadata_router)

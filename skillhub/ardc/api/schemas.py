#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 数据模型定义
"""

from typing import Optional, List
from pydantic import BaseModel


class SkillCreate(BaseModel):
    """技能创建请求模型"""

    id: str
    name: str
    version: str
    description: Optional[str] = ""
    author: str
    category: str
    entry_point: str
    tags: Optional[List[str]] = []
    release_notes: Optional[str] = ""


class SkillResponse(BaseModel):
    """技能响应模型"""

    id: str
    name: str
    version: str
    description: str
    author: str
    category: str
    entry_point: str
    tags: List[str]


class VersionInfo(BaseModel):
    """版本信息模型"""

    version: str
    release_notes: Optional[str] = ""
    released_at: Optional[str] = ""


class SearchResult(BaseModel):
    """搜索结果模型"""

    total: int
    skills: List[SkillResponse]


class CategoryInfo(BaseModel):
    """分类信息模型"""

    name: str
    count: int


class StatsResponse(BaseModel):
    """统计信息模型"""

    total_skills: int
    total_categories: int
    total_versions: int


class UpdateCheckResponse(BaseModel):
    """更新检查响应模型"""

    has_update: bool
    current_version: Optional[str]
    latest_version: str
    changelog: Optional[str]

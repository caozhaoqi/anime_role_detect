#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技能元数据模型
定义技能的基本信息和版本管理相关数据结构
"""

from datetime import datetime
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


class SkillDependency(BaseModel):
    """技能依赖定义"""
    skill_id: str = Field(description="依赖技能ID")
    version: str = Field(default="*", description="版本约束，支持语义化版本")
    optional: bool = Field(default=False, description="是否为可选依赖")


class SkillConfigSchema(BaseModel):
    """技能配置参数定义"""
    name: str = Field(description="配置项名称")
    type: Literal['string', 'integer', 'float', 'boolean', 'json'] = Field(description="配置类型")
    default: Optional[dict] = Field(default=None, description="默认值")
    required: bool = Field(default=False, description="是否必填")
    description: str = Field(default="", description="配置说明")


class SkillMetadata(BaseModel):
    """
    技能元数据模型
    描述技能的基本信息和功能特性
    """
    # 基础信息
    id: str = Field(description="技能唯一标识符，格式: ardc-xxx")
    name: str = Field(description="技能名称")
    version: str = Field(description="语义化版本号，如: 1.0.0")
    description: str = Field(default="", description="技能描述")
    author: str = Field(description="作者名称或邮箱")
    author_url: Optional[str] = Field(default=None, description="作者主页")
    
    # 分类与标签
    category: Literal['collector', 'cleaner', 'classifier', 'trainer', 'search', 'analyzer', 'utility'] = Field(
        description="技能分类"
    )
    tags: List[str] = Field(default_factory=list, description="标签列表")
    
    # 技术信息
    entry_point: str = Field(description="技能入口文件路径")
    runtime: Literal['python', 'shell', 'nodejs'] = Field(default="python", description="运行时环境")
    dependencies: List[SkillDependency] = Field(default_factory=list, description="依赖列表")
    config_schema: List[SkillConfigSchema] = Field(default_factory=list, description="配置参数定义")
    
    # 状态信息
    status: Literal['development', 'testing', 'stable', 'deprecated'] = Field(default="development", description="技能状态")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    # 兼容性
    min_platform_version: str = Field(default="1.0.0", description="最低平台版本要求")
    max_platform_version: Optional[str] = Field(default=None, description="最高平台版本要求")
    
    # 资源需求
    memory_mb: int = Field(default=256, description="内存需求(MB)")
    cpu_cores: float = Field(default=1.0, description="CPU核心数")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "ardc-collector",
                "name": "数据采集技能",
                "version": "1.0.0",
                "description": "从多个数据源采集动漫角色图片",
                "author": "anime-role-detect",
                "category": "collector",
                "tags": ["采集", "图片", "动漫"],
                "entry_point": "scripts/collect_images.py",
                "dependencies": [{"skill_id": "ardc-client", "version": ">=1.0.0"}],
                "status": "stable"
            }
        }


class VersionInfo(BaseModel):
    """版本信息"""
    version: str = Field(description="版本号")
    metadata: SkillMetadata = Field(description="该版本的元数据")
    release_notes: str = Field(default="", description="版本更新说明")
    released_at: datetime = Field(default_factory=datetime.now, description="发布时间")
    download_count: int = Field(default=0, description="下载次数")


class SkillSearchResult(BaseModel):
    """技能搜索结果"""
    total: int = Field(description="总数量")
    skills: List[SkillMetadata] = Field(description="技能列表")


class InstalledSkill(BaseModel):
    """已安装技能信息"""
    metadata: SkillMetadata = Field(description="技能元数据")
    install_path: str = Field(description="安装路径")
    installed_at: datetime = Field(default_factory=datetime.now, description="安装时间")
    last_used_at: Optional[datetime] = Field(default=None, description="最后使用时间")
    enabled: bool = Field(default=True, description="是否启用")
    config: Dict[str, dict] = Field(default_factory=dict, description="当前配置")
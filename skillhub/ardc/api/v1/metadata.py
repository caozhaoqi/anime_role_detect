#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
元数据 API v1 版本
提供分类、统计等元数据查询功能，支持缓存优化
"""

from fastapi import APIRouter

from ardc.store.index import SkillIndex
from ardc.utils.logging import get_logger
from ardc.cache import cache, CacheKeys

logger = get_logger(__name__)

router = APIRouter(prefix="", tags=["metadata"])

# 全局索引实例
index = SkillIndex()


@router.get("/stats")
def get_stats():
    """获取统计信息（带缓存）"""
    cached_result = cache.get(CacheKeys.STATS)
    if cached_result is not None:
        logger.debug(f"缓存命中: {CacheKeys.STATS}")
        return cached_result
    
    result = index.get_statistics()
    cache.set(CacheKeys.STATS, result, ttl_seconds=300)  # 统计数据缓存5分钟
    return result


@router.get("/categories")
def get_categories():
    """获取所有技能分类（带缓存）"""
    cached_result = cache.get(CacheKeys.CATEGORIES)
    if cached_result is not None:
        logger.debug(f"缓存命中: {CacheKeys.CATEGORIES}")
        return cached_result
    
    try:
        categories = index.get_categories()
        if not categories:
            result = {"categories": [], "message": "暂无分类数据"}
        else:
            result = {"categories": [{"name": name, "count": count} for name, count in categories.items()]}
        
        cache.set(CacheKeys.CATEGORIES, result, ttl_seconds=600)  # 分类缓存10分钟
        return result
    except Exception as e:
        logger.error(f"获取分类失败: {e}")
        return {"categories": [], "message": "获取分类失败"}


@router.get("/health")
def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "ARD Skill Hub API", "version": "1.0.0"}

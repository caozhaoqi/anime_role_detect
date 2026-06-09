#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索 API v1 版本
提供技能搜索功能，支持缓存优化
"""

from typing import Optional
from fastapi import APIRouter, HTTPException

from ardc.store.index import SkillIndex
from ardc.utils.logging import get_logger
from ardc.cache import cache, CacheKeys

logger = get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

# 全局索引实例
index = SkillIndex()


@router.get("")
def search_skills(keyword: str, category: Optional[str] = None, limit: int = 20):
    """搜索技能（带缓存）"""
    cache_key = (
        f"{CacheKeys.SKILL_SEARCH.format(keyword=keyword, category=category or 'all')}:{limit}"
    )

    cached_result = cache.get(cache_key)
    if cached_result is not None:
        logger.debug(f"缓存命中: {cache_key}")
        return cached_result

    try:
        results = index.search(keyword, category, limit=limit)
        result = {"total": len(results), "skills": [s.dict() for s in results]}

        cache.set(cache_key, result, ttl_seconds=1800)  # 搜索结果缓存30分钟
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

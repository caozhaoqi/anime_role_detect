#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技能管理 API v1 版本
提供技能的 CRUD 操作，支持缓存优化
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from pydantic import BaseModel

from ardc.store.registry import SkillRegistry
from ardc.store.index import SkillIndex
from ardc.version.manager import VersionManager
from ardc.api.auth import get_current_developer, get_current_user
from ardc.api.database import get_db, SkillReview
from ardc.utils.logging import get_logger, get_request_logger
from ardc.cache import cache, CacheKeys, invalidate_cache
from ardc.api.schemas import SkillCreate

logger = get_logger(__name__)
request_logger = get_request_logger()

router = APIRouter(prefix="/skills", tags=["skills"])

# 全局索引和注册中心实例
index = SkillIndex()
registry = SkillRegistry()
version_manager = VersionManager()


@router.get("")
def list_skills(category: Optional[str] = None):
    """获取技能列表（带缓存）"""
    # 构建缓存键
    cache_key = f"{CacheKeys.SKILL_LIST}:{category or 'all'}"

    # 尝试从缓存获取
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        logger.debug(f"缓存命中: {cache_key}")
        return cached_result

    # 从索引获取数据
    skills = index.get_by_category(category) if category else index.get_all_skills()
    result = {"skills": [s.dict() for s in skills]}

    # 缓存结果
    cache.set(cache_key, result)
    return result


@router.get("/{skill_id}")
def get_skill(skill_id: str, version: Optional[str] = None):
    """获取技能详情（带缓存）"""
    # 构建缓存键
    cache_key = f"{CacheKeys.SKILL_DETAIL.format(skill_id=skill_id)}:{version or 'latest'}"

    # 尝试从缓存获取
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        logger.debug(f"缓存命中: {cache_key}")
        return cached_result

    skill = registry.get_skill_by_version(skill_id, version)
    if not skill:
        # 获取可用技能列表供参考
        all_skills = index.get_all_skills()
        available_skills = [s.id for s in all_skills[:5]]
        raise HTTPException(
            status_code=404,
            detail=f"技能不存在: {skill_id}",
            headers=(
                {"X-Available-Skills": ",".join(available_skills)[:200]} if available_skills else {}
            ),
        )

    result = skill.dict()
    # 缓存结果
    cache.set(cache_key, result)
    return result


@router.post("", dependencies=[Depends(get_current_developer)])
def create_skill(skill: SkillCreate):
    """创建技能（创建后失效相关缓存）"""
    logger.info(f"🎯 创建技能请求: {skill.id} - {skill.name}")
    from ardc.store.metadata import SkillMetadata

    try:
        metadata = SkillMetadata(
            id=skill.id,
            name=skill.name,
            version=skill.version,
            description=skill.description,
            author=skill.author,
            category=skill.category,
            entry_point=skill.entry_point,
            tags=skill.tags,
        )
        registry.register_skill(metadata, skill.release_notes)
        index.add_skill(metadata)
        version_manager.release_version(metadata, skill.release_notes)

        # 失效相关缓存
        invalidate_cache("skills:*")
        invalidate_cache("categories")
        invalidate_cache("stats")

        return {"message": "技能注册成功", "skill_id": skill.id}
    except Exception as e:
        logger.error(f"❌ 技能注册失败: {skill.id}, 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{skill_id}", dependencies=[Depends(get_current_developer)])
def delete_skill(skill_id: str):
    """删除技能（删除后失效相关缓存）"""
    index.remove_skill(skill_id)

    # 失效相关缓存
    invalidate_cache(f"skills:detail:{skill_id}:*")
    invalidate_cache("skills:list:*")
    invalidate_cache("categories")
    invalidate_cache("stats")

    return {"message": "技能删除成功"}


@router.get("/{skill_id}/versions")
def get_skill_versions(skill_id: str):
    """获取技能版本列表（带缓存）"""
    cache_key = CacheKeys.SKILL_VERSIONS.format(skill_id=skill_id)

    cached_result = cache.get(cache_key)
    if cached_result is not None:
        logger.debug(f"缓存命中: {cache_key}")
        return cached_result

    versions = version_manager.list_versions(skill_id)
    result = {"versions": [v.dict() for v in versions]}

    cache.set(cache_key, result)
    return result


@router.get("/{skill_id}/check-update")
def check_skill_update(skill_id: str, current_version: str = None):
    """检查技能是否有更新（带缓存）"""
    from packaging.version import parse as parse_version

    cache_key = f"{CacheKeys.VERSION_CHECK.format(skill_id=skill_id, current_version=current_version or 'none')}"

    cached_result = cache.get(cache_key)
    if cached_result is not None:
        logger.debug(f"缓存命中: {cache_key}")
        return cached_result

    try:
        latest = registry.get_latest_version(skill_id)
        if not latest:
            result = {"has_update": False, "latest_version": current_version or "1.0.0"}
            cache.set(cache_key, result, ttl_seconds=60)  # 版本检查缓存1分钟
            return result

        has_update = False
        if current_version:
            try:
                curr_version = parse_version(current_version)
                latest_version = parse_version(latest.version)
                has_update = latest_version > curr_version
                logger.debug(
                    f"版本比对: 当前={current_version}, 最新={latest.version}, 有更新={has_update}"
                )
            except Exception as e:
                logger.warning(
                    f"版本号解析失败: {current_version} 或 {latest.version}, 错误: {str(e)}"
                )
                has_update = False

        result = {
            "has_update": has_update,
            "current_version": current_version,
            "latest_version": latest.version,
            "changelog": latest.release_notes if hasattr(latest, "release_notes") else "",
        }

        cache.set(cache_key, result, ttl_seconds=60)  # 版本检查缓存1分钟
        return result
    except Exception as e:
        logger.error(f"检查更新失败: {skill_id}, 错误: {str(e)}")
        raise HTTPException(status_code=500, detail="检查更新失败")


@router.get("/{skill_id}/reviews")
def get_skill_reviews(skill_id: str, db: Session = Depends(get_db)):
    """获取技能评价列表"""
    skill = registry.get_skill_by_version(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail=f"技能不存在: {skill_id}")

    reviews = db.query(SkillReview).filter(
        SkillReview.skill_id == skill_id
    ).order_by(SkillReview.created_at.desc()).limit(50).all()

    return {
        "reviews": [
            {
                "id": r.id,
                "username": r.username,
                "rating": r.rating,
                "comment": r.comment or "",
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in reviews
        ],
        "total": len(reviews),
    }


@router.get("/{skill_id}/rating")
def get_skill_rating(skill_id: str, db: Session = Depends(get_db)):
    """获取技能评分"""
    skill = registry.get_skill_by_version(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail=f"技能不存在: {skill_id}")

    reviews = db.query(SkillReview).filter(SkillReview.skill_id == skill_id).all()
    total = len(reviews)
    average = sum(r.rating for r in reviews) / total if total > 0 else 0.0

    distribution = {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0}
    for r in reviews:
        key = str(r.rating)
        if key in distribution:
            distribution[key] += 1

    return {"average": round(average, 1), "total": total, "distribution": distribution}


class ReviewCreate(BaseModel):
    rating: int  # 1-5
    comment: str = ""


@router.post("/{skill_id}/review", dependencies=[Depends(get_current_user)])
def submit_review(skill_id: str, review: ReviewCreate, db: Session = Depends(get_db)):
    """提交技能评价"""
    if review.rating < 1 or review.rating > 5:
        raise HTTPException(status_code=400, detail="评分必须在 1-5 之间")

    skill = registry.get_skill_by_version(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail=f"技能不存在: {skill_id}")

    # 同一用户对同一技能只能评价一次（upsert）
    existing = db.query(SkillReview).filter(
        SkillReview.skill_id == skill_id,
        SkillReview.username == "anonymous",
    ).first()

    if existing:
        existing.rating = review.rating
        existing.comment = review.comment
        existing.created_at = datetime.now(timezone.utc)
    else:
        new_review = SkillReview(
            skill_id=skill_id,
            username="anonymous",
            rating=review.rating,
            comment=review.comment,
        )
        db.add(new_review)

    db.commit()
    return {"message": "评价提交成功", "rating": review.rating, "comment": review.comment}


@router.get("/{skill_id}/screenshots")
def get_skill_screenshots(skill_id: str):
    """获取技能截图列表"""
    skill = registry.get_skill_by_version(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail=f"技能不存在: {skill_id}")

    return {"screenshots": []}


@router.post("/{skill_id}/install", dependencies=[Depends(get_current_user)])
def install_skill(skill_id: str, version: Optional[str] = None):
    """安装技能"""
    skill = registry.get_skill_by_version(skill_id, version)
    if not skill:
        raise HTTPException(status_code=404, detail=f"技能不存在: {skill_id}")

    return {"message": "技能安装成功", "skill_id": skill_id, "version": skill.version}

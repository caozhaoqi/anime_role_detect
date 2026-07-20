#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索路由模块 - 使用独立搜索服务
"""

import os
import sys
import io

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.logging.global_logger import get_logger
from src.services.search_service.search_client import get_search_client

logger = get_logger("search_routes")

# 创建路由器
router = APIRouter(prefix="/api/search", tags=["搜索服务"])


@router.post("/image")
async def search_similar_images(file: UploadFile = File(...), top_k: int = 10):
    """
    以图搜图 - 搜索相似图像

    Args:
        file: 上传的查询图像
        top_k: 返回前k个相似图像

    Returns:
        相似图像列表，包含路径和相似度
    """
    try:
        # 获取搜索客户端
        client = get_search_client()

        # 检查服务状态
        if not client.health_check():
            raise HTTPException(status_code=503, detail="搜索服务未启动")

        # 读取图像
        content = await file.read()

        # 调用搜索服务
        result = client.search_image_bytes(content, file.filename, top_k)

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "搜索失败"))

        return {"query": file.filename, "count": result["count"], "results": result["results"]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"以图搜图失败: {e}")
        import traceback

        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {e}")


@router.post("/build-index")
async def build_search_index(dataset_dir: str = "data/merged_english_dataset"):
    """
    构建搜索索引

    Args:
        dataset_dir: 数据集目录

    Returns:
        索引构建结果
    """
    try:
        # 获取搜索客户端
        client = get_search_client()

        # 检查服务状态
        if not client.health_check():
            raise HTTPException(status_code=503, detail="搜索服务未启动")

        # 调用搜索服务构建索引
        result = client.build_index(dataset_dir)

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "构建索引失败"))

        return {
            "status": "success",
            "dataset_dir": result["dataset_dir"],
            "added_images": result["added_images"],
            "index_stats": result["index_stats"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"构建索引失败: {e}")
        import traceback

        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"构建索引失败: {e}")


@router.get("/stats")
async def get_search_stats():
    """
    获取搜索服务统计信息

    Returns:
        索引统计信息
    """
    try:
        # 获取搜索客户端
        client = get_search_client()

        # 检查服务状态
        if not client.health_check():
            raise HTTPException(status_code=503, detail="搜索服务未启动")

        # 获取统计信息
        stats = client.get_stats()

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {e}")


@router.get("/health")
async def search_service_health():
    """
    检查搜索服务健康状态

    Returns:
        健康状态
    """
    try:
        client = get_search_client()
        healthy = client.health_check()
        return {"status": "healthy" if healthy else "unhealthy", "service_url": client.service_url}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# 视频识别功能
@router.post("/video/recognize")
async def recognize_video(
    file: UploadFile = File(...), frame_interval: float = 1.0, confidence_threshold: float = 0.5
):
    """
    视频文件角色识别

    Args:
        file: 上传的视频文件
        frame_interval: 抽帧间隔（秒）
        confidence_threshold: 置信度阈值

    Returns:
        识别结果列表
    """
    try:
        import tempfile
        import time

        # 保存上传的视频文件
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_video_path = temp_file.name

        logger.info(f"开始处理视频: {file.filename}")

        # 延迟导入视频识别服务
        from src.services.video_service.video_recognition_service import VideoRecognitionService

        # 创建视频识别服务实例
        service = VideoRecognitionService(
            frame_interval=frame_interval, confidence_threshold=confidence_threshold
        )

        # 处理视频
        results = service.process_video_file(temp_video_path)

        # 清理临时文件
        os.remove(temp_video_path)

        # 统计识别到的角色
        role_counts = {}
        for result in results:
            role = result["role"]
            role_counts[role] = role_counts.get(role, 0) + 1

        # 构建响应
        response = {
            "video": file.filename,
            "frame_interval": frame_interval,
            "total_frames": service.frame_count,
            "detections": len(results),
            "roles": [
                {"role": role, "count": count}
                for role, count in sorted(role_counts.items(), key=lambda x: x[1], reverse=True)
            ],
            "timestamps": [
                {
                    "timestamp": result["timestamp"],
                    "role": result["role"],
                    "similarity": result["similarity"],
                    "attributes": [attr["tag"] for attr in result.get("attributes", [])[:5]],
                }
                for result in results
            ],
        }

        logger.info(f"视频识别完成: {file.filename}, 检测到 {len(role_counts)} 个角色")
        return response

    except Exception as e:
        logger.error(f"视频识别失败: {e}")
        import traceback

        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"视频识别失败: {e}")

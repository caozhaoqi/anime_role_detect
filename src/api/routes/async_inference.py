#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步推理路由
支持通过Redis队列进行异步推理
"""

import os
import sys
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, Field

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.logging import get_enhanced_logger as get_logger
from src.services.inference_queue.queue_manager import get_queue_manager, TaskStatus

logger = get_logger("async_inference")

# 创建路由器
router = APIRouter(prefix="/api/async", tags=["异步推理"])


# 请求/响应模型
class InferenceRequest(BaseModel):
    """推理请求"""
    model_name: str = Field(default="ViT-B/32", description="使用的模型名称")
    top_k: int = Field(default=5, ge=1, le=20, description="返回前K个结果")
    use_cache: bool = Field(default=True, description="是否使用缓存")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "ViT-B/32",
                "top_k": 5,
                "use_cache": True,
            }
        }


class InferenceResponse(BaseModel):
    """推理响应"""
    task_id: str = Field(description="任务ID")
    status: str = Field(description="任务状态")
    message: str = Field(description="状态消息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "message": "任务已提交，请通过状态接口查询结果",
            }
        }


class InferenceResult(BaseModel):
    """推理结果"""
    character: str = Field(description="角色名称")
    similarity: float = Field(description="相似度")
    
    class Config:
        json_schema_extra = {
            "example": {
                "character": "Arona",
                "similarity": 0.9234,
            }
        }


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str = Field(description="任务ID")
    status: str = Field(description="任务状态")
    updated_at: Optional[str] = Field(None, description="最后更新时间")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "updated_at": "2024-01-01T12:00:00",
            }
        }


class TaskResultResponse(BaseModel):
    """任务结果响应"""
    task_id: str = Field(description="任务ID")
    status: str = Field(description="任务状态")
    results: Optional[list[InferenceResult]] = Field(None, description="识别结果")
    error: Optional[str] = Field(None, description="错误信息")
    completed_at: Optional[str] = Field(None, description="完成时间")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "results": [
                    {"character": "Arona", "similarity": 0.9234},
                    {"character": "Hoshino", "similarity": 0.8123},
                ],
                "error": None,
                "completed_at": "2024-01-01T12:00:05",
            }
        }


class QueueStatsResponse(BaseModel):
    """队列统计响应"""
    pending_tasks: int = Field(description="待处理任务数")
    processing_tasks: int = Field(description="处理中任务数")
    total_active: int = Field(description="总活跃任务数")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pending_tasks": 5,
                "processing_tasks": 2,
                "total_active": 7,
            }
        }


@router.post(
    "/inference",
    response_model=InferenceResponse,
    summary="提交异步推理任务",
    description="上传图片并提交异步推理任务，返回任务ID用于查询结果",
    response_description="任务提交成功，返回任务ID",
)
async def submit_inference(
    file: UploadFile = File(..., description="上传的动漫角色图片"),
    model_name: str = Query(default="ViT-B/32", description="使用的模型名称"),
    top_k: int = Query(default=5, ge=1, le=20, description="返回前K个结果"),
    use_cache: bool = Query(default=True, description="是否使用缓存"),
):
    """
    提交异步推理任务
    
    上传动漫角色图片，系统将自动识别角色并返回最相似的角色列表。
    由于推理可能需要一定时间，此接口采用异步方式，提交任务后通过任务ID查询结果。
    
    - **file**: 待识别的动漫角色图片（JPG/PNG）
    - **model_name**: CLIP模型名称，可选 ViT-B/32 或 ViT-L/14
    - **top_k**: 返回最相似的K个角色，范围 1-20
    - **use_cache**: 是否使用特征缓存加速
    """
    try:
        # 读取图片
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="上传的文件为空")
        
        # 提交任务
        queue_manager = get_queue_manager()
        task_id = queue_manager.submit_task(
            image_data=content,
            model_name=model_name,
            top_k=top_k,
            use_cache=use_cache,
        )
        
        logger.info(f"异步推理任务提交: {task_id}, 文件: {file.filename}")
        
        return InferenceResponse(
            task_id=task_id,
            status=TaskStatus.PENDING.value,
            message="任务已提交，请通过状态接口查询结果",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交推理任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"提交失败: {e}")


@router.get(
    "/status/{task_id}",
    response_model=TaskStatusResponse,
    summary="查询任务状态",
    description="根据任务ID查询异步推理任务的当前状态",
    response_description="返回任务当前状态",
)
async def get_task_status(task_id: str):
    """
    查询任务状态
    
    根据提交任务时返回的任务ID，查询任务的当前处理状态。
    
    - **task_id**: 任务ID，提交任务时返回
    
    状态说明：
    - **pending**: 等待处理
    - **processing**: 正在处理
    - **completed**: 处理完成，可通过结果接口获取结果
    - **failed**: 处理失败，可通过结果接口查看错误信息
    """
    try:
        queue_manager = get_queue_manager()
        status = queue_manager.get_status(task_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")
        
        return TaskStatusResponse(
            task_id=task_id,
            status=status["status"],
            updated_at=status.get("updated_at"),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {e}")


@router.get(
    "/result/{task_id}",
    response_model=TaskResultResponse,
    summary="获取任务结果",
    description="根据任务ID获取异步推理任务的最终结果",
    response_description="返回任务结果或错误信息",
)
async def get_task_result(task_id: str):
    """
    获取任务结果
    
    根据任务ID获取异步推理的最终结果。
    只有状态为 completed 或 failed 时才能获取到结果。
    
    - **task_id**: 任务ID，提交任务时返回
    
    返回结果包含：
    - 识别到的角色列表（按相似度排序）
    - 每个角色的相似度分数
    - 错误信息（如果处理失败）
    """
    try:
        queue_manager = get_queue_manager()
        
        # 先检查状态
        status = queue_manager.get_status(task_id)
        if status is None:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")
        
        # 获取结果
        result = queue_manager.get_result(task_id)
        
        if result is None:
            return TaskResultResponse(
                task_id=task_id,
                status=status["status"],
                results=None,
                error=None,
                completed_at=None,
            )
        
        # 解析结果
        if result.get("status") == "completed":
            inference_result = result.get("result", {})
            results = inference_result.get("results", [])
            
            return TaskResultResponse(
                task_id=task_id,
                status="completed",
                results=[InferenceResult(**r) for r in results],
                error=None,
                completed_at=result.get("completed_at"),
            )
        else:
            return TaskResultResponse(
                task_id=task_id,
                status="failed",
                results=None,
                error=result.get("error", "未知错误"),
                completed_at=result.get("failed_at"),
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取结果失败: {e}")


@router.get(
    "/queue/stats",
    response_model=QueueStatsResponse,
    summary="获取队列统计",
    description="获取当前推理队列的统计信息",
    response_description="返回队列统计信息",
)
async def get_queue_statistics():
    """
    获取队列统计
    
    返回当前推理队列的统计信息，包括：
    - 待处理任务数
    - 处理中任务数
    - 总活跃任务数
    """
    try:
        queue_manager = get_queue_manager()
        stats = queue_manager.get_queue_stats()
        
        return QueueStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"获取队列统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计失败: {e}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗流水线API路由
提供数据清洗流水线的Web接口
"""

import sys
import os
from pathlib import Path
import logging

# 添加项目根目录
# __file__ = src/api/routes/cleaning_routes.py -> 需要上溯4层到项目根目录
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks, Query
from typing import Optional, List, Dict
from pydantic import BaseModel
import time
import json
import asyncio
import subprocess

from src.data_pipeline.cleaning_pipeline import CleaningPipeline, CleaningConfig
from src.middleware.auth_enhanced import get_current_admin

router = APIRouter(prefix="/api/cleaning", tags=["数据清洗"])


class CleaningConfigRequest(BaseModel):
    """清洗配置请求模型"""
    # 阶段开关
    enable_deduplication: bool = True
    enable_consistency_filter: bool = True
    enable_cluster_filter: bool = True
    enable_mislabeled_detector: bool = True
    enable_danbooru_enrichment: bool = False
    
    # 参数
    similarity_threshold: float = 0.95
    consistency_threshold: float = 0.25
    outlier_threshold: float = 0.7
    text_threshold: float = 0.2
    confusion_gap: float = 0.08
    
    # 模式
    dry_run: bool = False
    min_images_per_character: int = 5


class CleaningResponse(BaseModel):
    """清洗响应模型"""
    success: bool
    message: str
    data: Optional[dict] = None
    task_id: Optional[str] = None


class CleaningReport(BaseModel):
    """清洗报告模型"""
    start_time: str
    end_time: str
    duration_seconds: float
    
    total_characters: int
    total_original_images: int
    total_cleaned_images: int
    total_removed_images: int
    overall_keep_rate: float
    
    dedup_removed: int
    consistency_removed: int
    cluster_removed: int
    mislabeled_removed: int
    
    character_results: Dict[str, dict]
    config: dict


@router.get("/config/default", response_model=CleaningResponse)
async def get_default_config():
    """
    获取默认清洗配置
    
    Returns:
        默认清洗配置
    """
    config = CleaningConfig()
    return {
        "success": True,
        "message": "获取默认配置成功",
        "data": {
            "enable_deduplication": config.enable_deduplication,
            "enable_consistency_filter": config.enable_consistency_filter,
            "enable_cluster_filter": config.enable_cluster_filter,
            "enable_mislabeled_detector": config.enable_mislabeled_detector,
            "enable_danbooru_enrichment": config.enable_danbooru_enrichment,
            "similarity_threshold": config.similarity_threshold,
            "consistency_threshold": config.consistency_threshold,
            "outlier_threshold": config.outlier_threshold,
            "text_threshold": config.text_threshold,
            "confusion_gap": config.confusion_gap,
            "min_images_per_character": config.min_images_per_character,
        }
    }


@router.post("/run", response_model=CleaningResponse)
async def run_cleaning_pipeline(
    input_dir: str = Form(..., description="输入目录路径"),
    output_dir: str = Form(..., description="输出目录路径"),
    enable_deduplication: bool = Form(True, description="启用CLIP去重"),
    enable_consistency_filter: bool = Form(True, description="启用角色一致性过滤"),
    enable_cluster_filter: bool = Form(True, description="启用HDBSCAN聚类过滤"),
    enable_mislabeled_detector: bool = Form(True, description="启用错误标签检测"),
    enable_danbooru_enrichment: bool = Form(False, description="启用Danbooru增强"),
    similarity_threshold: float = Form(0.95, description="相似度阈值"),
    consistency_threshold: float = Form(0.25, description="一致性阈值"),
    outlier_threshold: float = Form(0.7, description="异常阈值"),
    text_threshold: float = Form(0.2, description="文本匹配阈值"),
    confusion_gap: float = Form(0.08, description="混淆差距阈值"),
    dry_run: bool = Form(False, description="干运行模式"),
    min_images_per_character: int = Form(5, description="角色最小图片数"),
    max_workers: int = Form(4, description="并发线程数"),
    current_admin: dict = Depends(get_current_admin),
):
    """
    运行数据清洗流水线（同步模式）
    
    Args:
        input_dir: 输入目录（包含角色子目录）
        output_dir: 输出目录
        enable_deduplication: 是否启用CLIP去重
        enable_consistency_filter: 是否启用角色一致性过滤
        enable_cluster_filter: 是否启用HDBSCAN聚类过滤
        enable_mislabeled_detector: 是否启用错误标签检测
        enable_danbooru_enrichment: 是否启用Danbooru标签增强
        similarity_threshold: 去重相似度阈值
        consistency_threshold: 一致性阈值
        outlier_threshold: 异常检测阈值
        text_threshold: 文本匹配阈值
        confusion_gap: 混淆差距阈值
        dry_run: 干运行模式（不实际删除文件）
        min_images_per_character: 角色最小图片数
    
    Returns:
        清洗报告
    """
    try:
        # 验证输入目录
        if not Path(input_dir).exists():
            raise HTTPException(status_code=400, detail=f"输入目录不存在: {input_dir}")
        
        # 构建配置
        config = CleaningConfig(
            enable_deduplication=enable_deduplication,
            enable_consistency_filter=enable_consistency_filter,
            enable_cluster_filter=enable_cluster_filter,
            enable_mislabeled_detector=enable_mislabeled_detector,
            enable_danbooru_enrichment=enable_danbooru_enrichment,
            similarity_threshold=similarity_threshold,
            consistency_threshold=consistency_threshold,
            outlier_threshold=outlier_threshold,
            text_threshold=text_threshold,
            confusion_gap=confusion_gap,
            dedup_dry_run=dry_run,
            consistency_dry_run=dry_run,
            cluster_dry_run=dry_run,
            min_images_per_character=min_images_per_character,
            max_workers=max_workers,
        )
        
        # 创建流水线
        pipeline = CleaningPipeline(input_dir, output_dir, config)
        
        # 运行流水线
        report = pipeline.run()
        
        # 返回结果
        return {
            "success": True,
            "message": "清洗完成",
            "data": {
                "duration_seconds": report.duration_seconds,
                "total_characters": report.total_characters,
                "total_original_images": report.total_original_images,
                "total_cleaned_images": report.total_cleaned_images,
                "total_removed_images": report.total_removed_images,
                "overall_keep_rate": report.overall_keep_rate,
                "dedup_removed": report.dedup_removed,
                "consistency_removed": report.consistency_removed,
                "cluster_removed": report.cluster_removed,
                "mislabeled_removed": report.mislabeled_removed,
                "character_results": report.character_results,
                "report_path": f"{output_dir}/cleaning_report.json",
            }
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        print(f"清洗失败: {e}")
        print(traceback.format_exc())
        return {
            "success": False,
            "message": f"清洗失败: {str(e)}",
            "data": None
        }


@router.post("/run/async", response_model=CleaningResponse)
async def run_cleaning_pipeline_async(
    input_dir: str = Form(..., description="输入目录路径"),
    output_dir: str = Form(..., description="输出目录路径"),
    enable_deduplication: bool = Form(True, description="启用CLIP去重"),
    enable_consistency_filter: bool = Form(True, description="启用角色一致性过滤"),
    enable_cluster_filter: bool = Form(True, description="启用HDBSCAN聚类过滤"),
    enable_mislabeled_detector: bool = Form(True, description="启用错误标签检测"),
    enable_danbooru_enrichment: bool = Form(False, description="启用Danbooru增强"),
    similarity_threshold: float = Form(0.95, description="相似度阈值"),
    consistency_threshold: float = Form(0.25, description="一致性阈值"),
    outlier_threshold: float = Form(0.7, description="异常阈值"),
    text_threshold: float = Form(0.2, description="文本匹配阈值"),
    confusion_gap: float = Form(0.08, description="混淆差距阈值"),
    dry_run: bool = Form(False, description="干运行模式"),
    min_images_per_character: int = Form(5, description="角色最小图片数"),
    current_admin: dict = Depends(get_current_admin),
):
    """
    运行数据清洗流水线（异步模式）
    
    Args:
        input_dir: 输入目录（包含角色子目录）
        output_dir: 输出目录
        enable_deduplication: 是否启用CLIP去重
        enable_consistency_filter: 是否启用角色一致性过滤
        enable_cluster_filter: 是否启用HDBSCAN聚类过滤
        enable_mislabeled_detector: 是否启用错误标签检测
        enable_danbooru_enrichment: 是否启用Danbooru标签增强
        similarity_threshold: 去重相似度阈值
        consistency_threshold: 一致性阈值
        outlier_threshold: 异常检测阈值
        text_threshold: 文本匹配阈值
        confusion_gap: 混淆差距阈值
        dry_run: 干运行模式（不实际删除文件）
        min_images_per_character: 角色最小图片数
    
    Returns:
        任务ID，用于查询进度
    """
    try:
        # 验证输入目录
        if not Path(input_dir).exists():
            raise HTTPException(status_code=400, detail=f"输入目录不存在: {input_dir}")
        
        # 生成任务ID
        import uuid
        task_id = f"cleaning_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # 保存任务配置
        task_config = {
            "task_id": task_id,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "enable_deduplication": enable_deduplication,
            "enable_consistency_filter": enable_consistency_filter,
            "enable_cluster_filter": enable_cluster_filter,
            "enable_mislabeled_detector": enable_mislabeled_detector,
            "enable_danbooru_enrichment": enable_danbooru_enrichment,
            "similarity_threshold": similarity_threshold,
            "consistency_threshold": consistency_threshold,
            "outlier_threshold": outlier_threshold,
            "text_threshold": text_threshold,
            "confusion_gap": confusion_gap,
            "dry_run": dry_run,
            "min_images_per_character": min_images_per_character,
            "status": "pending",
            "start_time": time.time(),
        }
        
        # 保存任务状态
        tasks_dir = Path(project_root) / "data" / "cleaning_tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        
        task_file = tasks_dir / f"{task_id}.json"
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task_config, f)
        
        # 在子进程中运行清洗任务，避免PyTorch多线程死锁
        worker_script = Path(project_root) / "scripts" / "_run_cleaning_worker.py"
        log_dir = Path(project_root) / "logs" / "cleaning_workers"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{task_id}.log"
        err_file = log_dir / f"{task_id}.err.log"
        
        # 启动子进程（传递环境变量，避免macOS PyTorch多线程死锁）
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["MKL_THREADING_LAYER"] = "GNU"
        
        with open(log_file, "w") as stdout_f, open(err_file, "w") as stderr_f:
            subprocess.Popen(
                [sys.executable, str(worker_script), str(task_file)],
                stdout=stdout_f,
                stderr=stderr_f,
                cwd=str(project_root),
                env=env,
            )
        
        logger.info(f"清洗任务已提交: task_id={task_id}, 子进程已启动")
        
        return {
            "success": True,
            "message": "清洗任务已提交",
            "task_id": task_id,
            "data": {
                "status": "pending",
                "message": "任务正在后台运行，请通过 /api/cleaning/task/{task_id} 查询进度",
            }
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        return {
            "success": False,
            "message": f"提交任务失败: {str(e)}",
            "data": None
        }


@router.get("/task/{task_id}", response_model=CleaningResponse)
async def get_cleaning_task_status(task_id: str, current_admin: dict = Depends(get_current_admin)):
    """
    查询清洗任务状态
    
    Args:
        task_id: 任务ID
    
    Returns:
        任务状态和结果
    """
    try:
        tasks_dir = Path(project_root) / "data" / "cleaning_tasks"
        task_file = tasks_dir / f"{task_id}.json"
        
        if not task_file.exists():
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        
        with open(task_file, "r", encoding="utf-8") as f:
            task_config = json.load(f)
        
        return {
            "success": True,
            "message": "获取任务状态成功",
            "data": {
                "task_id": task_id,
                "status": task_config.get("status"),
                "input_dir": task_config.get("input_dir"),
                "output_dir": task_config.get("output_dir"),
                "start_time": task_config.get("start_time"),
                "end_time": task_config.get("end_time"),
                "duration_seconds": task_config.get("duration_seconds"),
                "result": task_config.get("result"),
                "error": task_config.get("error"),
            }
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        return {
            "success": False,
            "message": f"获取任务状态失败: {str(e)}",
            "data": None
        }


@router.get("/report/{task_id}", response_model=CleaningResponse)
async def get_cleaning_report(task_id: str, current_admin: dict = Depends(get_current_admin)):
    """
    获取清洗报告
    
    Args:
        task_id: 任务ID
    
    Returns:
        清洗报告详情
    """
    try:
        tasks_dir = Path(project_root) / "data" / "cleaning_tasks"
        task_file = tasks_dir / f"{task_id}.json"
        
        if not task_file.exists():
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        
        with open(task_file, "r", encoding="utf-8") as f:
            task_config = json.load(f)
        
        if task_config.get("status") != "completed":
            return {
                "success": True,
                "message": "任务尚未完成",
                "data": {
                    "status": task_config.get("status"),
                    "task_id": task_id,
                }
            }
        
        return {
            "success": True,
            "message": "获取报告成功",
            "data": task_config.get("result", {})
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        return {
            "success": False,
            "message": f"获取报告失败: {str(e)}",
            "data": None
        }


@router.get("/tasks", response_model=CleaningResponse)
async def list_cleaning_tasks():
    """
    获取任务列表
    
    Returns:
        任务列表
    """
    try:
        tasks_dir = Path(project_root) / "data" / "cleaning_tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        
        tasks = []
        for task_file in sorted(tasks_dir.glob("*.json"), reverse=True):
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    task_config = json.load(f)
                    tasks.append({
                        "task_id": task_file.stem,
                        "status": task_config.get("status"),
                        "input_dir": task_config.get("input_dir"),
                        "output_dir": task_config.get("output_dir"),
                        "start_time": task_config.get("start_time"),
                        "end_time": task_config.get("end_time"),
                        "duration_seconds": task_config.get("duration_seconds"),
                        "total_characters": task_config.get("result", {}).get("total_characters"),
                        "total_original_images": task_config.get("result", {}).get("total_original_images"),
                        "total_cleaned_images": task_config.get("result", {}).get("total_cleaned_images"),
                    })
            except Exception as e:
                print(f"读取任务文件失败 {task_file}: {e}")
        
        return {
            "success": True,
            "message": "获取任务列表成功",
            "data": {"tasks": tasks, "count": len(tasks)}
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"获取任务列表失败: {str(e)}",
            "data": None
        }


@router.delete("/task/{task_id}", response_model=CleaningResponse)
async def delete_cleaning_task(task_id: str, current_admin: dict = Depends(get_current_admin)):
    """
    删除任务记录
    
    Args:
        task_id: 任务ID
    
    Returns:
        删除结果
    """
    try:
        tasks_dir = Path(project_root) / "data" / "cleaning_tasks"
        task_file = tasks_dir / f"{task_id}.json"
        
        if not task_file.exists():
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        
        task_file.unlink()
        
        return {
            "success": True,
            "message": "删除任务成功",
            "data": {"task_id": task_id}
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        return {
            "success": False,
            "message": f"删除任务失败: {str(e)}",
            "data": None
        }


@router.get("/browse", response_model=CleaningResponse)
async def browse_directory(path: str = Query("/", description="要浏览的目录路径")):
    """
    浏览服务器上的目录结构 - 用于选择输入/输出目录

    Args:
        path: 要浏览的目录路径

    Returns:
        目录下的子目录列表
    """
    try:
        dir_path = Path(path)
        if not dir_path.exists():
            return {
                "success": False,
                "message": f"目录不存在: {path}",
                "data": {"entries": [], "current_path": path, "parent_path": str(dir_path.parent)}
            }
        if not dir_path.is_dir():
            return {
                "success": False,
                "message": f"路径不是目录: {path}",
                "data": {"entries": [], "current_path": path, "parent_path": str(dir_path.parent)}
            }

        entries = []
        for entry in sorted(dir_path.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                entries.append({
                    "name": entry.name,
                    "path": str(entry.absolute()),
                })

        return {
            "success": True,
            "message": "浏览成功",
            "data": {
                "entries": entries,
                "current_path": str(dir_path.absolute()),
                "parent_path": str(dir_path.parent),
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"浏览目录失败: {str(e)}",
            "data": None
        }


@router.get("/preview", response_model=CleaningResponse)
async def preview_cleaning(
    input_dir: str = Form(..., description="输入目录路径"),
    sample_count: int = Form(5, description="每个角色采样数量"),
    current_admin: dict = Depends(get_current_admin),
):
    """
    预览清洗配置效果（不实际执行清洗）
    
    Args:
        input_dir: 输入目录
        sample_count: 每个角色采样数量
    
    Returns:
        预览报告
    """
    try:
        input_path = Path(input_dir)
        if not input_path.exists():
            raise HTTPException(status_code=400, detail=f"输入目录不存在: {input_dir}")
        
        # 统计角色和图片数量
        characters = []
        total_images = 0
        
        for char_dir in sorted(input_path.iterdir()):
            if char_dir.is_dir():
                images = list(char_dir.glob("*.jpg"))
                if images:
                    characters.append({
                        "name": char_dir.name,
                        "image_count": len(images),
                    })
                    total_images += len(images)
        
        return {
            "success": True,
            "message": "预览成功",
            "data": {
                "input_dir": input_dir,
                "character_count": len(characters),
                "total_images": total_images,
                "characters": characters,
                "preview_info": f"将处理 {len(characters)} 个角色，共 {total_images} 张图片",
            }
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        return {
            "success": False,
            "message": f"预览失败: {str(e)}",
            "data": None
        }

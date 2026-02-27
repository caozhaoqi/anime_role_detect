#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API接口服务

提供外部系统集成的API接口
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.utils.http_utils import HTTPUtils
from src.utils.image_utils import ImageUtils
from src.utils.monitoring_system import MonitoringSystem
from src.utils.cache_manager import cache_manager
from src.data_collection.keyword_based_collector import KeywordBasedDataCollector
from src.utils.distributed_manager import DistributedManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api_service')

# 创建FastAPI应用
app = FastAPI(
    title="Anime Role Detect API",
    description="二次元角色检测系统API接口",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例
monitoring_system = MonitoringSystem()
distributed_manager = DistributedManager()


@app.on_event("startup")
async def startup_event():
    """
    启动事件
    """
    logger.info("启动API服务")
    monitoring_system.start()
    distributed_manager.start()


@app.on_event("shutdown")
async def shutdown_event():
    """
    关闭事件
    """
    logger.info("关闭API服务")
    monitoring_system.stop()
    distributed_manager.stop()


@app.get("/api/health", tags=["系统"])
async def health_check():
    """
    健康检查接口
    """
    return {
        "status": "healthy",
        "service": "Anime Role Detect API"
    }


@app.get("/api/status", tags=["系统"])
async def system_status():
    """
    系统状态接口
    """
    stats = monitoring_system.get_all_stats()
    return {
        "status": "running",
        "stats": stats
    }


@app.get("/api/monitoring", tags=["监控"])
async def get_monitoring_data():
    """
    获取监控数据
    """
    dashboard_data = monitoring_system.get_dashboard_data()
    alerts = monitoring_system.get_alerts()
    
    return {
        "dashboard": dashboard_data,
        "alerts": alerts
    }


@app.get("/api/cache/stats", tags=["缓存"])
async def get_cache_stats():
    """
    获取缓存统计信息
    """
    stats = cache_manager.get_stats()
    return stats


@app.post("/api/cache/clear", tags=["缓存"])
async def clear_cache():
    """
    清除所有缓存
    """
    cache_manager.clear()
    return {
        "status": "success",
        "message": "缓存已清除"
    }


@app.post("/api/collect", tags=["数据采集"])
async def collect_data(
    keyword: str = Query(..., description="搜索关键词"),
    character_name: Optional[str] = Query(None, description="角色名称"),
    max_images: int = Query(10, description="最大图片数量"),
    min_size: int = Query(300, description="最小图片尺寸"),
    enable_cache: bool = Query(True, description="启用缓存")
):
    """
    数据采集接口
    """
    try:
        logger.info(f"开始采集数据: 关键词={keyword}, 角色={character_name}, 最大图片数={max_images}")
        
        # 创建采集器
        collector = KeywordBasedDataCollector()
        
        # 执行采集
        result = collector._process_character(
            keyword,
            character_name or keyword,
            max_images=max_images,
            min_size=min_size
        )
        
        logger.info(f"数据采集完成: 共采集 {result} 张图片")
        
        return {
            "status": "success",
            "message": f"数据采集完成",
            "result": {
                "keyword": keyword,
                "character_name": character_name or keyword,
                "collected_images": result,
                "max_images": max_images
            }
        }
    except Exception as e:
        logger.error(f"数据采集失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据采集失败: {str(e)}")


@app.post("/api/collect/distributed", tags=["数据采集"])
async def distributed_collect(
    keywords: List[str] = Query(..., description="搜索关键词列表"),
    max_images_per_keyword: int = Query(10, description="每个关键词的最大图片数量"),
    workers: int = Query(3, description="工作节点数量")
):
    """
    分布式数据采集接口
    """
    try:
        logger.info(f"开始分布式采集: 关键词数量={len(keywords)}, 工作节点数={workers}")
        
        # 创建任务
        task_ids = []
        for keyword in keywords:
            task_id = distributed_manager.add_task(
                "image_collection",
                keyword=keyword,
                max_images=max_images_per_keyword
            )
            task_ids.append(task_id)
        
        # 启动工作节点
        distributed_manager.start_workers(workers)
        
        return {
            "status": "success",
            "message": "分布式采集任务已启动",
            "result": {
                "keywords": keywords,
                "task_ids": task_ids,
                "workers": workers
            }
        }
    except Exception as e:
        logger.error(f"分布式采集失败: {e}")
        raise HTTPException(status_code=500, detail=f"分布式采集失败: {str(e)}")


@app.get("/api/tasks", tags=["任务管理"])
async def get_tasks():
    """
    获取任务列表
    """
    tasks = distributed_manager.get_tasks()
    return {
        "tasks": tasks
    }


@app.get("/api/tasks/{task_id}", tags=["任务管理"])
async def get_task_status(task_id: str):
    """
    获取任务状态
    """
    status = distributed_manager.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="任务不存在")
    return status


@app.post("/api/tasks/{task_id}/cancel", tags=["任务管理"])
async def cancel_task(task_id: str):
    """
    取消任务
    """
    try:
        result = distributed_manager.cancel_task(task_id)
        return {
            "status": "success" if result else "failed",
            "message": "任务已取消" if result else "任务取消失败"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")


@app.post("/api/image/analyze", tags=["图片处理"])
async def analyze_image(file: UploadFile = File(...)):
    """
    分析图片质量
    """
    try:
        # 读取文件内容
        content = await file.read()
        
        # 分析图片
        score = ImageUtils.calculate_image_quality(content)
        analysis = ImageUtils.analyze_image(content)
        
        return {
            "status": "success",
            "result": {
                "quality_score": score,
                "analysis": analysis
            }
        }
    except Exception as e:
        logger.error(f"图片分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"图片分析失败: {str(e)}")


@app.post("/api/image/batch/analyze", tags=["图片处理"])
async def batch_analyze_images(files: List[UploadFile] = File(...)):
    """
    批量分析图片
    """
    try:
        images = []
        for file in files:
            content = await file.read()
            images.append(content)
        
        # 批量分析
        results = ImageUtils.batch_analyze_images(images)
        
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        logger.error(f"批量图片分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量图片分析失败: {str(e)}")


@app.post("/api/image/deduplicate", tags=["图片处理"])
async def deduplicate_images(files: List[UploadFile] = File(...)):
    """
    图片去重
    """
    try:
        images = []
        for file in files:
            content = await file.read()
            images.append(content)
        
        # 去重
        unique_images = ImageUtils.deduplicate_images(images)
        
        return {
            "status": "success",
            "result": {
                "original_count": len(images),
                "unique_count": len(unique_images)
            }
        }
    except Exception as e:
        logger.error(f"图片去重失败: {e}")
        raise HTTPException(status_code=500, detail=f"图片去重失败: {str(e)}")


@app.get("/api/distributed/workers", tags=["分布式"])
async def get_workers():
    """
    获取工作节点状态
    """
    workers = distributed_manager.get_workers()
    return {
        "workers": workers
    }


@app.post("/api/distributed/workers/scale", tags=["分布式"])
async def scale_workers(count: int = Query(..., description="工作节点数量")):
    """
    调整工作节点数量
    """
    try:
        distributed_manager.start_workers(count)
        return {
            "status": "success",
            "message": f"工作节点已调整为 {count} 个"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调整工作节点失败: {str(e)}")


@app.get("/api/data/sources", tags=["数据源"])
async def get_data_sources():
    """
    获取数据源信息
    """
    try:
        from src.utils.data_source_manager import data_source_manager
        sources = data_source_manager.get_sources()
        return {
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据源失败: {str(e)}")


# 错误处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    全局异常处理器
    """
    logger.error(f"API错误: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": f"服务器内部错误: {str(exc)}"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # 运行API服务
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

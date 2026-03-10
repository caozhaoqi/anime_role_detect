#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API接口服务

提供外部系统集成的API接口
"""

import os
import sys
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
from src.utils.distributed_manager import DistributedManager
from src.core.logging.global_logger import get_logger

logger = get_logger("api_service")

# 创建FastAPI应用
app = FastAPI(
    title="Anime Role Detect API",
    description="二次元角色检测系统API接口",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.get("/api/models", tags=["模型"])
async def get_models():
    """
    获取模型列表
    """
    try:
        # 这里可以从配置或数据库中获取模型列表
        # 暂时返回默认模型
        models = [
            {"name": "default", "path": "", "description": "默认分类模型", "available": True},
            {"name": "augmented_training", "path": "models/augmented_training", "description": "增强训练模型", "available": False},
            {"name": "arona_plana", "path": "models/arona_plana", "description": "阿罗娜普拉娜模型", "available": False},
            {"name": "arona_plana_efficientnet", "path": "models/arona_plana_efficientnet", "description": "EfficientNet模型", "available": False},
            {"name": "arona_plana_resnet18", "path": "models/arona_plana_resnet18", "description": "ResNet18模型", "available": False},
            {"name": "optimized", "path": "models/optimized", "description": "优化模型", "available": False}
        ]
        
        return {"models": models}
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@app.post("/api/classify", tags=["分类"])
async def classify_image(file: UploadFile = File(...), use_model: bool = Query(False, description="是否使用专用模型"), use_attributes: bool = Query(True, description="是否使用属性预测"), model_name: str = Query("default", description="模型名称")):
    """
    分类图片
    """
    try:
        # 读取文件内容
        content = await file.read()
        logger.info(f"接收到文件，大小: {len(content)} 字节")
        logger.info(f"文件类型: {file.content_type}")
        
        # 保存临时文件
        import tempfile
        import os
        # 根据文件类型确定后缀
        suffix = ".jpg"
        if file.content_type == "image/png":
            suffix = ".png"
        elif file.content_type == "image/gif":
            suffix = ".gif"
        elif file.content_type == "image/bmp":
            suffix = ".bmp"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        logger.info(f"临时文件已创建: {temp_path}")
        
        # 检查临时文件是否存在且大小大于0
        if not os.path.exists(temp_path):
            raise ValueError(f"临时文件不存在: {temp_path}")
        if os.path.getsize(temp_path) == 0:
            raise ValueError(f"临时文件为空: {temp_path}")
        logger.info(f"临时文件大小: {os.path.getsize(temp_path)} 字节")
        
        # 尝试使用 PIL 加载图像，验证文件是否是有效的图像
        from PIL import Image
        try:
            pil_img = Image.open(temp_path)
            pil_img.verify()  # 验证图像文件的有效性
            logger.info(f"PIL 加载图像成功，格式: {pil_img.format}, 大小: {pil_img.size}")
        except Exception as e:
            logger.error(f"PIL 加载图像失败: {e}")
            raise ValueError(f"无效的图像文件: {str(e)}")
        
        try:
            # 初始化特征提取和分类模块
            from src.core.feature_extraction.feature_extraction import FeatureExtraction
            from src.core.classification.classification import Classification
            
            extractor = FeatureExtraction()
            classifier = Classification("role_index")
            
            # 打开图像
            with Image.open(temp_path) as img:
                # 调整图像大小
                img = img.resize((224, 224))
                
                # 提取特征
                logger.info("开始提取特征")
                feature = extractor.extract_features(img)
                logger.info(f"特征提取完成，特征维度: {feature.shape}")
                
                # 分类图片
                logger.info("开始分类图片")
                try:
                    role, similarity = classifier.classify(feature)
                    logger.info(f"分类完成，角色: {role}, 相似度: {similarity}")
                except ValueError as e:
                    if "索引尚未构建" in str(e):
                        logger.warning("索引尚未构建，返回默认值")
                        role = "unknown"
                        similarity = 0.0
                    else:
                        raise
            
            # 构建响应
            result = {
                "role": role,
                "similarity": float(similarity),
                "attributes": []  # 暂时返回空属性列表
            }
            
            return result
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"临时文件已删除: {temp_path}")
    except Exception as e:
        logger.error(f"分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"分类失败: {str(e)}")


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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API接口服务

提供外部系统集成的API接口
"""

import os
import sys
import time
import asyncio
from typing import Dict, Any, List, Optional
from PIL import Image

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(project_root))  # 添加父目录，确保能导入src模块
print(f"添加到Python路径: {project_root}")
print(f"Python路径: {sys.path[:3]}")

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils.monitoring_system import MonitoringSystem
from utils.cache_manager import cache_manager
from utils.distributed_manager import DistributedManager
from utils.memory_monitor import init_memory_monitoring, shutdown_memory_monitoring, get_memory_monitor
from utils.image_utils import ImageUtils
from core.logging.global_logger import get_logger
from core.classification.classification import Classification
from core.feature_extraction.feature_extraction import FeatureExtraction
from core.preprocessing.preprocessing import Preprocessing
from core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
from core.keypoint.mediapipe_keypoint_detector import MediaPipeKeypointDetector
from core.detection.multi_role_detection import MultiRoleDetector
from scripts.ai_role_prediction import AIRolePredictor
from config.config_manager import config_manager

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
monitoring_system = MonitoringSystem()  # 系统监控实例
distributed_manager = DistributedManager()  # 分布式管理实例

# 模型实例缓存
extractor = None  # 特征提取器实例
classifiers = {}  # 缓存不同模型的分类器实例
classifiers_max_size = config_manager.get_classifiers_max_size()  # 分类器缓存的最大大小
classifiers_usage = {}  # 记录分类器使用时间，用于LRU缓存
preprocessor = None  # 预处理器实例
tagger = None  # 标签生成器实例
keypoint_detector = None  # 关键点检测器实例

# 新训练的模型缓存
trained_models = {}  # 缓存训练好的分类模型
model_usage = {}  # 记录模型使用频率
model_memory_usage = {
    "mobilenet_v2": 200,  # MB
    "efficientnet_b0": 300,  # MB
    "efficientnet_b3": 600,  # MB
    "resnet50": 1000,  # MB
}

# 最大内存使用阈值（MB）
MAX_MEMORY_USAGE = 6000  # 6GB

# 模型加载时间缓存
model_load_times = {}  # 记录模型加载时间

# 模型推理时间缓存
model_inference_times = {}  # 记录模型推理时间

# 多角色检测器实例
multi_role_detector = None  # 多角色检测器实例
current_model_name = None  # 当前使用的模型名称


@app.on_event("startup")
async def startup_event():
    """
    启动事件
    
    在服务启动时执行的初始化操作，包括：
    1. 启动监控系统和分布式管理系统
    2. 初始化内存监控系统
    3. 初始化模型实例，避免首次请求延迟
    4. 初始化特征提取器、分类器、预处理器、标签生成器和关键点检测器
    """
    logger.info("启动API服务")
    monitoring_system.start()  # 启动监控系统
    distributed_manager.start()  # 启动分布式管理系统
    
    # 初始化内存监控系统
    try:
        init_memory_monitoring()
        logger.info("内存监控系统初始化完成")
    except Exception as e:
        logger.error(f"内存监控系统初始化失败: {e}")
    
    # 延迟初始化模型实例，减少内存使用
    global extractor, classifiers, preprocessor, tagger, keypoint_detector, multi_role_detector
    try:
        logger.info("开始初始化模型...")
        
        # 延迟加载所有模型，只在需要时初始化
        extractor = None
        logger.info("特征提取器将在需要时初始化")
        
        classifiers = {}
        logger.info("分类器将在需要时初始化")
        
        preprocessor = None
        logger.info("预处理器将在需要时初始化")
        
        tagger = None
        logger.info("标签生成器将在需要时初始化")
        
        keypoint_detector = None
        logger.info("关键点检测器将在需要时初始化")
        
        multi_role_detector = None
        logger.info("多角色检测器将在需要时初始化")
        
        logger.info("所有模型初始化完成（延迟加载模式）")
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    关闭事件
    
    在服务关闭时执行的清理操作，包括：
    1. 清理模型实例，释放内存
    2. 停止监控系统和分布式管理系统
    3. 停止内存监控系统
    """
    logger.info("关闭API服务")
    
    # 清理模型实例，释放内存
    global extractor, classifiers, preprocessor, tagger, keypoint_detector, multi_role_detector
    
    if extractor is not None:
        logger.info("清理特征提取器...")
        extractor = None
    
    if classifiers:
        logger.info(f"清理分类器缓存，共 {len(classifiers)} 个实例...")
        classifiers.clear()
    
    if preprocessor is not None:
        logger.info("清理预处理器...")
        preprocessor = None
    
    if tagger is not None:
        logger.info("清理标签生成器...")
        tagger = None
    
    if keypoint_detector is not None:
        logger.info("清理关键点检测器...")
        keypoint_detector = None
    
    if multi_role_detector is not None:
        logger.info("清理多角色检测器...")
        multi_role_detector = None
    
    monitoring_system.stop()  # 停止监控系统
    distributed_manager.stop()  # 停止分布式管理系统
    
    # 停止内存监控系统
    try:
        shutdown_memory_monitoring()
        logger.info("内存监控系统已停止")
    except Exception as e:
        logger.error(f"内存监控系统停止失败: {e}")
    
    logger.info("所有模型实例已清理")


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
        analysis = ImageUtils.analyze_image_content(content)
        
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
        # 动态生成模型列表，根据本地目录存在性设置available
        model_configs = [
            {"name": "default", "path": "", "description": "默认分类模型"},
            {"name": "augmented_training", "path": "models/augmented_training", "description": "增强训练模型"},
            {"name": "arona_plana", "path": "models/arona_plana", "description": "阿罗娜普拉娜模型"},
            {"name": "arona_plana_efficientnet", "path": "models/arona_plana_efficientnet", "description": "EfficientNet模型"},
            {"name": "arona_plana_resnet18", "path": "models/arona_plana_resnet18", "description": "ResNet18模型"},
            {"name": "optimized", "path": "models/optimized", "description": "优化模型"}
        ]
        
        models = []
        
        for config in model_configs:
            # 直接返回所有模型为可用状态
            models.append({
                "name": config["name"],
                "path": config["path"],
                "description": config["description"],
                "available": True
            })
        
        return {"models": models}
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


def validate_image(file, content):
    """
    验证图像有效性
    
    Args:
        file: 上传的文件
        content: 文件内容
    
    Returns:
        temp_path: 临时文件路径
    """
    import tempfile
    import os
    from core.logging.global_logger import get_logger
    logger = get_logger("validate_image")
    
    try:
        file_size = len(content)
        logger.info(f"处理文件，大小: {file_size} 字节")
        logger.info(f"文件类型: {file.content_type}")
        logger.info(f"文件名: {file.filename}")
        
        # 检查文件大小
        max_file_size = config_manager.get_max_file_size()
        if file_size > max_file_size:
            raise ValueError(f"文件大小超过限制 (最大 {max_file_size / 1024 / 1024:.1f}MB)")
        
        # 检查文件类型
        allowed_content_types = config_manager.get_allowed_file_types()
        
        # 当content_type为None时，根据文件扩展名推断
        content_type = file.content_type
        if content_type is None:
            import os
            ext = os.path.splitext(file.filename)[1].lower()
            ext_to_content_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.svg': 'image/svg+xml'
            }
            content_type = ext_to_content_type.get(ext, 'application/octet-stream')
            logger.info(f"根据文件扩展名推断文件类型: {content_type}")
        
        if content_type not in allowed_content_types:
            raise ValueError(f"不支持的文件类型: {content_type}，仅支持 {', '.join(allowed_content_types)}")
        
        # 保存临时文件
        # 直接为SVG文件创建带.svg后缀的临时文件
        logger.info(f"检查文件类型: {content_type}")
        logger.info(f"文件类型是否为SVG: {content_type == 'image/svg+xml'}")
        if content_type == "image/svg+xml":
            logger.info("创建SVG临时文件")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            logger.info(f"SVG临时文件创建成功: {temp_path}")
        else:
            logger.info("创建JPG临时文件")
            # 其他文件类型使用默认后缀
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            logger.info(f"JPG临时文件创建成功: {temp_path}")
        
        logger.info(f"临时文件已创建: {temp_path}")
        
        # 检查临时文件是否存在且大小大于0
        if not os.path.exists(temp_path):
            raise ValueError(f"临时文件不存在: {temp_path}")
        if os.path.getsize(temp_path) == 0:
            raise ValueError(f"临时文件为空: {temp_path}")
        logger.info(f"临时文件大小: {os.path.getsize(temp_path)} 字节")
        
        # 对于SVG文件，直接返回临时文件路径
        if content_type == "image/svg+xml":
            logger.info("跳过 SVG 文件的验证，直接返回临时文件路径")
            return temp_path
        
        # 对于非 SVG 文件，验证图像的有效性
        try:
            pil_img = Image.open(temp_path)
            pil_img.verify()
            logger.info(f"PIL 加载图像成功，格式: {pil_img.format}, 大小: {pil_img.size}")
        except Exception as e:
            logger.error(f"PIL 加载图像失败: {e}")
            raise ValueError(f"无效的图像文件: {str(e)}")
        
        return temp_path
    except Exception as e:
        logger.error(f"validate_image 函数异常: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise

def get_project_root():
    """
    获取项目根目录
    
    Returns:
        project_root: 项目根目录路径
    """
    project_root = config_manager.get_project_root()
    logger.info(f"使用项目根目录: {project_root}")
    return project_root

def get_model_path(model_name, project_root):
    """
    获取模型路径
    
    Args:
        model_name: 模型名称
        project_root: 项目根目录
    
    Returns:
        index_path: 模型索引路径
    """
    # 使用配置管理器获取模型路径
    index_path = config_manager.get_model_path(model_name)
    logger.info(f"使用模型: {model_name}, 索引路径: {index_path}")
    return index_path

def get_current_memory_usage():
    """
    获取当前系统内存使用情况
    
    Returns:
        当前内存使用量（MB）
    """
    import psutil
    memory = psutil.virtual_memory()
    return memory.used / (1024 * 1024)  # 转换为MB

def unload_unused_models(required_memory=0):
    """
    卸载不常用的模型以释放内存
    
    Args:
        required_memory: 需要的内存（MB）
    """
    global trained_models, model_usage, model_memory_usage, MAX_MEMORY_USAGE
    
    current_memory = get_current_memory_usage()
    target_memory = current_memory + required_memory
    
    if target_memory <= MAX_MEMORY_USAGE:
        return
    
    # 计算需要释放的内存
    memory_to_release = target_memory - MAX_MEMORY_USAGE
    
    # 按使用频率排序模型（从低到高）
    sorted_models = sorted(model_usage.items(), key=lambda x: x[1], reverse=False)
    
    released_memory = 0
    models_to_unload = []
    
    for model_name, usage in sorted_models:
        if model_name in trained_models:
            memory_used = model_memory_usage.get(model_name, 0)
            models_to_unload.append(model_name)
            released_memory += memory_used
            
            if released_memory >= memory_to_release:
                break
    
    # 卸载模型
    for model_name in models_to_unload:
        if model_name in trained_models:
            del trained_models[model_name]
            logger.info(f"已卸载模型: {model_name}，释放内存: {model_memory_usage.get(model_name, 0)}MB")
    
    if models_to_unload:
        # 强制垃圾回收
        import gc
        gc.collect()
        logger.info(f"已释放内存: {released_memory}MB，当前内存使用: {get_current_memory_usage():.2f}MB")

def load_trained_model(model_name):
    """
    加载训练好的模型
    
    Args:
        model_name: 模型名称
    
    Returns:
        model: 加载的模型
    """
    import torch
    import torch.nn as nn
    from torchvision import models
    
    global trained_models, model_usage, model_memory_usage
    
    # 检查模型是否已经加载
    if model_name in trained_models:
        # 更新使用频率
        model_usage[model_name] = model_usage.get(model_name, 0) + 1
        logger.info(f"从缓存加载模型: {model_name}，使用频率: {model_usage[model_name]}")
        return trained_models[model_name]
    
    # 模型路径映射
    model_paths = {
        "mobilenet_v2": "models/incremental/model_best.pth",
        "efficientnet_b0": "models/incremental_efficientnet_b0/model_best.pth",
        "efficientnet_b3": "models/incremental_efficientnet_b3/model_best.pth",
        "resnet50": "models/incremental_resnet50/model_best.pth"
    }
    
    # 检查模型是否存在
    if model_name not in model_paths:
        logger.error(f"不支持的模型类型: {model_name}")
        return None
    
    model_path = model_paths[model_name]
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None
    
    # 计算需要的内存
    required_memory = model_memory_usage.get(model_name, 200)
    
    # 卸载不常用的模型以释放内存
    unload_unused_models(required_memory)
    
    try:
        # 加载模型数据（使用map_location和低内存模式）
        model_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        class_to_idx = model_data.get('class_to_idx', {})
        
        # 只加载必要的模型类型
        if model_name == 'mobilenet_v2':
            # 使用更轻量级的模型
            model = models.mobilenet_v2(pretrained=False)
            # 修改分类器，简化结构以提高速度
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(model.classifier[1].in_features, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, len(class_to_idx))
            )
        else:
            # 对于其他模型，使用默认分类器
            logger.warning(f"模型 {model_name} 可能会占用较多内存")
            if model_name == 'efficientnet_b0':
                model = models.efficientnet_b0(pretrained=False)
                # 修改分类器，简化结构以提高速度
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(model.classifier[1].in_features, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, len(class_to_idx))
                )
            elif model_name == 'efficientnet_b3':
                model = models.efficientnet_b3(pretrained=False)
                # 修改分类器，简化结构以提高速度
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(model.classifier[1].in_features, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, len(class_to_idx))
                )
            elif model_name == 'resnet50':
                model = models.resnet50(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
            else:
                logger.error(f"不支持的模型类型: {model_name}")
                return None
        
        # 加载模型权重
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        
        # 保存到缓存
        trained_models[model_name] = (model, class_to_idx)
        model_usage[model_name] = 1  # 初始化使用频率
        logger.info(f"模型 {model_name} 加载完成，类别数: {len(class_to_idx)}")
        logger.info(f"当前内存使用: {get_current_memory_usage():.2f}MB")
        
        return (model, class_to_idx)
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return None

async def get_or_create_classifier(index_path):
    """
    获取或创建分类器
    
    Args:
        index_path: 模型索引路径
    
    Returns:
        classifier: 分类器实例
    """
    import asyncio
    
    # 使用全局模型实例
    global classifiers, classifiers_max_size, classifiers_usage
    
    # 检查分类器缓存大小，如果超过限制，移除最不常用的分类器
    if len(classifiers) >= classifiers_max_size:
        # 找出最不常用的分类器（最早使用的）
        oldest_classifier = min(classifiers_usage.items(), key=lambda x: x[1])[0]
        logger.info(f"分类器缓存达到上限，移除最不常用的分类器: {oldest_classifier}")
        del classifiers[oldest_classifier]
        del classifiers_usage[oldest_classifier]
    
    # 检查缓存中是否已有分类器
    if index_path in classifiers:
        logger.info(f"从缓存获取分类器: {index_path}")
        classifier = classifiers[index_path]
        # 更新使用时间
        classifiers_usage[index_path] = time.time()
    else:
        # 初始化分类器
        logger.info(f"初始化分类器: {index_path}...")
        # 尝试使用Core ML分类器
        try:
            # 使用异步方式初始化Core ML分类器
            classifier = await asyncio.to_thread(Classification.use_coreml)
            logger.info("使用Core ML分类器成功")
        except Exception as e:
            logger.warning(f"Core ML分类器初始化失败: {e}")
            logger.info("回退到默认分类器")
            # 使用异步方式初始化默认分类器，降低阈值以提高识别率
            classifier = await asyncio.to_thread(Classification, index_path, threshold=0.3)
        # 更新缓存
        classifiers[index_path] = classifier
        classifiers_usage[index_path] = time.time()  # 更新使用时间
        logger.info(f"分类器 {index_path} 初始化完成")
    
    # 检查分类器是否成功初始化
    if classifier.index is None:
        logger.error("分类器初始化失败，索引为None")
        return None
    
    # 打印分类器的模块路径，确认使用的是修改后的版本
    logger.info(f"分类器模块路径: {Classification.__module__}")
    logger.info(f"分类器初始化成功，角色数量: {len(classifier.role_mapping)}")
    
    return classifier

async def extract_image_features(temp_path):
    """
    提取图像特征
    
    Args:
        temp_path: 临时文件路径
    
    Returns:
        feature: 提取的特征向量
    """
    import asyncio
    from PIL import Image
    
    # 使用全局模型实例
    global extractor
    
    # 确保提取器已初始化
    if extractor is None:
        # 使用异步方式初始化特征提取器（不使用量化，提高识别准确率）
        # 强制使用PyTorch模式，与构建索引时一致
        extractor = await asyncio.to_thread(FeatureExtraction, quantize=False, use_coreml=False)
        logger.info("特征提取器初始化完成（PyTorch模式）")
    
    # 检查文件是否是SVG格式
    import os
    file_extension = os.path.splitext(temp_path)[1].lower()
    is_svg = file_extension == '.svg'
    
    if is_svg:
        logger.info("处理SVG文件，使用实际SVG内容")
        # 对于SVG文件，使用PIL直接加载
        try:
            img = Image.open(temp_path)
            img = img.convert('RGB')
            # 调整图像大小
            img = img.resize((224, 224))
            
            # 提取特征
            logger.info("开始提取特征")
            # 使用异步方式提取特征
            feature = await asyncio.to_thread(extractor.extract_features, img)
            logger.info(f"特征提取完成，特征维度: {feature.shape}")
            
            return feature
        except Exception as e:
            logger.error(f"处理SVG文件失败: {e}")
            # 创建一个空白图像作为替代
            from PIL import ImageDraw
            logger.info("创建空白图像作为替代")
            img = Image.new('RGB', (224, 224), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Error SVG", fill='black')
            
            # 提取特征
            logger.info("开始提取特征")
            # 使用异步方式提取特征
            feature = await asyncio.to_thread(extractor.extract_features, img)
            logger.info(f"特征提取完成，特征维度: {feature.shape}")
            
            return feature
    else:
        # 对于非SVG文件，正常处理
        try:
            with Image.open(temp_path) as img:
                # 调整图像大小
                img = img.resize((224, 224))
                
                # 提取特征
                logger.info("开始提取特征")
                # 使用异步方式提取特征
                feature = await asyncio.to_thread(extractor.extract_features, img)
                logger.info(f"特征提取完成，特征维度: {feature.shape}")
                
            return feature
        except Exception as e:
            logger.error(f"打开图像失败: {e}")
            # 创建一个空白图像作为替代
            from PIL import ImageDraw
            logger.info("创建空白图像作为替代")
            img = Image.new('RGB', (224, 224), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Error Image", fill='black')
            
            # 提取特征
            logger.info("开始提取特征")
            # 使用异步方式提取特征
            feature = await asyncio.to_thread(extractor.extract_features, img)
            logger.info(f"特征提取完成，特征维度: {feature.shape}")
            
            return feature

async def detect_image_text(temp_path):
    """
    检测图像中的文本
    
    Args:
        temp_path: 临时文件路径
    
    Returns:
        text_detections: 文本检测结果
    """
    
    
    # 检查文件是否是SVG格式
    import os
    file_extension = os.path.splitext(temp_path)[1].lower()
    is_svg = file_extension == '.svg'
    
    # 对于SVG文件，跳过文本检测
    if is_svg:
        logger.info("跳过SVG文件的文本检测")
        return []
    
    # 使用全局预处理器实例
    global preprocessor
    
    text_detections = []
    try:
        if preprocessor is None:
            # 使用异步方式初始化预处理器
            preprocessor = await asyncio.to_thread(Preprocessing)
        # 使用异步方式检测文本
        text_detections = await asyncio.to_thread(preprocessor.detect_text, temp_path)
        logger.info(f"文本检测完成，检测到 {len(text_detections)} 个文本")
    except Exception as e:
        logger.warning(f"文本检测失败: {e}")
    
    return text_detections

async def generate_image_tags(temp_path):
    """
    生成图像标签
    
    Args:
        temp_path: 临时文件路径
    
    Returns:
        attributes: 生成的标签
    """
    
    
    # 使用全局标签生成器实例
    global tagger
    
    attributes = []
    try:
        if tagger is None:
            tagger = WDViTV3Tagger()
            # 使用异步方式加载模型
            await asyncio.to_thread(tagger.load_model)
        # 使用异步方式生成标签
        attributes = await asyncio.to_thread(tagger.generate_tags, temp_path)
        logger.info(f"标签生成完成，生成 {len(attributes)} 个标签")
    except Exception as e:
        logger.warning(f"标签生成失败: {e}")
    
    return attributes

async def classify_image_internal(classifier, feature, attributes):
    """
    分类图像
    
    Args:
        classifier: 分类器实例
        feature: 特征向量
        attributes: 图像标签
    
    Returns:
        role: 角色名称
        similarity: 相似度
    """
    
    role = "unknown"
    similarity = 0.0
    
    # 分类图片
    logger.info("开始分类图片")
    try:
        # 使用异步方式分类，传入标签信息
        role, similarity = await asyncio.to_thread(classifier.classify, feature, tags=attributes)
        logger.info(f"分类完成，角色: {role}, 相似度: {similarity}")
    except ValueError as e:
        if "索引尚未构建" in str(e):
            logger.warning("索引尚未构建，返回默认值")
            role = "unknown"
            similarity = 0.0
        else:
            logger.error(f"分类时发生值错误: {e}")
            raise
    except Exception as e:
        logger.error(f"分类时发生未知错误: {e}")
        # 分类失败时返回默认值，避免整个请求失败
        role = "unknown"
        similarity = 0.0
    
    return role, similarity

async def detect_keypoints(temp_path):
    """
    检测关键点
    
    Args:
        temp_path: 临时文件路径
    
    Returns:
        keypoints: 关键点检测结果
    """
    
    
    # 检查文件是否是SVG格式
    import os
    file_extension = os.path.splitext(temp_path)[1].lower()
    is_svg = file_extension == '.svg'
    
    # 对于SVG文件，跳过关键点检测
    if is_svg:
        logger.info("跳过SVG文件的关键点检测")
        return []
    
    # 使用全局关键点检测器实例
    global keypoint_detector
    
    keypoints = None
    try:
        if keypoint_detector is None:
            # 使用异步方式初始化关键点检测器
            keypoint_detector = await asyncio.to_thread(MediaPipeKeypointDetector)
        # 使用异步方式检测关键点
        keypoints = await asyncio.to_thread(keypoint_detector.detect_keypoints, temp_path)
        logger.info(f"关键点检测完成")
    except Exception as e:
        logger.warning(f"关键点检测失败: {e}")
    
    return keypoints

async def predict_role(attributes, classification_result=None):
    """
    AI 角色预测
    
    Args:
        attributes: 图像标签
        classification_result: 分类器的结果，格式为(角色名称, 相似度)
    
    Returns:
        ai_predicted_role: 预测的角色名称
    """
    
    
    ai_predicted_role = None
    try:
        predictor = AIRolePredictor()
        # 使用异步方式预测角色
        ai_predicted_role = await asyncio.to_thread(predictor.predict_role, attributes)
        logger.info(f"AI 角色预测完成，预测角色: {ai_predicted_role}")
    except Exception as e:
        logger.warning(f"AI 角色预测失败: {e}")
        ai_predicted_role = "未知角色"
    
    # 如果AI预测结果为"未知角色"，使用分类器的结果
    if ai_predicted_role == "未知角色" and classification_result and classification_result[0] != "unknown":
        ai_predicted_role = classification_result[0]
        logger.info(f"使用分类器结果作为AI预测结果: {ai_predicted_role}")
    
    return ai_predicted_role

async def process_single_image(file, model_name, cache_bypass=False):
    """
    处理单个图像
    
    Args:
        file: 上传的文件
        model_name: 模型名称
        cache_bypass: 是否绕过缓存
    
    Returns:
        处理结果
    """
    import os
    import hashlib
    import time
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    
    temp_path = None
    start_time = time.time()
    
    try:
        # 读取文件内容
        content = await file.read()
        process_time = time.time() - start_time
        logger.debug(f"读取文件耗时: {process_time:.4f}秒")
        
        # 生成文件哈希作为缓存键
        file_hash = hashlib.md5(content).hexdigest()
        cache_key = f"image_processing_{file_hash}_{model_name}"
        
        # 尝试从缓存获取结果
        if not cache_bypass:
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"缓存命中，直接返回结果: {file.filename}")
                # 添加文件名
                cached_result["filename"] = file.filename
                cached_result["processing_time"] = time.time() - start_time
                return cached_result
        else:
            logger.info(f"缓存绕过，重新处理: {file.filename}")
        
        # 验证图像
        validate_start = time.time()
        temp_path = validate_image(file, content)
        validate_time = time.time() - validate_start
        logger.debug(f"验证图像耗时: {validate_time:.4f}秒")
        
        # 检查是否使用新训练的模型
        trained_model_names = ["mobilenet_v2", "efficientnet_b0", "efficientnet_b3", "resnet50"]
        
        if model_name in trained_model_names:
            logger.info(f"使用新训练的模型: {model_name}")
            
            # 加载训练好的模型
            model_info = load_trained_model(model_name)
            if model_info is None:
                result = {"role": "unknown", "similarity": 0.0, "attributes": []}
                # 缓存结果
                cache_manager.set(result, cache_key, ttl=3600)
                result["processing_time"] = time.time() - start_time
                return result
            
            model, class_to_idx = model_info
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            
            # 图像预处理
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # 加载图像
            img = Image.open(temp_path).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0)  # 添加批次维度
            
            # 预测
            with torch.no_grad():
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
            
            # 获取预测结果
            role = idx_to_class.get(predicted.item(), "unknown")
            similarity = float(confidence)
            
            # 并行处理文本检测、标签生成
            tasks = []
            
            # 检测文本（非SVG文件）
            if file.content_type != "image/svg+xml":
                tasks.append(detect_image_text(temp_path))
            else:
                logger.info("跳过SVG文件的文本检测")
            
            # 生成标签（非SVG文件）
            if file.content_type != "image/svg+xml":
                tasks.append(generate_image_tags(temp_path))
            else:
                logger.info("跳过SVG文件的标签生成")
            
            # 并行执行
            parallel_start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            parallel_time = time.time() - parallel_start
            logger.debug(f"并行处理耗时: {parallel_time:.4f}秒")
            
            # 处理结果
            text_detections = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else []
            attributes = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []
            
            # 并行处理关键点检测和角色预测
            keypoints = []
            ai_predicted_role = None
            
            if file.content_type != "image/svg+xml":
                tasks = [detect_keypoints(temp_path), predict_role(attributes, (role, similarity))]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                keypoints = results[0] if not isinstance(results[0], Exception) else []
                ai_predicted_role = results[1] if not isinstance(results[1], Exception) else None
            else:
                logger.info("跳过SVG文件的关键点检测")
                ai_predicted_role = await predict_role(attributes, (role, similarity))
            
        else:
            # 使用传统的分类器
            logger.info(f"使用传统分类器: {model_name}")
            
            # 获取项目根目录
            project_root = get_project_root()
            
            # 获取模型路径
            index_path = get_model_path(model_name, project_root)
            
            # 获取或创建分类器
            classifier_start = time.time()
            classifier = await get_or_create_classifier(index_path)
            classifier_time = time.time() - classifier_start
            logger.debug(f"获取分类器耗时: {classifier_time:.4f}秒")
            
            if classifier is None:
                result = {"role": "unknown", "similarity": 0.0, "attributes": []}
                # 缓存结果
                cache_manager.set(result, cache_key, ttl=3600)
                result["processing_time"] = time.time() - start_time
                return result
            
            # 并行处理特征提取、文本检测、标签生成
            tasks = []
            
            # 提取特征
            tasks.append(extract_image_features(temp_path))
            
            # 检测文本（非SVG文件）
            if file.content_type != "image/svg+xml":
                tasks.append(detect_image_text(temp_path))
            else:
                logger.info("跳过SVG文件的文本检测")
            
            # 生成标签（非SVG文件）
            if file.content_type != "image/svg+xml":
                tasks.append(generate_image_tags(temp_path))
            else:
                logger.info("跳过SVG文件的标签生成")
            
            # 并行执行
            parallel_start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            parallel_time = time.time() - parallel_start
            logger.debug(f"并行处理耗时: {parallel_time:.4f}秒")
            
            # 处理结果
            feature = results[0]
            text_detections = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []
            attributes = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else []
            
            # 分类图像
            classify_start = time.time()
            role, similarity = await classify_image_internal(classifier, feature, attributes)
            classify_time = time.time() - classify_start
            logger.debug(f"分类图像耗时: {classify_time:.4f}秒")
            
            # 并行处理关键点检测和角色预测
            keypoints = []
            ai_predicted_role = None
            
            if file.content_type != "image/svg+xml":
                tasks = [detect_keypoints(temp_path), predict_role(attributes, (role, similarity))]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                keypoints = results[0] if not isinstance(results[0], Exception) else []
                ai_predicted_role = results[1] if not isinstance(results[1], Exception) else None
            else:
                logger.info("跳过SVG文件的关键点检测")
                ai_predicted_role = await predict_role(attributes, (role, similarity))
        
        # 构建响应
        result = {
            "filename": file.filename,
            "role": role,
            "similarity": float(similarity),
            "attributes": attributes,
            "ai_predicted_role": ai_predicted_role,
            "processing_time": time.time() - start_time
        }
        
        # 添加文本检测结果（如果有）
        if text_detections:
            result["text_detections"] = text_detections
            logger.info(f"返回文本检测结果: {len(text_detections)} 个文本")
        
        # 添加关键点检测结果（如果有）
        if keypoints:
            result["keypoints"] = keypoints
            logger.info("返回关键点检测结果")
        
        # 缓存结果，根据相似度调整TTL
        ttl = 3600 if similarity > 0.8 else 1800
        cache_manager.set(result, cache_key, ttl=ttl)
        logger.info(f"结果已缓存: {cache_key}, TTL: {ttl}秒")
        
        return result
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"临时文件已删除: {temp_path}")
            except Exception as e:
                logger.error(f"删除临时文件失败: {e}")
        
        # 强制垃圾回收
        import gc
        gc.collect()
        logger.debug("执行垃圾回收，释放内存")


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "message": "API服务运行正常"}

@app.post("/api/classify", tags=["分类"])
async def classify_image(file: UploadFile = File(...), use_model: bool = Form(False, description="是否使用专用模型"), use_attributes: bool = Form(True, description="是否使用属性预测"), model_name: str = Form("default", description="模型名称"), cache_bypass: bool = Form(False, description="是否绕过缓存")):
    """
    分类图片
    
    自动检测图像中的角色数量，根据角色数量选择单角色或多角色检测
    """
    start_time = time.time()
    try:
        # 增加请求限流
        active_tasks = monitoring_system.monitors['task'].get_active_tasks()
        if active_tasks > 5:
            raise HTTPException(status_code=429, detail="请求过多，请稍后再试")
        
        # 读取文件内容
        content = await file.read()
        
        # 验证图像
        temp_path = validate_image(file, content)
        
        # 初始化多角色检测器（根据模型名称）
        global multi_role_detector
        global current_model_name
        
        if multi_role_detector is None or current_model_name != model_name:
            from core.detection.multi_role_detection import MultiRoleDetector
            multi_role_detector = MultiRoleDetector(model_name=model_name)
            current_model_name = model_name
            logger.info(f"多角色检测器初始化完成，使用模型: {model_name}")
        
        # 检测多个角色
        detected_roles = multi_role_detector.detect_roles(temp_path)
        logger.info(f"自动检测到 {len(detected_roles)} 个角色")
        
        # 检测文本
        text_detections = []
        if file.content_type != "image/svg+xml":
            text_detections = await detect_image_text(temp_path)
            logger.info(f"文本检测完成，检测到 {len(text_detections)} 个文本")
        else:
            logger.info("跳过SVG文件的文本检测")
        
        # 根据角色数量选择检测模式
        if len(detected_roles) > 1:
            logger.info("检测到多个角色，使用多角色检测")
            # 构建多角色响应
            response = {
                "filename": file.filename,
                "roles": detected_roles,
                "text_detections": text_detections,
                "processing_time": time.time() - start_time,
                "detection_mode": "multi_role"
            }
        else:
            logger.info("检测到单个角色或未检测到角色，使用单角色检测")
            # 重置文件指针
            await file.seek(0)
            # 使用单角色检测
            result = await process_single_image(file, model_name, cache_bypass)
            result["detection_mode"] = "single_role"
            response = result
        
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"临时文件已删除: {temp_path}")
            except Exception as e:
                logger.error(f"删除临时文件失败: {e}")
        
        # 强制垃圾回收
        import gc
        gc.collect()
        logger.debug("执行垃圾回收，释放内存")
        
        response_time = time.time() - start_time
        # 更新网络监控统计信息
        monitoring_system.monitors['network'].update_request_stats(True, response_time)
        return response
    except Exception as e:
        response_time = time.time() - start_time
        # 更新网络监控统计信息
        monitoring_system.monitors['network'].update_request_stats(False, response_time)
        logger.error(f"分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"分类失败: {str(e)}")


@app.post("/api/classify/multi-role", tags=["分类"])
async def classify_multi_role(file: UploadFile = File(...), cache_bypass: bool = Form(False, description="是否绕过缓存"), model_name: str = Form("efficientnet_b0", description="模型名称")):
    """
    多角色识别
    
    检测图像中的多个角色，并对每个角色进行分类
    """
    start_time = time.time()
    try:
        # 增加请求限流
        active_tasks = monitoring_system.monitors['task'].get_active_tasks()
        if active_tasks > 5:
            raise HTTPException(status_code=429, detail="请求过多，请稍后再试")
        
        # 读取文件内容
        content = await file.read()
        
        # 验证图像
        temp_path = validate_image(file, content)
        
        # 初始化多角色检测器（根据模型名称）
        global multi_role_detector
        global current_model_name
        
        if multi_role_detector is None or current_model_name != model_name:
            multi_role_detector = MultiRoleDetector(model_name=model_name)
            current_model_name = model_name
            logger.info(f"多角色检测器初始化完成，使用模型: {model_name}")
        
        # 检测多个角色
        results = multi_role_detector.detect_roles(temp_path)
        
        # 构建响应
        response = {
            "filename": file.filename,
            "roles": results,
            "processing_time": time.time() - start_time
        }
        
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"临时文件已删除: {temp_path}")
            except Exception as e:
                logger.error(f"删除临时文件失败: {e}")
        
        # 强制垃圾回收
        import gc
        gc.collect()
        logger.debug("执行垃圾回收，释放内存")
        
        return response
    except Exception as e:
        response_time = time.time() - start_time
        # 更新网络监控统计信息
        monitoring_system.monitors['network'].update_request_stats(False, response_time)
        logger.error(f"多角色识别失败: {e}")
        raise HTTPException(status_code=500, detail=f"多角色识别失败: {str(e)}")


@app.post("/api/classify/batch", tags=["分类"])
async def classify_batch(files: List[UploadFile] = File(...), model_name: str = Form("default", description="模型名称")):
    """
    批量分类图片
    """
    start_time = time.time()
    import asyncio
    
    try:
        # 增加请求限流
        active_tasks = monitoring_system.monitors['task'].get_active_tasks()
        if active_tasks > 5:
            raise HTTPException(status_code=429, detail="请求过多，请稍后再试")
        
        logger.info(f"接收到批量请求，文件数量: {len(files)}")
        
        # 限制批量处理的文件数量
        max_batch_files = config_manager.get_max_batch_files()
        if len(files) > max_batch_files:
            raise HTTPException(status_code=400, detail=f"批量处理的文件数量不能超过{max_batch_files}个")
        
        # 分批处理文件，每批最多4个文件，提高并发处理能力
        batch_size = 4
        batches = [files[i:i+batch_size] for i in range(0, len(files), batch_size)]
        processed_results = []
        success_count = 0
        
        for batch in batches:
            # 并行处理当前批次的文件
            tasks = [process_single_image(file, model_name) for file in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # 处理错误，提供更详细的错误信息
                    error_message = str(result)
                    if isinstance(result, HTTPException):
                        error_message = f"HTTP错误: {result.detail}"
                    elif isinstance(result, FileTypeError):
                        error_message = f"文件类型错误: {str(result)}"
                    
                    processed_results.append({
                        "filename": batch[i].filename,
                        "error": error_message
                    })
                    logger.error(f"处理文件 {batch[i].filename} 时出错: {error_message}")
                else:
                    processed_results.append(result)
                    success_count += 1
            
            # 每批处理完成后，暂停一下，避免系统过载
            await asyncio.sleep(0.3)
        
        response_time = time.time() - start_time
        # 更新网络监控统计信息
        monitoring_system.monitors['network'].update_request_stats(True, response_time)
        
        logger.info(f"批量处理完成，成功: {success_count}, 失败: {len(files) - success_count}")
        
        return {
            "total": len(files),
            "success": success_count,
            "failed": len(files) - success_count,
            "results": processed_results
        }
    except Exception as e:
        response_time = time.time() - start_time
        # 更新网络监控统计信息
        monitoring_system.monitors['network'].update_request_stats(False, response_time)
        logger.error(f"批量分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量分类失败: {str(e)}")


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


# 监控相关API
@app.get("/api/monitoring", tags=["监控"])
async def get_monitoring_data():
    """
    获取监控数据
    """
    stats = monitoring_system.get_all_stats()
    return {
        "status": "success",
        "data": stats
    }


@app.get("/api/monitoring/system", tags=["监控"])
async def get_system_monitoring():
    """
    获取系统监控数据
    """
    system_stats = monitoring_system.monitors['system'].get_stats()
    system_data = monitoring_system.monitors['system'].get_data(limit=20)
    return {
        "status": "success",
        "stats": system_stats,
        "data": system_data
    }


@app.get("/api/monitoring/network", tags=["监控"])
async def get_network_monitoring():
    """
    获取网络监控数据
    """
    network_stats = monitoring_system.monitors['network'].get_stats()
    network_data = monitoring_system.monitors['network'].get_data(limit=20)
    return {
        "status": "success",
        "stats": network_stats,
        "data": network_data
    }


@app.get("/api/monitoring/task", tags=["监控"])
async def get_task_monitoring():
    """
    获取任务监控数据
    """
    task_stats = monitoring_system.monitors['task'].get_stats()
    task_data = monitoring_system.monitors['task'].get_data(limit=20)
    return {
        "status": "success",
        "stats": task_stats,
        "data": task_data
    }


@app.get("/api/monitoring/memory", tags=["监控"])
async def get_memory_monitoring():
    """
    获取内存监控数据
    """
    memory_monitor = get_memory_monitor()
    if memory_monitor:
        memory_stats = memory_monitor.get_stats()
        memory_data = memory_monitor.get_data(limit=20)
        return {
            "status": "success",
            "stats": memory_stats,
            "data": memory_data
        }
    else:
        return {
            "status": "error",
            "message": "内存监控系统未初始化"
        }


if __name__ == "__main__":
    import uvicorn
    
    # 获取API配置
    api_config = config_manager.get_api_config()
    host = api_config.get("host", "127.0.0.1")
    port = api_config.get("port", 8000)
    reload = api_config.get("reload", False)
    
    # 运行API服务
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload
    )

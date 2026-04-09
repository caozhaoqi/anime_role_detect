#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型服务API

提供模型预测和特征提取功能
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

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.logging.global_logger import get_logger
from core.classification.classification import Classification
from core.feature_extraction.feature_extraction import FeatureExtraction
from core.preprocessing.preprocessing import Preprocessing
from core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
from config.config_manager import config_manager

logger = get_logger("model_service")

# 创建FastAPI应用
app = FastAPI(
    title="Model Service API",
    description="模型服务API，提供模型预测和特征提取功能",
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
classifier = None  # 分类器实例
feature_extractor = None  # 特征提取器实例
preprocessor = None  # 预处理器实例
tagger = None  # 标签生成器实例

# 初始化模型
async def init_models():
    """
    初始化模型实例
    """
    global classifier, feature_extractor, preprocessor, tagger
    
    try:
        # 延迟加载模型，只初始化预处理器
        preprocessor = Preprocessing()
        logger.info("预处理器初始化完成")
        
        # 其他模型在需要时才初始化
        logger.info("模型服务启动完成，其他模型将在需要时自动初始化")
        
    except Exception as e:
        logger.error(f"初始化模型失败: {e}")

# 启动事件
@app.on_event("startup")
async def startup_event():
    """启动事件"""
    logger.info("启动模型服务")
    await init_models()
    logger.info("模型服务启动完成")

# 健康检查
@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "message": "模型服务运行正常"}

# 验证图像
async def validate_image(file: UploadFile) -> str:
    """
    验证图像
    
    Args:
        file: 上传的文件
    
    Returns:
        临时文件路径
    """
    try:
        # 读取文件内容
        content = await file.read()
        
        # 验证文件类型
        if file.content_type is None or not file.content_type.startswith("image/"):
            # 尝试从文件名推断文件类型
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
            if ext not in ext_to_content_type:
                raise HTTPException(status_code=400, detail="文件类型错误，只支持图像文件")
        
        # 验证文件大小
        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="文件大小超过限制，最大支持10MB")
        
        # 保存为临时文件
        temp_path = f"temp_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # 验证图像是否有效
        try:
            img = Image.open(temp_path)
            img.verify()
        except Exception as e:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"图像文件无效: {e}")
        
        return temp_path
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"验证图像失败: {e}")

# 模型预测
@app.post("/api/model/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form("default"),
    use_attributes: bool = Form(True)
):
    """
    模型预测
    
    Args:
        file: 上传的图像文件
        model_name: 模型名称
        use_attributes: 是否使用属性预测
    
    Returns:
        预测结果
    """
    temp_path = None
    try:
        # 验证图像
        temp_path = await validate_image(file)
        
        # 预处理图像
        if preprocessor:
            processed_result = preprocessor.process(temp_path)
            # 预处理器返回的是元组 (normalized_img, boxes)
            if isinstance(processed_result, tuple):
                processed_image = processed_result[0]
            else:
                processed_image = processed_result
        else:
            processed_image = temp_path
        
        # 延迟初始化特征提取器
        global feature_extractor
        if not feature_extractor:
            logger.info("初始化特征提取器...")
            try:
                feature_extractor = FeatureExtraction()
                logger.info("特征提取器初始化完成")
            except Exception as e:
                logger.error(f"特征提取器初始化失败: {e}")
                raise HTTPException(status_code=503, detail=f"特征提取器初始化失败: {e}")
        
        # 提取特征
        try:
            feature = feature_extractor.extract_features(processed_image)
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            raise HTTPException(status_code=503, detail=f"特征提取失败: {e}")
        
        # 生成标签
        attributes = []
        if use_attributes:
            global tagger
            if not tagger:
                logger.info("初始化标签生成器...")
                tagger = WDViTV3Tagger()
                logger.info("标签生成器初始化完成")
            attributes = tagger.generate_tags(processed_image)
        
        # 延迟初始化分类器
        global classifier
        if not classifier:
            logger.info("初始化分类器...")
            model_name = "mobilenet_v2"  # 使用默认模型名称
            index_path = f"./models/{model_name}"  # 使用默认模型路径
            classifier = Classification(index_path)
            logger.info(f"分类器初始化完成，模型: {model_name}")
        
        # 检查分类器是否有索引
        if classifier.index is not None:
            # 分类图像
            role, similarity = classifier.classify(feature, 5, attributes)  # 减少top_k值，提高速度
        else:
            # 如果没有索引，返回特征向量和未知角色
            logger.warning("分类器索引不存在，返回特征向量和未知角色")
            role = "unknown"
            similarity = 0.0
        
        # 构建响应
        result = {
            "role": role,
            "similarity": float(similarity),
            "attributes": attributes,
            "feature": feature.tolist() if hasattr(feature, 'tolist') else feature  # 返回特征向量供后端使用
        }
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型预测失败: {e}")
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"删除临时文件失败: {e}")

# 特征提取
@app.post("/api/model/extract")
async def extract_features(
    file: UploadFile = File(...)
):
    """
    特征提取
    
    Args:
        file: 上传的图像文件
    
    Returns:
        特征向量
    """
    temp_path = None
    try:
        # 验证图像
        temp_path = await validate_image(file)
        
        # 预处理图像
        if preprocessor:
            processed_image = preprocessor.process(temp_path)
        else:
            processed_image = temp_path
        
        # 延迟初始化特征提取器
        global feature_extractor
        if not feature_extractor:
            logger.info("初始化特征提取器...")
            feature_extractor = FeatureExtraction()
            logger.info("特征提取器初始化完成")
        
        # 提取特征
        feature = feature_extractor.extract_features(processed_image)
        
        # 构建响应
        result = {
            "feature": feature.tolist() if hasattr(feature, "tolist") else list(feature)
        }
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"特征提取失败: {e}")
        raise HTTPException(status_code=500, detail=f"特征提取失败: {e}")
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"删除临时文件失败: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        timeout_keep_alive=60,
        limit_concurrency=5,
        limit_max_requests=500
    )

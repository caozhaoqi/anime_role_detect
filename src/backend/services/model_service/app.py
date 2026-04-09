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
        # 初始化预处理器
        preprocessor = Preprocessing()
        logger.info("预处理器初始化完成")
        
        # 初始化特征提取器
        feature_extractor = FeatureExtraction()
        logger.info("特征提取器初始化完成")
        
        # 初始化分类器
        model_name = "mobilenet_v2"  # 使用默认模型名称
        index_path = f"./models/{model_name}"  # 使用默认模型路径
        classifier = Classification(index_path)
        logger.info(f"分类器初始化完成，模型: {model_name}")
        
        # 初始化标签生成器
        tagger = WDViTV3Tagger()
        logger.info("标签生成器初始化完成")
        
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
        if not file.content_type.startswith("image/"):
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
            processed_image = preprocessor.process(temp_path)
        else:
            processed_image = temp_path
        
        # 提取特征
        if feature_extractor:
            feature = feature_extractor.extract_features(processed_image)
        else:
            raise HTTPException(status_code=500, detail="特征提取器未初始化")
        
        # 生成标签
        attributes = []
        if use_attributes and tagger:
            attributes = tagger.generate_tags(processed_image)
        
        # 分类图像
        if classifier:
            role, similarity = classifier.classify(feature, attributes)
        else:
            raise HTTPException(status_code=500, detail="分类器未初始化")
        
        # 构建响应
        result = {
            "role": role,
            "similarity": float(similarity),
            "attributes": attributes
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
        
        # 提取特征
        if feature_extractor:
            feature = feature_extractor.extract_features(processed_image)
        else:
            raise HTTPException(status_code=500, detail="特征提取器未初始化")
        
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
    uvicorn.run(app, host="0.0.0.0", port=8001)

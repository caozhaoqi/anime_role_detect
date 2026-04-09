#!/usr/bin/env python3
"""
模型服务主文件
提供模型预测、特征提取等功能
"""
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.preprocessing.preprocessing import Preprocessing
from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
from src.core.classification.classification import Classification
from src.utils.log_manager import LogManager

# 初始化日志管理器
log_manager = LogManager(log_dir='logs', log_level='INFO')
logger = log_manager.get_logger("model_service")

# 初始化FastAPI应用
app = FastAPI(
    title="Model Service",
    description="Anime Role Detect Model Service",
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

# 全局变量
preprocessor = None  # 预处理器实例
feature_extractor = None  # 特征提取器实例
tagger = None  # 标签生成器实例

# 模型实例缓存
model_cache = {}

# 初始化模型
async def init_models():
    """初始化模型"""
    global preprocessor, feature_extractor, tagger
    
    try:
        # 初始化预处理器
        preprocessor = Preprocessing()
        logger.info("预处理器初始化完成")
        
        # 其他模型在需要时自动初始化
        logger.info("模型服务启动完成，其他模型将在需要时自动初始化")
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")

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
    return {"status": "healthy", "service": "Model Service"}

# 模型预测
@app.post("/api/model/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form("mobilenet_v2"),
    use_attributes: bool = Form(True)
):
    """
    模型预测
    
    Args:
        file: 上传的图像文件
        model_name: 模型名称
        use_attributes: 是否使用属性
    
    Returns:
        预测结果
    """
    global preprocessor, feature_extractor, tagger
    
    temp_path = None
    
    try:
        # 读取文件内容
        content = await file.read()
        
        # 保存临时文件
        temp_path = f"temp_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # 预处理图像
        if preprocessor is None:
            preprocessor = Preprocessing()
        
        processed_image, _ = preprocessor.process(temp_path)
        
        # 初始化特征提取器
        if feature_extractor is None:
            logger.info("初始化特征提取器...")
            try:
                feature_extractor = FeatureExtraction()
                logger.info("特征提取器初始化完成")
            except Exception as e:
                logger.error(f"特征提取器初始化失败: {e}")
                raise HTTPException(status_code=500, detail=f"特征提取器初始化失败: {e}")
        
        # 提取特征
        feature = feature_extractor.extract(processed_image)
        
        # 初始化标签生成器
        if tagger is None:
            logger.info("初始化标签生成器...")
            try:
                tagger = WDVitV3Tagger()
                logger.info("标签生成器初始化完成")
            except Exception as e:
                logger.error(f"标签生成器初始化失败: {e}")
                tagger = None
        
        # 生成属性标签
        attributes = []
        if use_attributes and tagger:
            try:
                attributes = tagger.generate_tags(processed_image)
            except Exception as e:
                logger.error(f"标签生成失败: {e}")
        
        # 初始化分类器
        if model_name not in model_cache:
            logger.info(f"初始化分类器: {model_name}")
            # 检查模型路径
            index_path = f"./models/{model_name}"
            if not os.path.exists(index_path):
                # 尝试使用默认模型路径
                index_path = f"./models/mobilenet_v2"
                model_name = "mobilenet_v2"  # 使用默认模型名称
                logger.warning(f"模型路径不存在，使用默认模型: {model_name}")
            
            classifier = Classification(index_path)
            model_cache[model_name] = classifier
            logger.info(f"分类器初始化完成，模型: {model_name}")
        else:
            classifier = model_cache[model_name]
        
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
        logger.info(f"准备返回结果: role={role}, similarity={similarity}, feature类型={type(feature)}, feature长度={len(feature) if hasattr(feature, '__len__') else 'N/A'}")
        result = {
            "role": role,
            "similarity": float(similarity),
            "attributes": attributes,
            "feature": feature.tolist() if hasattr(feature, 'tolist') else feature  # 返回特征向量供后端使用
        }
        logger.info(f"返回结果: {result}")
        
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
    global preprocessor, feature_extractor
    
    temp_path = None
    
    try:
        # 读取文件内容
        content = await file.read()
        
        # 保存临时文件
        temp_path = f"temp_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # 预处理图像
        if preprocessor is None:
            preprocessor = Preprocessing()
        
        processed_image, _ = preprocessor.process(temp_path)
        
        # 初始化特征提取器
        if feature_extractor is None:
            logger.info("初始化特征提取器...")
            try:
                feature_extractor = FeatureExtraction()
                logger.info("特征提取器初始化完成")
            except Exception as e:
                logger.error(f"特征提取器初始化失败: {e}")
                raise HTTPException(status_code=500, detail=f"特征提取器初始化失败: {e}")
        
        # 提取特征
        feature = feature_extractor.extract(processed_image)
        
        # 构建响应
        result = {
            "feature": feature.tolist() if hasattr(feature, 'tolist') else feature
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

# 主函数
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模型服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务主机")
    parser.add_argument("--port", type=int, default=8001, help="服务端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    
    args = parser.parse_args()
    
    # 启动服务
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        timeout_keep_alive=30,
        limit_concurrency=10,
        limit_max_requests=1000
    )
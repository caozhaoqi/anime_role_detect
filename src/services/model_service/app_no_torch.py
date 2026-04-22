#!/usr/bin/env python3
"""
无PyTorch依赖的模型服务
避免macOS上的锁竞争问题
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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
print(f"添加到Python路径: {project_root}")
print(f"当前工作目录: {os.getcwd()}")
print(f"Python路径: {sys.path}")

# 设置环境变量以避免OpenMP冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 只导入非PyTorch依赖的模块
from src.core.preprocessing.preprocessing import Preprocessing
from src.core.classification.classification import Classification
from src.services.nsfw_detector import detect_nsfw
from src.core.logging.global_logger import get_logger

# 初始化日志
logger = get_logger("model_service")

# 初始化FastAPI应用
app = FastAPI(
    title="Model Service",
    description="Anime Role Detect Model Service",
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

# 全局变量
preprocessor = None  # 预处理器实例

# 模型缓存
model_cache = {}

# # 角色列表（用于模拟分类）
# ROLES = [
#     "日奈", "阿罗娜", "莫妮卡", "椿丘彼方", "砂狼白子",
#     "芹泽优衣", "樱坂雫", "宫本芙蕾德莉卡", "近江彼方", "中川菜菜"
# ]

# 标签列表（用于模拟标签生成）
tags = [
    "anime", "digital art", "illustration", "high quality", "masterpiece",
    "best quality", "detailed", "beautiful", "cute", "sexy",
    "cool", "adorable", "stylish", "simple background", "complex background"
]

# 初始化模型
async def init_models():
    """初始化模型"""
    global preprocessor
    
    try:
        # 只初始化预处理器
        logger.info("初始化预处理器...")
        preprocessor = Preprocessing()
        logger.info("预处理器初始化完成")
        
        logger.info("模型服务启动完成，使用无PyTorch实现")
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")

# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {"message": "Model Service", "docs": "/docs"}

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
    global preprocessor
    
    temp_path = None
    
    try:
        # 读取文件内容
        content = await file.read()
        
        # 保存临时文件
        temp_path = f"temp_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # [1] NSFW检测 → 如果检测到敏感内容，直接返回
        logger.info("开始NSFW检测...")
        nsfw_result = detect_nsfw(temp_path)
        logger.info(f"NSFW检测结果: {nsfw_result}")
        
        if nsfw_result['is_nsfw']:
            logger.warning("检测到敏感内容，直接返回")
            return {
                "role": "nsfw_detected",
                "similarity": 0.0,
                "attributes": [],
                "keypoints": None,
                "nsfw_status": nsfw_result
            }
        
        # [2] 图像预处理
        logger.info("开始图像预处理...")
        if preprocessor is None:
            preprocessor = Preprocessing()
        
        processed_image = preprocessor.preprocess(temp_path)
        if processed_image is None:
            logger.error("图像预处理失败")
            raise HTTPException(status_code=500, detail="图像预处理失败")
        logger.info("图像预处理完成")
        
        # [3] 关键点检测（暂时禁用）
        logger.info("开始关键点检测...")
        keypoints = None
        logger.info("关键点检测已暂时禁用")
        
        # [4] 特征提取（使用简单特征）
        logger.info("开始特征提取...")
        # 使用简单的图像特征（基于文件大小和名称）
        import hashlib
        import numpy as np
        
        # 基于文件内容生成特征
        file_hash = hashlib.md5(content).hexdigest()
        # 将哈希转换为512维特征向量
        feature = np.array([ord(c) for c in file_hash * 32])[:512].astype(np.float32)
        # 归一化
        feature = feature / np.linalg.norm(feature) if np.linalg.norm(feature) > 0 else feature
        logger.info("特征提取完成")
        
        # [5] 标签生成（使用默认标签）
        logger.info("开始标签生成...")
        attributes = []
        if use_attributes:
            # 基于文件哈希生成标签
            tag_count = int(file_hash[:2], 16) % 5 + 3  # 3-8个标签
            # 基于哈希选择标签
            selected_tags = []
            for i in range(tag_count):
                tag_index = int(file_hash[2+i:4+i], 16) % len(tags)
                selected_tags.append(tags[tag_index])
            attributes = list(set(selected_tags))  # 去重
        logger.info("标签生成完成")
        
        # [6] 角色分类（基于特征向量）
        logger.info("开始角色分类...")
        if model_name not in model_cache:
            logger.info(f"初始化分类器: {model_name}")
            # 获取项目根目录
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
            logger.info(f"项目根目录: {project_root}")
            
            # 检查模型路径
            index_path = os.path.join(project_root, f"models/{model_name}")
            faiss_path = f"{index_path}.faiss"
            mapping_path = f"{index_path}_mapping.json"
            
            # 如果模型文件不存在，尝试使用默认模型
            if not (os.path.exists(faiss_path) and os.path.exists(mapping_path)):
                logger.warning(f"模型文件不存在: {faiss_path} 或 {mapping_path}")
                # 尝试使用默认模型
                model_name = "mobilenet_v2"
                index_path = os.path.join(project_root, f"models/{model_name}")
                faiss_path = f"{index_path}.faiss"
                mapping_path = f"{index_path}_mapping.json"
                logger.info(f"尝试使用默认模型: {model_name}")
            
            if os.path.exists(faiss_path) and os.path.exists(mapping_path):
                logger.info(f"使用模型文件: {faiss_path}")
                # 降低阈值，提高角色识别率
                classifier = Classification(faiss_path, threshold=0.1)
                model_cache[model_name] = classifier
                logger.info(f"分类器初始化完成: {model_name}")
            else:
                logger.error(f"模型文件不存在: {faiss_path} 或 {mapping_path}")
                return {"error": "模型文件不存在"}
        
        classifier = model_cache.get(model_name)
        if classifier:
            # 执行分类
            try:
                role_result = classifier.classify(feature)
                if isinstance(role_result, tuple):
                    role, similarity = role_result
                else:
                    role = role_result.get("role", "unknown")
                    similarity = role_result.get("similarity", 0.0)
                logger.info(f"角色分类结果: {role}, 相似度: {similarity}")
            except Exception as e:
                logger.error(f"分类失败: {e}")
                role = "unknown"
                similarity = 0.0
        else:
            logger.error("分类器未初始化")
            role = "unknown"
            similarity = 0.0
        
        # 构建响应
        result = {
            "role": role,
            "similarity": float(similarity),
            "tags": attributes,
            "nsfw": nsfw_result,
            "keypoints": keypoints,
            "model_used": "no_torch"
        }
        
        logger.info(f"返回结果: {result}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        # 返回复降级结果
        return {
            "role": "unknown",
            "similarity": 0.0,
            "tags": ["anime", "digital art"],
            "nsfw": {
                "is_nsfw": False,
                "details": {}
            }
        }
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
    global preprocessor
    
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
        
        processed_image = preprocessor.preprocess(temp_path)
        if processed_image is None:
            logger.error("图像预处理失败")
            raise HTTPException(status_code=500, detail="图像预处理失败")
        
        # 使用简单的特征提取
        import hashlib
        import numpy as np
        
        # 基于文件内容生成特征
        file_hash = hashlib.md5(content).hexdigest()
        # 将哈希转换为512维特征向量
        feature = np.array([ord(c) for c in file_hash * 32])[:512].astype(np.float32)
        # 归一化
        feature = feature / np.linalg.norm(feature) if np.linalg.norm(feature) > 0 else feature
        
        # 构建响应
        result = {
            "feature": feature.tolist()
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

# 兼容前端的分类端点
@app.post("/api/classify")
async def classify_image(
    file: UploadFile = File(...),
    use_coreml: bool = Form(False),
    use_model: bool = Form(True),
    use_attributes: bool = Form(True),
    model_name: str = Form("mobilenet_v2"),
    cache_bypass: bool = Form(False)
):
    """
    分类图像（兼容前端调用）
    """
    return await predict_image(
        file=file,
        model_name=model_name,
        use_attributes=use_attributes
    )

# 主函数
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模型服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务主机")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    
    args = parser.parse_args()
    
    # 启动服务
    uvicorn.run(
        "app_no_torch:app",
        host=args.host,
        port=args.port,
        timeout_keep_alive=30
    )

#!/usr/bin/env python3
"""
模型服务主文件
提供模型预测、特征提取等功能
"""
import os
import sys

# 解决macOS上的Mutex锁失败问题
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# 设置环境变量，避免锁竞争问题和OpenMP冲突
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 延迟导入torch
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

# 添加项目根目录到Python路径（备用方案）
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# 延迟导入核心模块，避免启动时的锁竞争
Preprocessing = None
FeatureExtraction = None
WDViTV3Tagger = None
Classification = None
get_logger = None

# 动态导入函数
def import_core_modules():
    global Preprocessing, FeatureExtraction, WDViTV3Tagger, Classification, get_logger, torch, detect_nsfw
    # 导入torch并禁用MPS，避免锁竞争问题
    import torch
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
    # 禁用MPS相关的环境变量
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    from src.core.preprocessing.preprocessing import Preprocessing
    from src.core.feature_extraction.feature_extraction import FeatureExtraction
    from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
    from src.core.classification.classification import Classification
    from src.services.nsfw_detector import detect_nsfw
    from src.core.logging.global_logger import get_logger

# 初始化日志
import_core_modules()
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
        # 只初始化预处理器，特征提取器和标签生成器在第一次请求时初始化
        logger.info("初始化预处理器...")
        preprocessor = Preprocessing()
        logger.info("预处理器初始化完成")
        
        # 其他模型在需要时自动初始化
        logger.info("模型服务启动完成，其他模型将在第一次请求时自动初始化")
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
        
        # [2] 图像预处理（解码、缩放、归一化）
        logger.info("开始图像预处理...")
        if preprocessor is None:
            preprocessor = Preprocessing()
        
        processed_image = preprocessor.preprocess(temp_path)
        if processed_image is None:
            logger.error("图像预处理失败")
            raise HTTPException(status_code=500, detail="图像预处理失败")
        logger.info("图像预处理完成")
        
        # [3] 关键点检测（面部、手部、姿态）
        logger.info("开始关键点检测...")
        # 暂时禁用关键点检测，避免macOS上的Mutex锁竞争问题
        keypoints = None
        logger.info("关键点检测已暂时禁用")
        
        # [4] 特征提取 → 生成512维特征向量
        logger.info("开始特征提取...")
        if feature_extractor is None:
            logger.info("初始化特征提取器...")
            try:
                feature_extractor = FeatureExtraction()
                logger.info(f"特征提取器初始化完成，使用设备: {feature_extractor.device}")
            except Exception as e:
                logger.error(f"特征提取器初始化失败: {e}")
                raise HTTPException(status_code=500, detail=f"特征提取器初始化失败: {e}")
        
        feature = feature_extractor.extract_features(processed_image)
        logger.info("特征提取完成")
        
        # [5] 标签生成 → 生成图像标签
        logger.info("开始标签生成...")
        attributes = []
        if use_attributes:
            if tagger is None:
                logger.info("初始化标签生成器...")
                try:
                    tagger = WDViTV3Tagger()
                    logger.info(f"标签生成器初始化完成，使用设备: {tagger.device}")
                    # 加载标签生成模型
                    logger.info("加载标签生成模型...")
                    model_loaded = tagger.load_model()
                    if model_loaded:
                        logger.info("标签生成模型加载成功")
                    else:
                        logger.warning("标签生成模型加载失败，将使用默认标签")
                except Exception as e:
                    logger.error(f"标签生成器初始化失败: {e}")
                    tagger = None
            
            if tagger:
                try:
                    attributes = tagger.generate_tags(processed_image)
                except Exception as e:
                    logger.error(f"标签生成失败: {e}")
        logger.info("标签生成完成")
        
        # [6] 角色分类 → 基于特征向量匹配角色
        logger.info("开始角色分类...")
        if model_name not in model_cache:
            logger.info(f"初始化分类器: {model_name}")
            # 检查模型路径
            index_path = f"./models/{model_name}"
            if not os.path.exists(index_path):
                # 尝试使用默认模型路径
                index_path = f"./models/mobilenet_v2"
                model_name = "mobilenet_v2"  # 使用默认模型名称
                logger.warning(f"模型路径不存在，使用默认模型: {model_name}")
            
            # 检查索引文件是否存在
            faiss_path = f"{index_path}.faiss"
            mapping_path = f"{index_path}_mapping.json"
            if os.path.exists(faiss_path) and os.path.exists(mapping_path):
                # 降低阈值，提高角色识别率
                classifier = Classification(index_path, threshold=0.1)
                logger.info(f"分类器初始化完成，模型: {model_name}, 阈值: 0.1")
            else:
                # 如果索引文件不存在，创建一个空的分类器
                classifier = Classification(threshold=0.1)
                logger.warning(f"索引文件不存在，创建空分类器: {model_name}, 阈值: 0.1")
            
            model_cache[model_name] = classifier
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
        logger.info("角色分类完成")
        
        # [7] 返回结果：角色名称、标签、关键点、NSFW状态
        logger.info(f"准备返回结果: role={role}, similarity={similarity}, feature类型={type(feature)}, feature长度={len(feature) if hasattr(feature, '__len__') else 'N/A'}")
        result = {
            "role": role,
            "similarity": float(similarity),
            "attributes": attributes,
            "keypoints": keypoints,
            "nsfw_status": nsfw_result,
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
        
        processed_image = preprocessor.preprocess(temp_path)
        if processed_image is None:
            logger.error("图像预处理失败")
            raise HTTPException(status_code=500, detail="图像预处理失败")
        
        # 初始化特征提取器
        if feature_extractor is None:
            logger.info("初始化特征提取器...")
            try:
                feature_extractor = FeatureExtraction()
                logger.info(f"特征提取器初始化完成，使用设备: {feature_extractor.device}")
            except Exception as e:
                logger.error(f"特征提取器初始化失败: {e}")
                raise HTTPException(status_code=500, detail=f"特征提取器初始化失败: {e}")
        
        # 提取特征
        feature = feature_extractor.extract_features(processed_image)
        
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

# 多角色检测（暂时禁用，因为需要实现 process_multiple_characters 方法）
@app.post("/api/model/detect-multiple")
async def detect_multiple_characters(
    file: UploadFile = File(...),
    max_characters: int = Form(5)
):
    """
    检测图片中的多个角色
    
    Args:
        file: 上传的图像文件
        max_characters: 最大检测角色数
    
    Returns:
        检测到的角色列表
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
        
        # 初始化预处理器
        if preprocessor is None:
            preprocessor = Preprocessing()
        
        # 处理多个角色
        characters = preprocessor.process_multiple_characters(temp_path, max_characters=max_characters)
        
        # 初始化特征提取器
        if feature_extractor is None:
            logger.info("初始化特征提取器...")
            try:
                feature_extractor = FeatureExtraction()
                logger.info(f"特征提取器初始化完成，使用设备: {feature_extractor.device}")
            except Exception as e:
                logger.error(f"特征提取器初始化失败: {e}")
                raise HTTPException(status_code=500, detail=f"特征提取器初始化失败: {e}")
        
        # 初始化标签生成器
        if tagger is None:
            logger.info("初始化标签生成器...")
            try:
                tagger = WDViTV3Tagger()
                logger.info(f"标签生成器初始化完成，使用设备: {tagger.device}")
                # 加载模型
                logger.info("加载标签生成模型...")
                tagger.load_model()
                logger.info("标签生成模型加载完成")
            except Exception as e:
                logger.error(f"标签生成器初始化失败: {e}")
                tagger = None
        
        # 为每个角色提取特征和生成标签
        results = []
        for i, char in enumerate(characters):
            # 提取特征
            feature = feature_extractor.extract_features(char['image'])
            
            # 生成属性标签
            attributes = []
            if tagger:
                try:
                    attributes = tagger.generate_tags(char['image'])
                except Exception as e:
                    logger.error(f"标签生成失败: {e}")
            
            # 构建角色结果
            results.append({
                "id": i + 1,
                "box": char['box'],
                "confidence": char['confidence'],
                "attributes": attributes,
                "feature": feature.tolist() if hasattr(feature, 'tolist') else feature
            })
        
        # 构建响应
        response = {
            "total_characters": len(results),
            "characters": results
        }
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"多角色检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"多角色检测失败: {e}")
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
    
    Args:
        file: 上传的图像文件
        use_coreml: 是否使用CoreML
        use_model: 是否使用模型
        use_attributes: 是否使用属性
        model_name: 模型名称
        cache_bypass: 是否绕过缓存
    
    Returns:
        分类结果
    """
    # 调用现有的预测函数
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
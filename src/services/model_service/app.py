#!/usr/bin/env python3
"""
模型服务主文件
提供模型预测、特征提取等功能
"""
import os
import sys
import platform

IS_MACOS = platform.system() == "Darwin"

if IS_MACOS:
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

def get_optimal_device():
    """自动选择最佳计算设备"""
    if IS_MACOS:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available() and not IS_MACOS:
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"

OPTIMAL_DEVICE = get_optimal_device()

# 注册 HEIF/HEIC 图像解码器
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    print("✅ HEIF/HEIC 解码器已注册")
except ImportError:
    print("⚠️ HEIF/HEIC 支持不可用，将无法处理 HEIC 格式图片")

# 延迟导入核心模块，避免启动时的锁竞争
Preprocessing = None
FeatureExtraction = None
WDViTV3Tagger = None
Classification = None
get_logger = None


# 动态导入函数
def import_core_modules():
    global Preprocessing, FeatureExtraction, WDViTV3Tagger, Classification, get_logger
    # 只导入非PyTorch依赖的模块
    from src.core.preprocessing.preprocessing import Preprocessing
    from src.core.feature_extraction.feature_extraction import FeatureExtraction
    from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
    from src.core.classification.classification import Classification
    from src.core.logging.global_logger import get_logger


# 使用统一配置
from src.core.config.service_config import get_service_config

config = get_service_config()

# 初始化日志
import_core_modules()
logger = get_logger("model_service")

# 添加线程锁和异步锁
import threading
import asyncio

torch_import_lock = threading.Lock()
model_init_lock = asyncio.Lock()  # 异步锁，用于异步函数中

# 导入模型缓存
from src.core.cache.model_cache import model_cache

# 导入角色信息加载器
from src.core.utils.role_info_loader import get_role_info


# 延迟导入torch
def import_torch():
    """延迟导入torch，避免启动时的锁竞争问题"""
    global torch
    with torch_import_lock:
        if "torch" not in globals():
            import torch

            # 启用MPS加速
            mps_available = torch.backends.mps.is_available()
            cuda_available = torch.cuda.is_available()
            # 配置MPS相关的环境变量
            import os

            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["MPS_HIGH_WATERMARK_RATIO"] = "0.5"
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.5"
            os.environ["TORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.5"
            if mps_available:
                logger.info("PyTorch已导入，MPS已启用")
            elif cuda_available:
                logger.info("PyTorch已导入，CUDA已启用")
            else:
                logger.info("PyTorch已导入，使用CPU")


# 模型加载装饰器
def safe_model_load(func):
    """安全的模型加载装饰器"""

    async def wrapper(*args, **kwargs):
        try:
            # 确保torch已导入
            if "torch" not in globals():
                import_torch()
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            import traceback

            logger.error(f"异常堆栈: {traceback.format_exc()}")
            # 返回降级结果
            return {"role": "unknown", "similarity": 0.0, "tags": ["anime", "digital art"]}

    # 保留原始函数的元数据
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    return wrapper


# 初始化FastAPI应用
app = FastAPI(title="Model Service", description="Anime Role Detect Model Service", version="1.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加响应压缩中间件
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)


# 健康检查
@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "Model Service"}


# 全局变量
preprocessor = None  # 预处理器实例
feature_extractor = None  # 特征提取器实例
tagger = None  # 标签生成器实例


# 初始化模型
async def init_models():
    """初始化模型并预热"""
    global preprocessor, feature_extractor, tagger

    try:
        # 1. 初始化预处理器
        logger.info("初始化预处理器...")
        preprocessor = Preprocessing()
        logger.info("预处理器初始化完成")

        # 2. 预热模型（在后台异步执行，不阻塞启动）
        asyncio.create_task(warmup_models())
        
        logger.info("模型服务启动完成，模型预热任务已启动")
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")


async def warmup_models():
    """
    模型预热 - 使用虚拟数据进行推理，加载模型到内存
    这样在用户请求时响应更快
    """
    global feature_extractor, tagger
    
    try:
        logger.info("开始模型预热...")
        start_time = time.time()
        
        # 确保torch已导入
        if "torch" not in globals():
            import_torch()
        
        import torch
        from PIL import Image
        import io
        
        # 创建虚拟图像数据（224x224 RGB）
        dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # 预热特征提取器
        logger.info("预热特征提取器...")
        if feature_extractor is None:
            async with model_init_lock:
                if feature_extractor is None:
                    feature_extractor = FeatureExtraction()
        
        # 执行一次虚拟推理
        try:
            processed = preprocessor.preprocess(dummy_image)
            if processed is not None:
                _ = feature_extractor.extract(processed)
                logger.info("特征提取器预热完成")
        except Exception as e:
            logger.warning(f"特征提取器预热失败: {e}")
        
        # 预热标签生成器
        logger.info("预热标签生成器...")
        if tagger is None:
            try:
                tagger = WDViTV3Tagger()
                # 执行一次虚拟推理
                _ = tagger.predict(dummy_image)
                logger.info("标签生成器预热完成")
            except Exception as e:
                logger.warning(f"标签生成器预热失败: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"模型预热完成，耗时: {elapsed:.2f}秒")
        
    except Exception as e:
        logger.error(f"模型预热失败: {e}")
        # 预热失败不影响服务启动，只是首次请求会慢一些


# 根路径，重定向到文档
@app.get("/model_service")
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


# 模型预测
@app.post("/api/model/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form("efficientnet_b3_loli_optimized_v2_20260529_133654"),
    use_attributes: bool = Form(True),
    multilabel: bool = Form(False),
    threshold: float = Form(0.4),
    use_cache: bool = Form(True),
):
    """
    预测图像

    Args:
        file: 上传的图像文件
        model_name: 模型名称
        use_attributes: 是否使用属性
        multilabel: 是否返回多标签结果
        threshold: 分类阈值
        use_cache: 是否使用缓存

    Returns:
        预测结果
    """
    global preprocessor, feature_extractor, tagger

    from src.services.cache_service.redis_cache import get_redis_cache

    redis_cache = get_redis_cache()
    image_hash = None
    temp_path = None

    try:
        if "torch" not in globals():
            import_torch()

        content = await file.read()

        if use_cache and redis_cache.available:
            image_hash = redis_cache.compute_image_hash(content)
            cached_result = redis_cache.get_image_result(image_hash)
            if cached_result:
                logger.info(f"缓存命中，返回结果: role={cached_result.get('role')}")
                return cached_result

        from PIL import Image
        import io

        try:
            # 从内存数据创建PIL图像
            image = Image.open(io.BytesIO(content)).convert("RGB")

            # [1] 图像预处理（解码、缩放、归一化）
            logger.info("开始图像预处理...")
            if preprocessor is None:
                async with model_init_lock:
                    if preprocessor is None:
                        preprocessor = Preprocessing()

            processed_image = preprocessor.preprocess(image)
            if processed_image is None:
                logger.error("图像预处理失败")
                raise HTTPException(status_code=500, detail="图像预处理失败")
            logger.info("图像预处理完成")
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            raise HTTPException(status_code=500, detail=f"图像处理失败: {e}")

        # [3] 关键点检测（面部、手部、姿态）
        logger.info("开始关键点检测...")
        # 暂时禁用关键点检测，避免macOS上的Mutex锁竞争问题
        keypoints = None
        logger.info("关键点检测已暂时禁用")

        # [4] 特征提取 → 生成512维特征向量
        logger.info("开始特征提取...")
        if feature_extractor is None:
            async with model_init_lock:
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
                async with model_init_lock:
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

        # 使用带过期机制的缓存
        classifier = model_cache.get(model_name)
        if classifier is None:
            logger.info(f"初始化分类器: {model_name}")
            # 检查模型路径
            index_path = f"./models/{model_name}"
            if not os.path.exists(index_path):
                # 尝试使用默认模型路径
                index_path = f"./models/efficientnet_b3_loli_optimized_v2_20260529_133654"
                model_name = "efficientnet_b3_loli_optimized_v2_20260529_133654"  # 使用默认模型名称
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

            # 缓存分类器
            model_cache.set(model_name, classifier)
            logger.info(f"模型已缓存，当前缓存大小: {model_cache.size()}")

        # 检查分类器是否有索引
        if classifier.index is not None:
            # 分类图像
            role, similarity = classifier.classify(feature, 5, attributes)  # 减少top_k值，提高速度
        else:
            # 如果没有索引，尝试使用本地模型进行分类
            logger.warning("分类器索引不存在，尝试使用本地模型进行分类")
            try:
                # 导入MultiRoleDetector
                from src.core.detection.multi_role_detection import MultiRoleDetector

                # 初始化检测器
                detector = MultiRoleDetector(model_name=model_name)

                # 加载模型
                detector._load_trained_model()

                # 分类角色
                if detector.model and detector.class_to_idx:
                    # 使用实际上传的图像进行分类
                    role, similarity = detector._classify_role(image)
                    logger.info(f"本地模型分类结果: {role}, 相似度: {similarity}")
                else:
                    role = "unknown"
                    similarity = 0.0
            except Exception as e:
                logger.error(f"使用本地模型分类失败: {e}")
                role = "unknown"
                similarity = 0.0
        logger.info("角色分类完成")

        # [7] 返回结果：角色名称、标签、关键点、NSFW状态
        logger.info(
            f"准备返回结果: role={role}, similarity={similarity}, feature类型={type(feature)}, feature长度={len(feature) if hasattr(feature, '__len__') else 'N/A'}"
        )

        role_full_info = get_role_info(role)
        result = {
            "role": role,
            "role_cn": role_full_info.get("cn", role),
            "role_jp": role_full_info.get("jp", ""),
            "role_anime": role_full_info.get("anime", ""),
            "similarity": float(similarity),
            "attributes": attributes,
            "keypoints": keypoints,
            "feature": feature.tolist() if hasattr(feature, "tolist") else feature,
        }
        logger.info(f"返回结果: {result}")

        if use_cache and redis_cache.available:
            redis_cache.set_image_result(image_hash, result)
            logger.info(f"结果已缓存，hash: {image_hash[:8]}...")

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        import traceback

        logger.error(f"异常堆栈: {traceback.format_exc()}")
        # 返回复降级结果，避免服务崩溃
        return {"role": "unknown", "similarity": 0.0, "tags": ["anime", "digital art"]}
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"删除临时文件失败: {e}")


# 特征提取
@app.post("/api/model/extract")
async def extract_features(file: UploadFile = File(...)):
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
        result = {"feature": feature.tolist() if hasattr(feature, "tolist") else feature}

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


# 多角色检测
@app.post("/api/model/detect-multiple")
async def detect_multiple_characters(file: UploadFile = File(...), max_characters: int = Form(5)):
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

        # 使用增强版多角色检测器
        try:
            from src.core.detection.multi_role_detection_enhanced import EnhancedMultiRoleDetector

            detector = EnhancedMultiRoleDetector(
                model_name="efficientnet_b3_loli_optimized_v2_20260529_133654",
                enable_open_set=True,
                enable_fuzzy_record=True,
                unknown_threshold=0.3,
                fuzzy_threshold=0.5,
            )
            detection_results = detector.detect_roles(temp_path)
            logger.info(f"增强版检测器检测到 {len(detection_results)} 个角色")
        except Exception as e:
            logger.warning(f"增强版检测器初始化失败，回退到原版: {e}")
            from src.core.detection.multi_role_detection import MultiRoleDetector

            detector = MultiRoleDetector(
                model_name="efficientnet_b3_loli_optimized_v2_20260529_133654"
            )
            detection_results = detector.detect_roles(temp_path)

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

        # 为每个检测到的角色生成标签
        results = []
        for i, detection in enumerate(detection_results[:max_characters]):
            role = detection.get("role", "unknown")
            similarity = detection.get("similarity", 0.0)
            bbox = detection.get("bbox", {})
            confidence = detection.get("confidence", 0.0)
            attributes = detection.get("attributes", [])

            # 如果没有属性标签，使用tagger生成
            if not attributes and tagger:
                try:
                    # 从检测结果中获取裁剪的角色图像
                    role_image = detection.get("cropped_image")
                    if role_image is not None:
                        attributes = tagger.generate_tags(role_image)
                except Exception as e:
                    logger.error(f"标签生成失败: {e}")

            # 构建角色结果
            role_full_info = get_role_info(role)
            role_result = {
                "id": i + 1,
                "role": role,
                "role_cn": role_full_info.get("cn", role),
                "role_jp": role_full_info.get("jp", ""),
                "role_anime": role_full_info.get("anime", ""),
                "similarity": float(similarity),
                "box": bbox,
                "confidence": float(confidence),
                "attributes": attributes,
            }

            # 添加增强版检测器的额外字段
            if "decision" in detection:
                role_result["decision"] = detection["decision"]
            if "is_unknown" in detection:
                role_result["is_unknown"] = detection["is_unknown"]
            if "is_fuzzy" in detection:
                role_result["is_fuzzy"] = detection["is_fuzzy"]

            results.append(role_result)

        # 构建响应
        response = {"roles": results, "count": len(results)}

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"多角色检测失败: {e}")
        import traceback

        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"多角色检测失败: {e}")
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"删除临时文件失败: {e}")


# YOLOv8 多目标检测 API
@app.post("/api/model/detect-yolo")
async def detect_with_yolo(
    file: UploadFile = File(...),
    yolo_model: str = Form("yolov8n.pt"),
    person_conf_threshold: float = Form(0.5),
    max_detections: int = Form(10),
):
    """
    使用 YOLOv8 进行多目标检测 + 角色识别

    Args:
        file: 上传的图像文件
        yolo_model: YOLOv8 模型名称 (yolov8n.pt, yolov8s.pt, yolov8m.pt 等)
        person_conf_threshold: 人体检测置信度阈值
        max_detections: 最大检测数量

    Returns:
        检测到的角色列表，包含边界框和置信度
    """
    from src.core.detection.multi_target_detector import MultiTargetDetector

    temp_path = None

    try:
        # 读取文件
        content = await file.read()
        temp_path = f"temp_yolo_{int(time.time())}_{file.filename}"

        # 保存临时文件
        with open(temp_path, "wb") as f:
            f.write(content)

        # 确定设备
        device = OPTIMAL_DEVICE
        logger.info(f"YOLOv8 多目标检测，使用设备: {device}")

        # 初始化检测器（延迟加载避免启动阻塞）
        if not hasattr(detect_with_yolo, "_detector") or detect_with_yolo._detector is None:
            logger.info("初始化 YOLOv8 多目标检测器...")
            detect_with_yolo._detector = MultiTargetDetector(
                yolo_model=yolo_model,
                device=device,
            )
            logger.info("YOLOv8 检测器初始化完成")

        detector = detect_with_yolo._detector

        # 加载图像
        from PIL import Image
        image = Image.open(temp_path).convert("RGB")

        # 执行检测
        logger.info("开始 YOLOv8 人体检测 + 角色识别...")
        results = detector.detect_and_classify(image, person_conf_threshold=person_conf_threshold)

        # 构建响应
        response_results = []
        for i, detection in enumerate(results.get("detections", [])[:max_detections]):
            role_pred = detection.get("role_prediction", {})
            role_name = role_pred.get("role", "unknown")
            role_conf = role_pred.get("confidence", 0.0)

            role_full_info = get_role_info(role_name)

            response_results.append({
                "id": i + 1,
                "role": role_name,
                "role_cn": role_full_info.get("cn", role_name),
                "role_jp": role_full_info.get("jp", ""),
                "role_anime": role_full_info.get("anime", ""),
                "confidence": float(role_conf),
                "person_confidence": float(detection.get("person_confidence", 0.0)),
                "bbox": detection.get("bbox", []),
                "class_id": role_pred.get("class_id", -1),
            })

        logger.info(f"YOLOv8 检测完成，返回 {len(response_results)} 个结果")

        return {
            "roles": response_results,
            "count": len(response_results),
            "image_size": results.get("image_size", []),
            "detector": "YOLOv8 + EfficientNet",
            "model": yolo_model,
        }

    except Exception as e:
        logger.error(f"YOLOv8 多目标检测失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"YOLOv8 检测失败: {e}")

    finally:
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
    model_name: str = Form("efficientnet_b3_loli_optimized_v2_20260529_133654"),
    cache_bypass: bool = Form(False),
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
    return await predict_image(file=file, model_name=model_name, use_attributes=use_attributes)


# 批量推理端点
@app.post("/api/model/batch-predict")
async def batch_predict_images(
    files: List[UploadFile] = File(...),
    model_name: str = Form("efficientnet_b3_loli_optimized_v2_20260529_133654"),
    use_attributes: bool = Form(True),
    batch_size: int = Form(8),
    multilabel: bool = Form(False),
    threshold: float = Form(0.4),
):
    """
    批量预测多张图像

    Args:
        files: 上传的图像文件列表
        model_name: 模型名称
        use_attributes: 是否使用属性
        batch_size: 批量大小
        multilabel: 是否返回多标签结果
        threshold: 分类阈值

    Returns:
        批量预测结果
    """
    global preprocessor, feature_extractor, tagger

    results = []

    try:
        # 安全导入torch，避免锁竞争问题
        if "torch" not in globals():
            import_torch()

        # 处理文件
        for i, file in enumerate(files):
            # 读取文件内容
            content = await file.read()

            try:
                # 直接从内存创建PIL图像，避免临时文件I/O
                from PIL import Image
                import io

                # 从内存数据创建PIL图像
                image = Image.open(io.BytesIO(content)).convert("RGB")

                # 图像预处理
                if preprocessor is None:
                    async with model_init_lock:
                        if preprocessor is None:
                            preprocessor = Preprocessing()

                processed_image = preprocessor.preprocess(image)
                if processed_image is None:
                    logger.error(f"图像预处理失败: {file.filename}")
                    results.append({"filename": file.filename, "error": "图像预处理失败"})
                    continue

                # 特征提取
                if feature_extractor is None:
                    async with model_init_lock:
                        if feature_extractor is None:
                            logger.info("初始化特征提取器...")
                            try:
                                feature_extractor = FeatureExtraction()
                                logger.info(
                                    f"特征提取器初始化完成，使用设备: {feature_extractor.device}"
                                )
                            except Exception as e:
                                logger.error(f"特征提取器初始化失败: {e}")
                                results.append(
                                    {
                                        "filename": file.filename,
                                        "error": f"特征提取器初始化失败: {e}",
                                    }
                                )
                                continue

                feature = feature_extractor.extract_features(processed_image)

                # 标签生成
                attributes = []
                if use_attributes:
                    if tagger is None:
                        async with model_init_lock:
                            if tagger is None:
                                logger.info("初始化标签生成器...")
                                try:
                                    tagger = WDViTV3Tagger()
                                    logger.info(f"标签生成器初始化完成，使用设备: {tagger.device}")
                                    # 加载标签生成模型
                                    logger.info("加载标签生成模型...")
                                    model_loaded = tagger.load_model()
                                    if not model_loaded:
                                        logger.warning("标签生成模型加载失败，将使用默认标签")
                                except Exception as e:
                                    logger.error(f"标签生成器初始化失败: {e}")
                                    tagger = None

                    if tagger:
                        try:
                            attributes = tagger.generate_tags(processed_image)
                        except Exception as e:
                            logger.error(f"标签生成失败: {e}")

                # 角色分类
                # 使用带过期机制的缓存
                classifier = model_cache.get(model_name)
                if classifier is None:
                    logger.info(f"初始化分类器: {model_name}")
                    # 检查模型路径
                    index_path = f"./models/{model_name}"
                    if not os.path.exists(index_path):
                        # 尝试使用默认模型路径
                        index_path = f"./models/efficientnet_b3_loli_optimized_v2_20260529_133654"
                        model_name = (
                            "efficientnet_b3_loli_optimized_v2_20260529_133654"  # 使用默认模型名称
                        )
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

                    # 缓存分类器
                    model_cache.set(model_name, classifier)

                # 检查分类器是否有索引
                if classifier.index is not None:
                    # 分类图像
                    if multilabel:
                        # 多标签分类
                        roles = classifier.classify(
                            feature, 5, attributes, multilabel=True, threshold=threshold
                        )
                        # 构建多标签结果
                        if roles:
                            role = roles[0][0]  # 主角色
                            similarity = roles[0][1]  # 主角色相似度
                        else:
                            role = "unknown"
                            similarity = 0.0
                    else:
                        # 单标签分类
                        role, similarity = classifier.classify(
                            feature, 5, attributes, threshold=threshold
                        )  # 减少top_k值，提高速度
                else:
                    # 如果没有索引，返回unknown
                    role = "unknown"
                    similarity = 0.0

                # 构建结果
                role_full_info = get_role_info(role)
                result = {
                    "filename": file.filename,
                    "role": role,
                    "role_cn": role_full_info.get("cn", role),
                    "role_jp": role_full_info.get("jp", ""),
                    "role_anime": role_full_info.get("anime", ""),
                    "similarity": float(similarity),
                    "attributes": attributes,
                }

                # 如果是多标签模式，添加所有识别到的角色
                if multilabel:
                    result["roles"] = [
                        {"role": r[0], "similarity": float(r[1]), **get_role_info(r[0])}
                        for r in roles
                    ]

                results.append(result)

            except Exception as e:
                logger.error(f"处理文件 {file.filename} 失败: {e}")
                results.append({"filename": file.filename, "error": str(e)})

        return {"results": results, "count": len(results)}

    except Exception as e:
        logger.error(f"批量预测失败: {e}")
        import traceback

        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"批量预测失败: {e}")


# 主函数
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模型服务")
    parser.add_argument("--host", type=str, default=config.MODEL_SERVICE_HOST, help="服务主机")
    parser.add_argument("--port", type=int, default=config.MODEL_SERVICE_PORT, help="服务端口")
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
        limit_max_requests=1000,
    )

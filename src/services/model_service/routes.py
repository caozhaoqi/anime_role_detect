"""
模型服务路由
"""
import os
import sys
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from PIL import Image
import io

from src.core.logging.global_logger import get_logger
from src.services.model_service.classifiers import EfficientNetClassifier
from src.core.cache.model_cache import model_cache
from src.core.utils.role_info_loader import get_role_info
from src.services.cache_service.redis_cache import get_redis_cache
from src.services.model.recognition_service import get_recognition_service
from src.models.recognition_record import RecognitionRecordCreate

# 模块级别导入 WDViTV3Tagger，确保在请求处理之前完成导入和 torch 初始化
# 避免在首次请求时触发 MPS C++ 后端 mutex 死锁
try:
    from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
except Exception:
    WDViTV3Tagger = None

logger = get_logger("model_service.routes")

router = APIRouter()

# 全局引用，由 app.py 在初始化时设置
preprocessor = None
feature_extractor = None
tagger = None
_efficientnet_classifier = None

model_init_lock = asyncio.Lock()
OPTIMAL_DEVICE = "cpu"

# 延迟导入核心模块
def import_core_modules():
    pass  # 由 app.py 初始化时调用


@router.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Model Service"}


@router.get("/model_service")
async def root():
    return {"message": "Model Service", "docs": "/docs"}


def _ensure_torch_imported():
    """确保torch已导入"""
    if "torch" not in globals():
        import torch
        mps_available = torch.backends.mps.is_available()
        cuda_available = torch.cuda.is_available()
        if mps_available:
            logger.info("MPS已启用")
        elif cuda_available:
            logger.info("CUDA已启用")
        else:
            logger.info("使用CPU")


@router.post("/api/model/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form("efficientnet_b3_loli_optimized_v2_20260529_133654"),
    use_attributes: bool = Form(True),
    use_keypoints: bool = Form(False),
    multilabel: bool = Form(False),
    threshold: float = Form(0.4),
    use_cache: bool = Form(True),
):
    """预测图像"""
    global preprocessor, feature_extractor, tagger

    redis_cache = get_redis_cache()
    image_hash = None
    temp_path = None

    try:
        _ensure_torch_imported()

        content = await file.read()

        if use_cache and redis_cache.available:
            image_hash = redis_cache.compute_image_hash(content)
            cached_result = redis_cache.get_image_result(image_hash)
            if cached_result:
                logger.info(f"缓存命中: role={cached_result.get('role')}")
                return cached_result

        image = Image.open(io.BytesIO(content)).convert("RGB")

        if preprocessor is None:
            async with model_init_lock:
                if preprocessor is None:
                    from src.core.preprocessing.preprocessing import Preprocessing
                    preprocessor = Preprocessing()

        processed_image = preprocessor.preprocess(image)
        if processed_image is None:
            raise HTTPException(status_code=500, detail="图像预处理失败")

        keypoints = None
        if use_keypoints:
            try:
                # 使用子进程运行关键点检测，避免与 uvicorn 服务中的 PyTorch MPS 后端冲突
                import subprocess, json, base64, sys
                from io import BytesIO
                # 将图像编码为 base64 传给子进程
                if hasattr(image, 'convert'):
                    buf = BytesIO()
                    image.save(buf, format='JPEG')
                    img_b64 = base64.b64encode(buf.getvalue()).decode()
                else:
                    img_b64 = base64.b64encode(content).decode()

                result = subprocess.run(
                    [sys.executable, '-c', '''
import base64, json, sys
from io import BytesIO
from PIL import Image
img = Image.open(BytesIO(base64.b64decode(sys.argv[1]))).convert("RGB")
# 在子进程内 lazily import mediapipe
from src.core.keypoint.mediapipe_keypoint_detector import detect_keypoints
kps = detect_keypoints(img)
print(json.dumps(kps), flush=True)
''', img_b64],
                    capture_output=True, text=True, timeout=30,
                    env={**__import__('os').environ, 'PYTORCH_MPS_DISABLE': '1', 'PYTHONPATH': __import__('os').path.dirname(__file__) + '/../../../..'},
                    cwd=__import__('os').path.dirname(__file__) + '/../../../..'
                )
                if result.returncode == 0 and result.stdout.strip():
                    keypoints = json.loads(result.stdout.strip())
                    logger.info(f"关键点检测完成: {len(keypoints)} 个关键点")
                else:
                    if result.stderr:
                        logger.warning(f"关键点检测子进程 stderr: {result.stderr[:200]}")
                    keypoints = []
            except Exception as e:
                logger.warning(f"关键点检测失败: {e}")
                keypoints = []
        global _efficientnet_classifier

        if _efficientnet_classifier is None:
            try:
                loop = asyncio.get_running_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    _efficientnet_classifier = await loop.run_in_executor(
                        executor, EfficientNetClassifier.get_instance
                    )
            except Exception as e:
                logger.error(f"EfficientNet分类器初始化失败: {e}")

        if _efficientnet_classifier is not None and _efficientnet_classifier.model is not None:
            direct_role, direct_confidence, feature = _efficientnet_classifier.classify_with_features(image)
            logger.info(f"EfficientNet直接分类: role={direct_role}, confidence={direct_confidence:.4f}")

            if direct_confidence >= 0.3:
                role = direct_role
                similarity = direct_confidence
                attributes = []
                if use_attributes:
                    if tagger is None:
                        async with model_init_lock:
                            if tagger is None:
                                try:
                                    from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
                                    tagger = WDViTV3Tagger.get_instance()
                                    loop = asyncio.get_running_loop()
                                    with ThreadPoolExecutor(max_workers=1) as executor:
                                        await loop.run_in_executor(executor, tagger.load_model)
                                    logger.info("标签生成器同步加载完成")
                                except Exception as e:
                                    logger.error(f"标签生成器初始化失败: {e}")
                                    tagger = None
                    if tagger:
                        try:
                            attributes = tagger.generate_tags(image)
                        except Exception as e:
                            logger.error(f"标签生成失败: {e}")

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

                if use_cache and redis_cache.available:
                    redis_cache.set_image_result(image_hash, result)
                # 保存识别记录
                try:
                    recognition_service = get_recognition_service()
                    record = RecognitionRecordCreate(
                        user_id="anonymous",
                        username="anonymous",
                        image_filename=file.filename if file.filename else "unknown",
                        image_path="",
                        recognition_result=result,
                        model_used=model_name,
                        processing_time=time.time(),
                        is_multi_role=False,
                        nsfw_status=False,
                        detected_text=False,
                    )
                    recognition_service.create_record(record)
                except Exception as e:
                    logger.warning(f"存储识别记录失败: {e}")
                return result

        from src.core.feature_extraction.feature_extraction import FeatureExtraction
        logger.info("EfficientNet置信度不足，降级到Faiss搜索...")
        if feature_extractor is None:
            async with model_init_lock:
                if feature_extractor is None:
                    loop = asyncio.get_running_loop()
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        feature_extractor = await loop.run_in_executor(executor, FeatureExtraction)

        feature = feature_extractor.extract_features(processed_image)

        attributes = []
        if use_attributes:
            if tagger is None:
                async with model_init_lock:
                    if tagger is None:
                        try:
                            from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
                            tagger = WDViTV3Tagger.get_instance()
                            loop = asyncio.get_running_loop()
                            with ThreadPoolExecutor(max_workers=1) as executor:
                                await loop.run_in_executor(executor, tagger.load_model)
                            logger.info("标签生成器同步加载完成")
                        except Exception:
                            tagger = None
            if tagger:
                try:
                    attributes = tagger.generate_tags(image)
                except Exception as e:
                    logger.error(f"标签生成失败: {e}")

        classifier = model_cache.get(model_name)
        if classifier is None:
            from src.core.classification.classification import Classification
            index_path = f"./models/{model_name}"
            if not os.path.exists(index_path):
                index_path = "./models/efficientnet_b3_loli_optimized_v2_20260529_133654"
                model_name = "efficientnet_b3_loli_optimized_v2_20260529_133654"

            faiss_path = f"{index_path}.faiss"
            mapping_path = f"{index_path}_mapping.json"
            if os.path.exists(faiss_path) and os.path.exists(mapping_path):
                classifier = Classification(index_path, threshold=0.1)
            else:
                classifier = Classification(threshold=0.1)
            model_cache.set(model_name, classifier)

        if classifier.index is not None:
            role, similarity = classifier.classify(feature, 5, attributes)
        else:
            try:
                from src.core.detection.multi_role_detection import MultiRoleDetector
                detector = MultiRoleDetector(model_name=model_name)
                detector._load_trained_model()
                if detector.model and detector.class_to_idx:
                    role, similarity = detector._classify_role(image)
                else:
                    role, similarity = "unknown", 0.0
            except Exception:
                role, similarity = "unknown", 0.0

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

        if use_cache and redis_cache.available:
            redis_cache.set_image_result(image_hash, result)

        # 保存识别记录
        try:
            recognition_service = get_recognition_service()
            record = RecognitionRecordCreate(
                user_id="anonymous",
                username="anonymous",
                image_filename=file.filename if file.filename else "unknown",
                image_path="",
                recognition_result=result,
                model_used=model_name,
                processing_time=time.time(),
                is_multi_role=False,
                nsfw_status=False,
                detected_text=False,
            )
            recognition_service.create_record(record)
        except Exception as e:
            logger.warning(f"存储识别记录失败: {e}")

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        return {"role": "unknown", "similarity": 0.0, "tags": ["anime", "digital art"]}
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.post("/api/model/extract")
async def extract_features(file: UploadFile = File(...)):
    """特征提取"""
    global preprocessor, feature_extractor
    temp_path = None
    try:
        content = await file.read()
        temp_path = f"temp_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)

        if preprocessor is None:
            from src.core.preprocessing.preprocessing import Preprocessing
            preprocessor = Preprocessing()

        processed_image = preprocessor.preprocess(temp_path)
        if processed_image is None:
            raise HTTPException(status_code=500, detail="图像预处理失败")

        if feature_extractor is None:
            from src.core.feature_extraction.feature_extraction import FeatureExtraction
            feature_extractor = FeatureExtraction()

        feature = feature_extractor.extract_features(processed_image)
        return {"feature": feature.tolist() if hasattr(feature, "tolist") else feature}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"特征提取失败: {e}")
        raise HTTPException(status_code=500, detail=f"特征提取失败: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.post("/api/model/detect-multiple")
async def detect_multiple_characters(
    file: UploadFile = File(...),
    max_characters: int = Form(5),
):
    """检测图片中的多个角色"""
    global preprocessor, feature_extractor, tagger
    temp_path = None
    try:
        content = await file.read()
        temp_path = f"temp_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)

        try:
            from src.core.detection.multi_role_detection_enhanced import EnhancedMultiRoleDetector
            detector = EnhancedMultiRoleDetector(
                model_name="efficientnet_b3_loli_optimized_v2_20260529_133654",
                enable_open_set=True, enable_fuzzy_record=True,
                unknown_threshold=0.3, fuzzy_threshold=0.5,
            )
            detection_results = detector.detect_roles(temp_path)
        except Exception:
            from src.core.detection.multi_role_detection import MultiRoleDetector
            detector = MultiRoleDetector(model_name="efficientnet_b3_loli_optimized_v2_20260529_133654")
            detection_results = detector.detect_roles(temp_path)

        if feature_extractor is None:
            from src.core.feature_extraction.feature_extraction import FeatureExtraction
            feature_extractor = FeatureExtraction()

        if tagger is None:
            from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
            tagger = WDViTV3Tagger.get_instance()
            tagger.load_model()

        results = []
        for i, detection in enumerate(detection_results[:max_characters]):
            role = detection.get("role", "unknown")
            similarity = detection.get("similarity", 0.0)
            bbox = detection.get("bbox", {})
            confidence = detection.get("confidence", 0.0)
            attributes = detection.get("attributes", [])

            if not attributes and tagger:
                role_image = detection.get("cropped_image")
                if role_image is not None:
                    attributes = tagger.generate_tags(role_image)

            role_full_info = get_role_info(role)
            role_result = {
                "id": i + 1, "role": role,
                "role_cn": role_full_info.get("cn", role),
                "role_jp": role_full_info.get("jp", ""),
                "role_anime": role_full_info.get("anime", ""),
                "similarity": float(similarity), "box": bbox,
                "confidence": float(confidence), "attributes": attributes,
            }
            if "decision" in detection:
                role_result["decision"] = detection["decision"]
            if "is_unknown" in detection:
                role_result["is_unknown"] = detection["is_unknown"]
            if "is_fuzzy" in detection:
                role_result["is_fuzzy"] = detection["is_fuzzy"]
            results.append(role_result)

        return {"roles": results, "count": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"多角色检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"多角色检测失败: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.post("/api/model/detect-yolo")
async def detect_with_yolo(
    file: UploadFile = File(...),
    yolo_model: str = Form("yolov8n.pt"),
    person_conf_threshold: float = Form(0.2),
    max_detections: int = Form(10),
):
    """使用 YOLOv8 进行多目标检测"""
    from src.core.detection.multi_target_detector import MultiTargetDetector
    temp_path = None
    try:
        content = await file.read()
        temp_path = f"temp_yolo_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)

        device = OPTIMAL_DEVICE
        if not hasattr(detect_with_yolo, "_detector") or detect_with_yolo._detector is None:
            detect_with_yolo._detector = MultiTargetDetector(yolo_model=yolo_model, device=device)

        detector = detect_with_yolo._detector
        image = Image.open(temp_path).convert("RGB")
        results = detector.detect_and_classify(image, person_conf_threshold=person_conf_threshold)

        response_results = []
        for i, detection in enumerate(results.get("detections", [])[:max_detections]):
            role_pred = detection.get("role_prediction", {})
            role_name = role_pred.get("role", "unknown")
            role_conf = role_pred.get("confidence", 0.0)
            role_full_info = get_role_info(role_name)
            response_results.append({
                "id": i + 1, "role": role_name,
                "role_cn": role_full_info.get("cn", role_name),
                "role_jp": role_full_info.get("jp", ""),
                "role_anime": role_full_info.get("anime", ""),
                "confidence": float(role_conf),
                "person_confidence": float(detection.get("person_confidence", 0.0)),
                "bbox": detection.get("bbox", []),
                "class_id": role_pred.get("class_id", -1),
            })

        return {
            "roles": response_results, "count": len(response_results),
            "image_size": results.get("image_size", []),
            "detector": "YOLOv8 + EfficientNet", "model": yolo_model,
        }
    except Exception as e:
        logger.error(f"YOLOv8 多目标检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"YOLOv8 检测失败: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.post("/api/classify")
async def classify_image(
    file: UploadFile = File(...),
    use_coreml: bool = Form(False),
    use_model: bool = Form(True),
    use_attributes: bool = Form(True),
    use_keypoints: bool = Form(False),
    model_name: str = Form("efficientnet_b3_loli_optimized_v2_20260529_133654"),
    cache_bypass: bool = Form(False),
):
    """分类图像（兼容前端调用）"""
    return await predict_image(file=file, model_name=model_name, use_attributes=use_attributes, use_keypoints=use_keypoints)


@router.post("/api/model/batch-predict")
async def batch_predict_images(
    files: List[UploadFile] = File(...),
    model_name: str = Form("efficientnet_b3_loli_optimized_v2_20260529_133654"),
    use_attributes: bool = Form(True),
    batch_size: int = Form(8),
    multilabel: bool = Form(False),
    threshold: float = Form(0.4),
):
    """批量预测多张图像"""
    global preprocessor, feature_extractor, tagger
    results = []
    try:
        _ensure_torch_imported()
        for file in files:
            content = await file.read()
            try:
                from src.core.classification.classification import Classification
                image = Image.open(io.BytesIO(content)).convert("RGB")
                if preprocessor is None:
                    async with model_init_lock:
                        if preprocessor is None:
                            from src.core.preprocessing.preprocessing import Preprocessing
                            preprocessor = Preprocessing()
                processed_image = preprocessor.preprocess(image)
                if processed_image is None:
                    results.append({"filename": file.filename, "error": "图像预处理失败"})
                    continue

                if feature_extractor is None:
                    async with model_init_lock:
                        if feature_extractor is None:
                            from src.core.feature_extraction.feature_extraction import FeatureExtraction
                            feature_extractor = FeatureExtraction()
                feature = feature_extractor.extract_features(processed_image)

                attributes = []
                if use_attributes and tagger:
                    try:
                        attributes = tagger.generate_tags(image)
                    except Exception:
                        pass

                classifier = model_cache.get(model_name)
                if classifier is None:
                    index_path = f"./models/{model_name}"
                    if not os.path.exists(index_path):
                        index_path = "./models/efficientnet_b3_loli_optimized_v2_20260529_133654"
                    faiss_path = f"{index_path}.faiss"
                    mapping_path = f"{index_path}_mapping.json"
                    if os.path.exists(faiss_path) and os.path.exists(mapping_path):
                        classifier = Classification(index_path, threshold=0.1)
                    else:
                        classifier = Classification(threshold=0.1)
                    model_cache.set(model_name, classifier)

                if classifier.index is not None:
                    role, similarity = classifier.classify(feature, 5, attributes, threshold=threshold)
                else:
                    role, similarity = "unknown", 0.0

                role_full_info = get_role_info(role)
                result = {
                    "filename": file.filename, "role": role,
                    "role_cn": role_full_info.get("cn", role),
                    "role_jp": role_full_info.get("jp", ""),
                    "role_anime": role_full_info.get("anime", ""),
                    "similarity": float(similarity), "attributes": attributes,
                }
                results.append(result)
            except Exception as e:
                logger.error(f"处理文件 {file.filename} 失败: {e}")
                results.append({"filename": file.filename, "error": str(e)})
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"批量预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量预测失败: {e}")


# 保存全局引用的字典
def set_globals(prep, feat_ext, tag, device):
    global preprocessor, feature_extractor, tagger, OPTIMAL_DEVICE
    preprocessor = prep
    feature_extractor = feat_ext
    tagger = tag
    OPTIMAL_DEVICE = device
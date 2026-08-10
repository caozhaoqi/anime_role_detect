"""
模型服务路由
"""
import os
import sys
import time
import asyncio
import gc
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import Response
from PIL import Image
import io

from src.core.logging import get_enhanced_logger as get_logger
from src.services.model_service.classifiers import EfficientNetClassifier
from src.services.cache_service.cache_service import model_cache
from src.core.utils.role_info_loader import get_role_info
from src.core.utils.utils import safe_temp_path as _safe_temp_path
from src.services.cache_service.redis_cache import get_redis_cache
from src.services.model.recognition_service import get_recognition_service
from src.models.recognition_record import RecognitionRecordCreate
from src.core.config.service_config import get_service_config as _get_svc_config

# 模块级别导入 WDViTV3Tagger，确保在请求处理之前完成导入和 torch 初始化
# 避免在首次请求时触发 MPS C++ 后端 mutex 死锁
try:
    from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
except Exception:
    WDViTV3Tagger = None

logger = get_logger("model_service.routes")


router = APIRouter()

# 拆分出的独立域路由（2026-08-09）：健康检查 + Grad-CAM/角色列表/纠错反馈。
# 子模块仅只读共享状态（函数内延迟 import 本模块），不参与写入竞争。
from .routes_health import router as _health_router
from .routes_feedback import router as _feedback_router
router.include_router(_health_router)
router.include_router(_feedback_router)

# 全局引用，由 app.py 在初始化时设置
preprocessor = None
feature_extractor = None
tagger = None
_efficientnet_classifier = None
keypoint_pool = None  # KeypointWorkerPool 实例

# P2-1: 全局复用的 ThreadPoolExecutor（不在函数内创建新实例）
_executor: ThreadPoolExecutor = None

model_init_lock = asyncio.Lock()
OPTIMAL_DEVICE = "cpu"


def _cleanup_after_inference():
    """推理后自动 GC，防止内存碎片化 (P2)

    在重型端点（classify / detect-multiple / batch-predict）完成后调用，
    释放 PyTorch 缓存的显存/统一内存。
    """
    import gc
    gc.collect()
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _append_role_records(endpoint: str, filename: str, roles, debug_boxes=None):
    """Task B：把每个角色的明细写成一行 JSON 追加到结构化日志，使重启后可回查。

    所有请求都会写入 per-role 明细；debug=True 时额外写一行 debug_boxes。
    写入失败不影响主流程（诊断日志，best-effort）。
    """
    try:
        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"anime_role_detect_structured_{today}.jsonl")

        ts = datetime.now().isoformat()
        lines = []
        for r in roles:
            rec = {
                "type": "role_record",
                "endpoint": endpoint,
                "filename": filename,
                "timestamp": ts,
                "role": r.get("role"),
                "confidence": r.get("confidence"),
                "class_id": r.get("class_id"),
                "fallback": bool(r.get("fallback", False)),
                "bbox": r.get("box") or r.get("bbox"),
            }
            lines.append(json.dumps(rec, ensure_ascii=False))

        if debug_boxes is not None:
            dbg = {
                "type": "debug_boxes",
                "endpoint": endpoint,
                "filename": filename,
                "timestamp": ts,
                "boxes": debug_boxes,
            }
            lines.append(json.dumps(dbg, ensure_ascii=False))

        with open(log_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    except Exception:
        # 诊断日志写入失败不应影响主流程
        pass


# 延迟导入核心模块
def import_core_modules():
    pass  # 由 app.py 初始化时调用


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


async def _run_ocr_and_nsfw(image, content, filename):
    """运行 OCR 和 NSFW 检测（可选增强，失败不阻塞主流程）

    在 Model Service 内部完成 OCR + NSFW 检测，避免 API Service 重复推理。

    Args:
        image: PIL Image 对象（用于 OCR）
        content: 原始图片字节（用于 NSFW 临时文件写入）
        filename: 原始文件名

    Returns:
        tuple: (text_detections, nsfw_result)
    """
    text_detections = []
    nsfw_result = {"is_nsfw": False, "skin_ratio": 0.0, "method": "default"}

    # OCR 检测（使用预加载的 EasyOCR，缩放图片加速 + 超时保护）
    try:
        from src.core.ocr.easyocr_detector import get_ocr_detector
        ocr_detector = get_ocr_detector()
        if ocr_detector.is_ready():
            # 缩放图片加速 OCR（最长边 ≤ 768px，大幅降低处理时间）
            ocr_image = image.copy()
            w, h = ocr_image.size
            max_dim = max(w, h)
            if max_dim > 768:
                scale = 768.0 / max_dim
                ocr_image = ocr_image.resize((int(w * scale), int(h * scale)))
                logger.debug(f"OCR 图片缩放: {w}x{h} → {ocr_image.size[0]}x{ocr_image.size[1]}")

            loop = asyncio.get_running_loop()
            text_detections = await asyncio.wait_for(
                loop.run_in_executor(_executor, ocr_detector.detect_text, ocr_image),
                timeout=10
            )
        else:
            logger.warning("[DEGRADE] OCR 检测器未就绪，跳过文字检测")
    except asyncio.TimeoutError:
        logger.warning("[DEGRADE] OCR 检测超时（10s），跳过文字检测")
        text_detections = []
    except Exception as e:
        logger.warning(f"OCR 检测失败: {e}")

    # NSFW 检测（使用本地 OpenCV 检测 + 规则回退，不触发 HF 下载）
    nsfw_temp_path = None
    try:
        nsfw_temp_path = _safe_temp_path(filename or "unknown", "nsfw")
        with open(nsfw_temp_path, "wb") as f:
            f.write(content)
        from src.services.model.nsfw_detector import detect_nsfw
        loop = asyncio.get_running_loop()
        nsfw_result = await asyncio.wait_for(
            loop.run_in_executor(_executor, detect_nsfw, nsfw_temp_path),
            timeout=15
        )
    except asyncio.TimeoutError:
        logger.warning("[DEGRADE] NSFW 检测超时（15s），降级为安全默认值")
        nsfw_result = {"is_nsfw": False, "skin_ratio": 0.0, "method": "timeout"}
    except Exception as e:
        logger.warning(f"[DEGRADE] NSFW 检测失败: {e}")
        nsfw_result = {"is_nsfw": False, "skin_ratio": 0.0, "method": "error"}
    finally:
        if nsfw_temp_path and os.path.exists(nsfw_temp_path):
            try:
                os.remove(nsfw_temp_path)
            except Exception:
                pass

    return text_detections, nsfw_result


@router.post("/api/model/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form("efficientnet_b0"),
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
                return {"success": True, "data": cached_result}

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
                # P0-2: 使用常驻进程池替代每次 subprocess.run fork
                if keypoint_pool is not None:
                    keypoints = await keypoint_pool.detect_keypoints(image)
                else:
                    logger.warning("keypoint_pool 未初始化，跳过关键点检测")
                    keypoints = []
            except Exception as e:
                logger.warning(f"关键点检测失败: {e}")
                keypoints = []
        global _efficientnet_classifier

        if _efficientnet_classifier is None:
            try:
                loop = asyncio.get_running_loop()
                _efficientnet_classifier = await loop.run_in_executor(
                    _executor, EfficientNetClassifier.get_instance
                )
            except Exception as e:
                logger.error(f"EfficientNet分类器初始化失败: {e}")

        if _efficientnet_classifier is not None and _efficientnet_classifier.model is not None:
            direct_role, direct_confidence, feature = _efficientnet_classifier.classify_with_features(image)
            logger.info(f"EfficientNet直接分类: role={direct_role}, confidence={direct_confidence:.4f}")

            # 51类模型，0.05 以上即视为有效识别（随机概率 1/51 ≈ 0.02）
            if direct_confidence >= 0.05:
                role = direct_role
                similarity = direct_confidence
                attributes = []
                if use_attributes:
                    if tagger is None:
                        async with model_init_lock:
                            if tagger is None:
                                try:
                                    from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
                                    WDViTV3Tagger.reset_instance()
                                    tagger = WDViTV3Tagger.get_instance()
                                    loop = asyncio.get_running_loop()
                                    await loop.run_in_executor(_executor, lambda: tagger.load_model(force_reload=True))
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
                candidates = []
                try:
                    if _efficientnet_classifier is not None:
                        candidates = _efficientnet_classifier.classify_topk(image, k=5)
                except Exception:
                    pass
                result = {
                    "role": role,
                    "role_cn": role_full_info.get("cn", role),
                    "role_jp": role_full_info.get("jp", ""),
                    "role_anime": role_full_info.get("anime", ""),
                    "similarity": float(similarity),
                    "attributes": attributes,
                    "tags": [a.get("tag", "") for a in attributes if a.get("tag")] if attributes else [],
                    "candidates": candidates,
                    "keypoints": keypoints,
                    "feature": feature.tolist() if hasattr(feature, "tolist") else feature,
                }

                # 在 Model Service 内部完成 OCR + NSFW 检测，避免 API Service 重复推理
                text_detections, nsfw_result = await _run_ocr_and_nsfw(image, content, file.filename)
                result["text_detections"] = text_detections
                result["nsfw"] = nsfw_result

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
                return {"success": True, "data": result}

        from src.core.feature_extraction.feature_extraction import FeatureExtraction
        logger.info("EfficientNet置信度不足，降级到Faiss搜索...")
        if feature_extractor is None:
            async with model_init_lock:
                if feature_extractor is None:
                    loop = asyncio.get_running_loop()
                    feature_extractor = await loop.run_in_executor(_executor, FeatureExtraction)

        feature = feature_extractor.extract_features(processed_image)

        attributes = []
        if use_attributes:
            if tagger is None:
                async with model_init_lock:
                    if tagger is None:
                        try:
                            from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
                            WDViTV3Tagger.reset_instance()
                            tagger = WDViTV3Tagger.get_instance()
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(_executor, lambda: tagger.load_model(force_reload=True))
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
                index_path = "./models/efficientnet_b0"
                model_name = "efficientnet_b0"

            faiss_path = f"{index_path}.faiss"
            mapping_path = f"{index_path}_mapping.json"
            if os.path.exists(faiss_path) and os.path.exists(mapping_path):
                classifier = Classification(index_path, threshold=0.1)
            else:
                classifier = Classification(threshold=0.1)
            model_cache.setdefault(model_name, classifier)

        if classifier.index is not None and getattr(classifier.index, "ntotal", 0) > 0:
            role, similarity = classifier.classify(feature, 5, attributes)
        else:
            if classifier.index is not None and getattr(classifier.index, "ntotal", 0) == 0:
                logger.warning("[DEGRADE] FAISS 索引为空 (ntotal=0)，降级到本地模型分类")
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
        candidates = []
        try:
            if _efficientnet_classifier is not None:
                candidates = _efficientnet_classifier.classify_topk(image, k=5)
        except Exception:
            pass
        result = {
            "role": role,
            "role_cn": role_full_info.get("cn", role),
            "role_jp": role_full_info.get("jp", ""),
            "role_anime": role_full_info.get("anime", ""),
            "similarity": float(similarity),
            "attributes": attributes,
            "tags": [a.get("tag", "") for a in attributes if a.get("tag")] if attributes else [],
            "candidates": candidates,
            "keypoints": keypoints,
            "feature": feature.tolist() if hasattr(feature, "tolist") else feature,
        }

        # 在 Model Service 内部完成 OCR + NSFW 检测，避免 API Service 重复推理
        text_detections, nsfw_result = await _run_ocr_and_nsfw(image, content, file.filename)
        result["text_detections"] = text_detections
        result["nsfw"] = nsfw_result

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

        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        return {"success": false, "message": "模型预测失败", "data": {"role": "unknown", "similarity": 0.0, "tags": ["anime", "digital art"]}}
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        _cleanup_after_inference()


@router.post("/api/model/extract")
async def extract_features(file: UploadFile = File(...)):
    """特征提取"""
    global preprocessor, feature_extractor
    temp_path = None
    try:
        content = await file.read()
        temp_path = _safe_temp_path(file.filename, "extract")
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
    debug: bool = Form(False),
):
    """检测图片中的多个角色"""
    global preprocessor, feature_extractor, tagger
    temp_path = None
    try:
        content = await file.read()
        temp_path = _safe_temp_path(file.filename, "multirole")
        with open(temp_path, "wb") as f:
            f.write(content)

        try:
            from src.core.detection.multi_role_detection_enhanced import EnhancedMultiRoleDetector
            detector = EnhancedMultiRoleDetector(
                model_name=EfficientNetClassifier.MODEL_DIR_NAME,
                enable_open_set=True, enable_fuzzy_record=True,
                unknown_threshold=0.3, fuzzy_threshold=0.5,
            )
            detection_results = detector.detect_roles(temp_path, debug=debug)
        except Exception:
            from src.core.detection.multi_role_detection import MultiRoleDetector
            detector = MultiRoleDetector(model_name=EfficientNetClassifier.MODEL_DIR_NAME)
            detection_results = detector.detect_roles(temp_path)

        if feature_extractor is None:
            from src.core.feature_extraction.feature_extraction import FeatureExtraction
            feature_extractor = FeatureExtraction()

        if tagger is None:
            from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
            WDViTV3Tagger.reset_instance()
            tagger = WDViTV3Tagger.get_instance()
            tagger.load_model(force_reload=True)

        results = []
        full_image = Image.open(temp_path).convert("RGB")
        
        for i, detection in enumerate(detection_results[:max_characters]):
            role = detection.get("role", "unknown")
            similarity = detection.get("similarity", 0.0)
            bbox = detection.get("bbox", {})
            confidence = detection.get("confidence", 0.0)
            attributes = detection.get("attributes", [])

            if not attributes and tagger:
                # 从检测结果获取裁剪图像，或根据 bbox 重新裁剪
                role_image = detection.get("cropped_image")
                if role_image is None and bbox:
                    try:
                        x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
                        role_image = full_image.crop((x1, y1, x2, y2))
                    except Exception:
                        pass
                
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
            if "fallback" in detection:
                role_result["fallback"] = detection["fallback"]
            if "used_model" in detection:
                role_result["used_model"] = detection["used_model"]
            results.append(role_result)

        # 在 Model Service 内部完成 OCR + NSFW 检测，避免 API Service 重复推理
        multi_image = Image.open(temp_path).convert("RGB")
        text_detections, nsfw_result = await _run_ocr_and_nsfw(multi_image, content, file.filename)

        fallback_used = any(d.get("fallback", False) for d in detection_results)

        # B. 落盘 per-role 明细（所有请求都写；debug 时额外写 debug_boxes）
        _append_role_records("detect-multiple", file.filename, results, getattr(detector, "_debug_boxes", None) if debug else None)

        debug_payload = None
        if debug:
            from src.core.detection.debug_annotator import annotate
            debug_payload = {
                "enabled": True,
                "degraded_path": bool(getattr(detector, "_debug_degraded_path", False)),
                "yolo_total_boxes": int(getattr(detector, "_debug_total_boxes", 0)),
                "annotated_image": annotate(full_image, getattr(detector, "_debug_boxes", [])),
                "boxes": getattr(detector, "_debug_boxes", []),
            }

        return {
            "success": True,
            "data": {
                "roles": results,
                "count": len(results),
                "fallback": fallback_used,
                "text_detections": text_detections,
                "nsfw": nsfw_result,
                **({"debug": debug_payload} if debug_payload is not None else {}),
            },
        }
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
        _cleanup_after_inference()


@router.post("/api/model/detect-yolo")
async def detect_with_yolo(
    file: UploadFile = File(...),
    yolo_model: str = Form("yolov8n.pt"),
    person_conf_threshold: float = Form(0.2),
    max_detections: int = Form(10),
    debug: bool = Form(False),
):
    """使用 YOLOv8 进行多目标检测"""
    from src.core.detection.multi_target_detector import MultiTargetDetector
    temp_path = None
    try:
        content = await file.read()
        temp_path = _safe_temp_path(file.filename, "yolo")
        with open(temp_path, "wb") as f:
            f.write(content)

        device = OPTIMAL_DEVICE
        if not hasattr(detect_with_yolo, "_detector") or detect_with_yolo._detector is None:
            detect_with_yolo._detector = MultiTargetDetector(yolo_model=yolo_model, device=device)

        detector = detect_with_yolo._detector
        image = Image.open(temp_path).convert("RGB")
        results = detector.detect_and_classify(image, person_conf_threshold=person_conf_threshold, debug=debug)

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
                # detect-yolo 走纯 EfficientNet 模型分类，无 FAISS，恒为 True
                "used_model": True,
            })

        fallback_used = False
        if len(response_results) == 0:
            fallback_crop = image.resize((224, 224), Image.BILINEAR)
            role_pred = detector._classify_crop(fallback_crop)
            role_name = role_pred.get("role", "unknown")
            role_full_info = get_role_info(role_name)
            response_results.append({
                "id": 1, "role": role_name,
                "role_cn": role_full_info.get("cn", role_name),
                "role_jp": role_full_info.get("jp", ""),
                "role_anime": role_full_info.get("anime", ""),
                "confidence": float(role_pred.get("confidence", 0.0)),
                "person_confidence": 0.0,
                "bbox": [0, 0, image.width, image.height],
                "class_id": role_pred.get("class_id", -1),
                "fallback": True,
                "used_model": True,
            })
            fallback_used = True

        # B. 落盘 per-role 明细（所有请求都写；debug 时额外写 debug_boxes）
        _append_role_records("detect-yolo", file.filename, response_results, results.get("debug_boxes") if debug else None)

        debug_payload = None
        if debug:
            from src.core.detection.debug_annotator import annotate
            debug_payload = {
                "enabled": True,
                "degraded_path": fallback_used,
                "yolo_total_boxes": int(results.get("yolo_total_boxes", 0)),
                "annotated_image": annotate(image, results.get("debug_boxes", [])),
                "boxes": results.get("debug_boxes", []),
            }

        return {
            "success": True,
            "data": {
                "roles": response_results, "count": len(response_results),
                "image_size": results.get("image_size", []),
                "detector": "YOLOv8 + EfficientNet", "model": yolo_model,
                "fallback": fallback_used,
                **({"debug": debug_payload} if debug_payload is not None else {}),
            }
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
        _cleanup_after_inference()


@router.post("/api/classify")
async def classify_image(
    file: UploadFile = File(...),
    use_coreml: bool = Form(False),
    use_model: bool = Form(True),
    use_attributes: bool = Form(True),
    use_keypoints: bool = Form(False),
    model_name: str = Form("efficientnet_b0"),
    cache_bypass: bool = Form(False),
):
    """分类图像（兼容前端调用）"""
    return await predict_image(file=file, model_name=model_name, use_attributes=use_attributes, use_keypoints=use_keypoints)


@router.post("/api/model/batch-predict")
async def batch_predict_images(
    files: List[UploadFile] = File(...),
    model_name: str = Form("efficientnet_b0"),
    use_attributes: bool = Form(True),
    batch_size: int = Form(8),
    multilabel: bool = Form(False),
    threshold: float = Form(0.4),
):
    """批量预测多张图像（P1-1: 并行读取 + 批量推理）"""
    global preprocessor, feature_extractor, tagger
    results = []
    try:
        _ensure_torch_imported()

        # P1-1: 从 ServiceConfig 读取 batch_size
        svc_config = _get_svc_config()
        effective_batch_size = batch_size or svc_config.INFERENCE_BATCH_SIZE

        # P1-1: 并行读取所有文件
        async def _read_file(f: UploadFile) -> tuple:
            content = await f.read()
            return (f, content)

        file_contents = await asyncio.gather(*[_read_file(f) for f in files])

        # 并行预处理所有图像
        valid_items = []  # [(file, content, image), ...]
        for file, content in file_contents:
            try:
                image = Image.open(io.BytesIO(content)).convert("RGB")
                valid_items.append((file, content, image))
            except Exception as e:
                logger.error(f"读取文件 {file.filename} 失败: {e}")
                results.append({"filename": file.filename, "error": str(e)})

        if not valid_items:
            return {"success": True, "results": results, "count": len(results)}

        # 确保预处理器已初始化
        if preprocessor is None:
            async with model_init_lock:
                if preprocessor is None:
                    from src.core.preprocessing.preprocessing import Preprocessing
                    preprocessor = Preprocessing()

        # 并行预处理
        def _preprocess_one(image):
            return preprocessor.preprocess(image)

        loop = asyncio.get_running_loop()
        processed_images = await loop.run_in_executor(
            _executor, lambda: [_preprocess_one(img) for _, _, img in valid_items]
        )

        # 过滤预处理失败的
        batch_images = []
        batch_meta = []  # [(file, image, attributes, processed), ...]
        for i, (file, content, image) in enumerate(valid_items):
            processed = processed_images[i]
            if processed is None:
                results.append({"filename": file.filename, "error": "图像预处理失败"})
                continue

            attributes = []
            if use_attributes and tagger:
                try:
                    attributes = tagger.generate_tags(image)
                except Exception:
                    pass

            batch_images.append(image)
            batch_meta.append((file, image, attributes, processed))

        if not batch_images:
            return {"success": True, "results": results, "count": len(results)}

        # P1-1: 使用 EfficientNet 批量推理
        global _efficientnet_classifier
        if _efficientnet_classifier is None:
            try:
                _efficientnet_classifier = await loop.run_in_executor(
                    _executor, EfficientNetClassifier.get_instance
                )
            except Exception as e:
                logger.error(f"EfficientNet分类器初始化失败: {e}")

        if _efficientnet_classifier is not None and _efficientnet_classifier.model is not None:
            # 批量分类
            batch_roles = _efficientnet_classifier.classify_batch(
                batch_images, batch_size=effective_batch_size
            )

            for i, (file, image, attributes, processed) in enumerate(batch_meta):
                role_info = batch_roles[i]
                role = role_info["role"]
                similarity = role_info["confidence"]

                role_full_info = get_role_info(role)
                result = {
                    "filename": file.filename, "role": role,
                    "role_cn": role_full_info.get("cn", role),
                    "role_jp": role_full_info.get("jp", ""),
                    "role_anime": role_full_info.get("anime", ""),
                    "similarity": float(similarity), "attributes": attributes,
                }
                results.append(result)
        else:
            # 降级：串行使用 Faiss
            from src.core.classification.classification import Classification
            from src.core.feature_extraction.feature_extraction import FeatureExtraction

            if feature_extractor is None:
                async with model_init_lock:
                    if feature_extractor is None:
                        feature_extractor = await loop.run_in_executor(_executor, FeatureExtraction)

            classifier = model_cache.get(model_name)
            if classifier is None:
                index_path = f"./models/{model_name}"
                if not os.path.exists(index_path):
                    index_path = "./models/efficientnet_b3"
                faiss_path = f"{index_path}.faiss"
                mapping_path = f"{index_path}_mapping.json"
                if os.path.exists(faiss_path) and os.path.exists(mapping_path):
                    classifier = Classification(index_path, threshold=0.1)
                else:
                    classifier = Classification(threshold=0.1)
                model_cache.setdefault(model_name, classifier)

            for i, (file, image, attributes, processed) in enumerate(batch_meta):
                try:
                    feature = feature_extractor.extract_features(processed)
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

        return {"success": True, "results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"批量预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量预测失败: {e}")
    finally:
        _cleanup_after_inference()


# 保存全局引用的字典
def set_globals(prep, feat_ext, tag, device, kp_pool=None):
    global preprocessor, feature_extractor, tagger, OPTIMAL_DEVICE, keypoint_pool, _executor
    preprocessor = prep
    feature_extractor = feat_ext
    tagger = tag
    OPTIMAL_DEVICE = device
    keypoint_pool = kp_pool
    # P2-1: 初始化全局 ThreadPoolExecutor（仅初始化一次）
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=4)
        logger.info("全局 ThreadPoolExecutor 已初始化 (max_workers=4)")


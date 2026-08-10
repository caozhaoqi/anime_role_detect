"""
模型服务 - Grad-CAM / 角色列表 / 纠错反馈路由（从 routes.py 拆出的独立域，2026-08-09）

- gradcam 复用主模块的全局 _executor（只读引用，函数内延迟 import）。
- roles / feedback 不依赖共享状态，仅用 EfficientNetClassifier 单例与数据集注册表。
"""
import os
import io
import asyncio
import json

from fastapi import APIRouter, UploadFile, File, Form, Body
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

from src.core.logging import get_enhanced_logger as get_logger
from src.core.detection.gradcam import GradCAMGenerator
from src.services.model_service.classifiers import EfficientNetClassifier

logger = get_logger("model_service.routes.feedback")

router = APIRouter()


@router.post("/api/model/gradcam")
async def gradcam_endpoint(file: UploadFile = File(...), target_class: int = Form(None)):
    """生成 Grad-CAM 热力图（懒加载 FP32 模型副本）"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        loop = asyncio.get_running_loop()

        def _generate():
            gen = GradCAMGenerator.get_instance()
            return gen.generate(image, target_class)

        # 复用主模块全局线程池（只读引用）
        from src.services.model_service import routes as R
        result = await loop.run_in_executor(R._executor, _generate)

        if "error" in result:
            logger.warning(f"GradCAM 生成失败: {result['error']}")
            return {"code": 1, "message": result["error"]}

        # cam_raw 是 np.ndarray，不可 JSON 序列化，从响应中移除
        result.pop("cam_raw", None)
        return {"code": 0, "data": result}
    except Exception as e:
        logger.error(f"GradCAM 端点异常: {e}", exc_info=True)
        return {"code": 1, "message": str(e)}


@router.get("/api/model/roles")
async def get_roles():
    """返回角色标签列表（以数据集注册表 class_registry_v2.json 为准，含中文名）"""
    try:
        from src.core.utils.role_info_loader import get_role_info

        names = []
        registry_path = os.path.join(project_root, "configs", "class_registry_v2.json")
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                reg = json.load(f)
            for c in reg.get("classes", []):
                if c.get("status") == "ACTIVE":
                    info = get_role_info(c["name"])
                    names.append({
                        "name": c["name"],
                        "cn": info.get("cn") or c.get("name"),
                        "anime": info.get("anime") or c.get("booru_tag") or "",
                    })
        # 兜底：注册表缺失时回退到当前模型类别
        if not names:
            clf = EfficientNetClassifier.get_instance()
            idx_to_class = clf.idx_to_class or {}
            for idx, name in sorted(idx_to_class.items(), key=lambda x: int(x[0])):
                info = get_role_info(name)
                names.append({
                    "name": name,
                    "cn": info.get("cn") or name,
                    "anime": info.get("anime") or "",
                })
        return {"code": 0, "data": {"roles": names, "total": len(names)}}
    except Exception as e:
        logger.error(f"获取角色列表异常: {e}", exc_info=True)
        return {"code": 1, "message": str(e)}


@router.post("/api/model/feedback")
async def feedback_endpoint(payload: dict = Body(...)):
    """用户纠错反馈，落盘 JSONL"""
    try:
        corrected_label = payload.get("corrected_label")
        if not corrected_label:
            return {"code": 1, "message": "缺少 corrected_label 字段"}

        # 校验 corrected_label ∈ 当前模型类别 ∪ 数据集注册表（与图片/模型类别对齐，#4）
        valid_labels = set(EfficientNetClassifier.get_instance().idx_to_class.values())
        registry_path = os.path.join(project_root, "configs", "class_registry_v2.json")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    reg = json.load(f)
                for c in reg.get("classes", []):
                    if c.get("status") == "ACTIVE":
                        valid_labels.add(c["name"])
            except Exception:
                pass
        if corrected_label not in valid_labels:
            return {
                "code": 1,
                "message": f"corrected_label '{corrected_label}' 不在 {len(valid_labels)} 类标签中",
            }

        # 图像缓存与时间戳均需用到 datetime（将局部导入前置，确保缓存块可用）
        from datetime import datetime

        # Phase2: 图像缓存 —— 把纠错对应的原图落地磁盘，让 image_ref 指向真实文件（支撑增量训练）
        image_data = payload.get("image_data")
        if image_data:
            try:
                import base64 as _b64
                if "," in image_data:
                    _header, _b64data = image_data.split(",", 1)
                else:
                    _b64data = image_data
                _img_bytes = _b64data.encode("utf-8") if isinstance(_b64data, str) else _b64data
                _img_bytes = _b64.b64decode(_img_bytes)
                _img_dir = os.path.join(project_root, "data", "feedback_images")
                os.makedirs(_img_dir, exist_ok=True)
                _rid = payload.get("recognition_id") or datetime.now().strftime("%Y%m%d%H%M%S")
                _img_path = os.path.join(_img_dir, f"{_rid}.jpg")
                with open(_img_path, "wb") as _f:
                    _f.write(_img_bytes)
                payload["image_ref"] = f"data/feedback_images/{_rid}.jpg"
            except Exception as _ie:
                logger.warning(f"反馈图像缓存失败（仅影响后续增量训练）: {_ie}")

        # 补充服务端时间戳（datetime 已在上方函数内导入）
        payload["server_timestamp"] = datetime.now().isoformat()

        # 落盘 logs/feedback/feedback_<date>.jsonl（append）
        log_dir = os.path.join(project_root, "logs", "feedback")
        os.makedirs(log_dir, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join(log_dir, f"feedback_{today}.jsonl")

        # Phase2: 落盘 JSONL 不再冗余存储 base64 原图（image_ref 已指向真实文件）
        payload.pop("image_data", None)

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as write_err:
            logger.error(f"反馈落盘失败: {write_err}")
            return {"code": 1, "message": f"反馈记录写入失败: {write_err}"}

        logger.info(f"用户反馈已记录: corrected_label={corrected_label} → {log_path}")
        return {"code": 0, "message": "反馈已记录"}
    except Exception as e:
        logger.error(f"反馈端点异常: {e}", exc_info=True)
        return {"code": 1, "message": str(e)}

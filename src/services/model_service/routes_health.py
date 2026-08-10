"""
模型服务 - 健康检查路由（从 routes.py 拆出的独立域，2026-08-09）

只读共享状态（preprocessor / feature_extractor / _efficientnet_classifier），
通过函数内延迟 import 主模块读取，避免循环依赖且不产生写入竞争。
"""
import os

from fastapi import APIRouter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

from src.core.logging import get_enhanced_logger as get_logger
from src.services.model_service.classifiers import EfficientNetClassifier
from src.core.version import APP_VERSION

logger = get_logger("model_service.routes.health")

router = APIRouter()


@router.get("/api/health")
async def health_check():
    """健康检查 - 返回结构化状态（模型文件存在 + 模型已加载）。"""
    # 延迟读取主模块共享状态（只读）
    from src.services.model_service import routes as R

    # 1. EfficientNet-B3 模型文件是否存在
    model_best = os.path.join(project_root, "models", EfficientNetClassifier.MODEL_DIR_NAME, "model_best.pth")
    model_file_ok = os.path.exists(model_best)
    checks = {
        "model_file": {
            "status": "ok" if model_file_ok else "missing",
            "path": model_best,
        },
        # 2. 模型是否已加载（服务内全局变量 / 单例非空）
        "preprocessor_loaded": R.preprocessor is not None,
        "feature_extractor_loaded": R.feature_extractor is not None,
        "classifier_loaded": R._efficientnet_classifier is not None,
    }

    if not model_file_ok:
        overall = "unhealthy"
    elif R.preprocessor is None:
        overall = "degraded"
    else:
        overall = "healthy"

    return {
        "status": overall,
        "service": "Model Service",
        "version": APP_VERSION,
        "checks": checks,
    }


@router.get("/live")
async def liveness_check():
    """K8s liveness 端点 - 进程存活检查"""
    return {"status": "alive"}


@router.get("/ready")
async def readiness_check():
    """K8s readiness 端点 - 模型就绪检查"""
    from src.services.model_service import routes as R

    checks = {"model_service": True}
    ready = True

    # 检查预处理器是否已初始化
    if R.preprocessor is None:
        checks["preprocessor"] = False
        ready = False
    else:
        checks["preprocessor"] = True

    # 检查特征提取器是否已初始化
    if R.feature_extractor is None:
        checks["feature_extractor"] = False
        ready = False
    else:
        checks["feature_extractor"] = True

    status_code = 200 if ready else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )


@router.get("/model_service")
async def root():
    return {"message": "Model Service", "docs": "/docs"}

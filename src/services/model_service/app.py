#!/usr/bin/env python3
"""
模型服务主文件 - 只负责初始化和生命周期管理
"""
import os
import sys
import platform
import asyncio
import time
import secrets
from concurrent.futures import ThreadPoolExecutor

IS_MACOS = platform.system() == "Darwin"

if IS_MACOS:
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# MPS 已通过 DeviceManager 自动管理，不再强制禁用
# 保留 PYTORCH_ENABLE_MPS_FALLBACK=1 作为安全网（上方第 16 行）

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import Request, HTTPException
from contextlib import asynccontextmanager
import uvicorn

from src.core.config.service_config import get_service_config
from src.core.config.device_manager import DeviceManager
from src.core.logging.global_logger import get_logger

config = get_service_config()
logger = None

# HEIF/HEIC解码器
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


@asynccontextmanager
async def model_service_lifespan(app: FastAPI):
    """模型服务生命周期管理"""
    global logger, Image, preprocessor, feature_extractor, tagger, keypoint_pool
    from src.core.logging.global_logger import get_logger as gl
    from PIL import Image as Img

    Image = Img
    logger = gl("model_service")
    logger.info("启动模型服务")

    # 使用 DeviceManager 替代旧的 get_optimal_device
    OPTIMAL_DEVICE = DeviceManager.get_device()
    logger.info(f"推理设备: {OPTIMAL_DEVICE}")

    # 初始化关键点检测进程池
    from src.services.model_service.keypoint_worker import KeypointWorkerPool
    keypoint_pool = KeypointWorkerPool(num_workers=config.KEYPOINT_WORKER_COUNT)
    keypoint_pool.start()
    logger.info(f"关键点检测进程池已启动 (workers={config.KEYPOINT_WORKER_COUNT})")

    set_globals(preprocessor, feature_extractor, tagger, OPTIMAL_DEVICE, keypoint_pool)
    await init_models()
    logger.info("模型服务启动完成")
    yield
    logger.info("模型服务关闭")
    if keypoint_pool is not None:
        keypoint_pool.shutdown()
        logger.info("关键点检测进程池已关闭")


# 创建应用
app = FastAPI(
    title="Model Service",
    description="Anime Role Detect Model Service",
    version="1.0.0",
    lifespan=model_service_lifespan,
)

# CORS - 同主 API 一样，allow_credentials=True 与 origins=["*"] 不兼容
allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if os.environ.get("CORS_ALLOWED_ORIGINS") else ["*"]
allow_credentials = False if allowed_origins == ["*"] else True
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=allow_credentials, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ========== 内部服务认证中间件 ==========
_INTERNAL_TOKEN = os.environ.get("INTERNAL_SERVICE_TOKEN", "")
_INTERNAL_IPS = {ip.strip() for ip in os.environ.get("INTERNAL_IPS", "127.0.0.1,localhost,::1").split(",") if ip.strip()}
_INTERNAL_EXEMPT_PATHS = {"/api/health", "/live", "/ready"}


@app.middleware("http")
async def internal_auth_middleware(request: Request, call_next):
    """内部服务认证：要求请求来自内部IP或持有内部服务令牌。

    模型服务是内部推理服务，不应直接暴露给外部。
    仅 /api/health、/live、/ready 免认证。
    """
    # 健康检查端点免认证
    if request.url.path in _INTERNAL_EXEMPT_PATHS:
        return await call_next(request)

    # 检查客户端IP
    client_host = request.client.host if request.client else "unknown"
    # 支持 IPv6 回环地址映射
    if client_host not in _INTERNAL_IPS and client_host not in {"127.0.0.1", "::1", "localhost"}:
        # 检查内部服务令牌
        token = request.headers.get("X-Internal-Service-Token", "")
        if _INTERNAL_TOKEN and not secrets.compare_digest(token, _INTERNAL_TOKEN):
            logger.warning(f"未授权的模型服务访问: IP={client_host}, Path={request.url.path}")
            raise HTTPException(status_code=403, detail="Forbidden: internal service only")

    return await call_next(request)


# 路由
from src.services.model_service.routes import router as model_router, set_globals
app.include_router(model_router)


def get_optimal_device():
    """自动选择最佳计算设备（委托给 DeviceManager）"""
    return DeviceManager.get_device()


OPTIMAL_DEVICE = get_optimal_device()

# 全局变量
preprocessor = None
feature_extractor = None
tagger = None
keypoint_pool = None
model_init_lock = asyncio.Lock()


async def init_models():
    """初始化模型并预热"""
    global preprocessor, feature_extractor, tagger
    try:
        from src.core.preprocessing.preprocessing import Preprocessing
        preprocessor = Preprocessing()
        logger.info("预处理器初始化完成")
        asyncio.create_task(warmup_models())
        logger.info("模型服务启动完成，模型预热任务已启动")
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")


async def warmup_models():
    """模型预热

    预加载 WD ViT Tagger、EasyOCR、NSFW 检测器，
    使首次请求不再等待模型下载/初始化。
    预加载失败不阻止服务启动，仅记录 [DEGRADE] 日志。
    """
    global feature_extractor, tagger
    try:
        logger.info("开始模型预热...")
        start_time = time.time()
        dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        logger.info("预热特征提取器...")
        if feature_extractor is None:
            async with model_init_lock:
                if feature_extractor is None:
                    from src.core.feature_extraction.feature_extraction import FeatureExtraction
                    loop = asyncio.get_running_loop()
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        feature_extractor = await loop.run_in_executor(executor, FeatureExtraction)

        processed = preprocessor.preprocess(dummy_image)
        if processed is not None:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                await loop.run_in_executor(executor, feature_extractor.extract_features, processed)
            logger.info("特征提取器预热完成")

        # ---- WD ViT Tagger：改为懒加载 + TTL 卸载（P1 优化）----
        # 不再在启动时预加载，首次使用时自动加载，空闲 5 分钟后自动卸载释放 ~700MB
        logger.info("WD ViT Tagger 已切换为懒加载模式（首次使用时加载，空闲 5 分钟后卸载）")

        # ---- EasyOCR：改为懒加载 + TTL 卸载（P1 优化）----
        # 不再在启动时预加载，首次使用时自动加载，空闲 3 分钟后自动卸载释放 ~200MB
        logger.info("EasyOCR 已切换为懒加载模式（首次使用时加载，空闲 3 分钟后卸载）")

        # ---- NSFW 检测器 ----
        # 注意：NSFW 模型从 HF 下载在当前网络环境不可用（TCP 层面卡死，asyncio.wait_for 无法穿透），
        # 直接跳过预热，请求时 _run_ocr_and_nsfw 会降级返回 is_nsfw=False 默认值
        logger.info("NSFW 检测器预热跳过（HF 网络不可用），请求时将返回安全默认值")


        elapsed = time.time() - start_time
        logger.info(f"模型预热完成，耗时: {elapsed:.2f}秒")
        # 启动 TTL 空闲卸载检查后台任务
        asyncio.create_task(_ttl_unload_checker())
        # 预热完成后同步全局变量到路由模块
        set_globals(preprocessor, feature_extractor, tagger, OPTIMAL_DEVICE, keypoint_pool)
    except Exception as e:
        logger.error(f"模型预热失败: {e}")


async def _ttl_unload_checker():
    """后台任务：每 60 秒检查一次 WD ViT Tagger 和 EasyOCR 是否空闲超时需要卸载"""
    while True:
        await asyncio.sleep(60)
        try:
            # 通过单例模式获取 tagger（routes.py 懒加载的也是同一个单例）
            try:
                from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
                t = WDViTV3Tagger.get_instance()
                if t.unload_if_idle():
                    logger.info("[TTL] WD ViT Tagger 已自动卸载")
            except Exception:
                pass
            from src.core.ocr.easyocr_detector import get_ocr_detector
            ocr = get_ocr_detector()
            if ocr.unload_if_idle():
                logger.info("[TTL] EasyOCR 已自动卸载")
        except Exception as e:
            logger.debug(f"TTL 检查出错: {e}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="模型服务")
    parser.add_argument("--host", type=str, default=config.MODEL_SERVICE_HOST, help="服务主机")
    parser.add_argument("--port", type=int, default=config.MODEL_SERVICE_PORT, help="服务端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    args = parser.parse_args()
    uvicorn.run("app:app", host=args.host, port=args.port, workers=args.workers,
                timeout_keep_alive=30, limit_concurrency=config.UVICORN_LIMIT_CONCURRENCY)

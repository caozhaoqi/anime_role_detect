#!/usr/bin/env python3
"""
模型服务主文件 - 只负责初始化和生命周期管理
"""
import os
import sys
import platform
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

IS_MACOS = platform.system() == "Darwin"

if IS_MACOS:
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_MPS_DISABLE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==================== MPS 强制禁用补丁 ====================
# macOS MPS 后端存在 mutex 死锁问题（[mutex.cc : 452] RAW: Lock blocking）
# 该死锁由 macOS 内核级信号量残留导致，无法通过环境变量完全清理
# 通过在 torch 初始化前 monkey-patch _mps_is_available 来彻底禁用 MPS
# 注意：此补丁必须在任何模块导入 torch 之前执行
try:
    import torch._C
    # 检查是否已有 MPS 可用性检查函数
    if hasattr(torch._C, '_mps_is_available'):
        torch._C._mps_is_available = lambda: False
except Exception:
    pass
# ==========================================================

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import uvicorn

from src.core.config.service_config import get_service_config
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
    global logger, Image, preprocessor, feature_extractor, tagger
    from src.core.logging.global_logger import get_logger as gl
    from PIL import Image as Img

    Image = Img
    logger = gl("model_service")
    logger.info("启动模型服务")
    set_globals(preprocessor, feature_extractor, tagger, OPTIMAL_DEVICE)
    await init_models()
    logger.info("模型服务启动完成")
    yield
    logger.info("模型服务关闭")


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

# 路由
from src.services.model_service.routes import router as model_router, set_globals
app.include_router(model_router)


def get_optimal_device():
    """自动选择最佳计算设备"""
    try:
        import torch
        # 如果环境变量禁用了MPS，跳过MPS检测
        if os.environ.get("PYTORCH_MPS_DISABLE", "0") == "1":
            if torch.cuda.is_available():
                print("✅ 检测到CUDA可用，将使用NVIDIA GPU加速")
                return "cuda"
            print("⚠️ MPS已被环境变量禁用，将使用CPU推理")
            return "cpu"
        if IS_MACOS and torch.backends.mps.is_available():
            print("✅ 检测到MPS可用，将使用Apple Silicon GPU加速")
            return "mps"
        if torch.cuda.is_available():
            print("✅ 检测到CUDA可用，将使用NVIDIA GPU加速")
            return "cuda"
        print("⚠️ 未检测到GPU，将使用CPU推理（速度较慢）")
        return "cpu"
    except ImportError:
        print("⚠️ PyTorch未安装，将使用CPU")
        return "cpu"


OPTIMAL_DEVICE = get_optimal_device()

# 全局变量
preprocessor = None
feature_extractor = None
tagger = None
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
    """模型预热"""
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

        # 标签生成器跳过预热（按需加载），
        # 避免 transformers 加载 SmilingWolf/wd-vit-tagger-v3 时触发 MPS mutex 死锁
        logger.info("标签生成器跳过预热（按需加载）")

        elapsed = time.time() - start_time
        logger.info(f"模型预热完成，耗时: {elapsed:.2f}秒")
        # 预热完成后同步全局变量到路由模块
        set_globals(preprocessor, feature_extractor, tagger, OPTIMAL_DEVICE)
    except Exception as e:
        logger.error(f"模型预热失败: {e}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="模型服务")
    parser.add_argument("--host", type=str, default=config.MODEL_SERVICE_HOST, help="服务主机")
    parser.add_argument("--port", type=int, default=config.MODEL_SERVICE_PORT, help="服务端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    args = parser.parse_args()
    uvicorn.run("app:app", host=args.host, port=args.port, workers=args.workers,
                timeout_keep_alive=30, limit_concurrency=10)
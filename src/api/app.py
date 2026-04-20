from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import time

# 初始化日志记录器
from src.core.logging.global_logger import get_logger
logger = get_logger("api")

# 设置Hugging Face和Keras缓存目录为项目目录
hf_cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'huggingface_cache')
keras_cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'keras_cache')
os.environ['HF_HOME'] = hf_cache_dir
os.environ['KERAS_HOME'] = keras_cache_dir

# 创建缓存目录
os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(keras_cache_dir, exist_ok=True)

# 从环境变量中读取配置
USE_MODEL_SERVICE = os.environ.get('USE_MODEL_SERVICE', 'false').lower() == 'true'
MODEL_SERVICE_URL = os.environ.get('MODEL_SERVICE_URL', 'http://localhost:8001')

# 创建FastAPI应用实例
app = FastAPI(
    title="Anime Role Detect API",
    description="Anime role detection API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置为具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 导入监控中间件
try:
    from src.middleware.monitoring import monitoring_middleware, get_service_monitor
    app.middleware("http")(monitoring_middleware)
except Exception as e:
    logger.error(f"导入监控中间件失败: {e}")

# 导入图像处理服务
try:
    from src.services.processor.image_processor import process_single_image, process_batch_images
except Exception as e:
    logger.error(f"导入图像处理服务失败: {e}")

# 导入模型加载服务
try:
    from src.services.processor.model_loader import load_models
except Exception as e:
    logger.error(f"导入模型加载服务失败: {e}")

# 导入缓存服务
try:
    from src.services.cache_service import init_cache_manager
except Exception as e:
    logger.error(f"导入缓存服务失败: {e}")


@app.post("/api/classify")
async def classify_image(
    file: UploadFile = File(...),
    model_name: str = Form("resnet50"),
    use_coreml: bool = Form(False),
    use_model: bool = Form(False),
    use_attributes: bool = Form(False),
    cache_bypass: bool = Form(False)
):
    """
    分类图像中的角色
    
    Args:
        file: 上传的图像文件
        model_name: 模型名称
        use_coreml: 是否使用 CoreML 模型（Mac 平台）
        use_model: 是否使用专用模型
        use_attributes: 是否使用属性预测
        cache_bypass: 是否绕过缓存
    
    Returns:
        dict: 分类结果
    """
    try:
        # 初始化缓存管理器
        init_cache_manager()
        
        # 加载模型
        load_models()
        
        # 处理图像
        result = await process_single_image(file, model_name, cache_bypass, use_coreml, use_model, use_attributes)
        
        # 构建响应
        response = {
            "success": True,
            "data": result,
            "message": "图像分类成功"
        }
        
        return response
    except Exception as e:
        logger.error(f"分类图像失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch_classify")
async def batch_classify_images(
    files: list[UploadFile] = File(...),
    model_name: str = Form("resnet50"),
    use_coreml: bool = Form(False),
    use_model: bool = Form(False),
    use_attributes: bool = Form(False),
    cache_bypass: bool = Form(False)
):
    """
    批量分类图像中的角色
    
    Args:
        files: 上传的图像文件列表
        model_name: 模型名称
        use_coreml: 是否使用 CoreML 模型（Mac 平台）
        use_model: 是否使用专用模型
        use_attributes: 是否使用属性预测
        cache_bypass: 是否绕过缓存
    
    Returns:
        list: 分类结果列表
    """
    try:
        # 初始化缓存管理器
        init_cache_manager()
        
        # 加载模型
        load_models()
        
        # 处理图像
        results = await process_batch_images(files, model_name, cache_bypass, use_coreml, use_model, use_attributes)
        
        # 构建响应
        response = {
            "success": True,
            "data": results,
            "message": f"成功处理 {len(results)} 个图像"
        }
        
        return response
    except Exception as e:
        logger.error(f"批量分类图像失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """
    健康检查
    
    Returns:
        dict: 健康状态
    """
    return {
        "status": "healthy",
        "service": "Anime Role Detect API",
        "version": "1.0.0",
        "timestamp": time.time()
    }


@app.get("/api/monitoring")
async def get_monitoring_info():
    """
    获取监控信息
    
    Returns:
        dict: 监控信息
    """
    monitor_info = get_service_monitor()
    return {
        "status": "ok",
        "monitoring": monitor_info
    }


@app.get("/api/models")
async def get_available_models():
    """
    获取可用的模型列表
    
    Returns:
        dict: 模型列表
    """
    # 检查models目录下的模型
    model_dirs = []
    models_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models')
    if os.path.exists(models_path):
        model_dirs = [d for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]
    
    return {
        "success": True,
        "models": model_dirs,
        "default_model": "default"
    }

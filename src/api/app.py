import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
USE_MODEL_SERVICE = os.environ.get('USE_MODEL_SERVICE', 'True').lower() == 'True'
MODEL_SERVICE_URL = os.environ.get('MODEL_SERVICE_URL', 'http://localhost:8888')

# 创建FastAPI应用实例
app = FastAPI(
    title="Anime Role Detect API",
    description="Anime role detection API - 用于检测和分类动画角色的API服务",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置为具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 导入错误处理
from src.core.error.error_handler import global_exception_handler
app.add_exception_handler(Exception, global_exception_handler)


# 导入监控中间件
try:
    from src.middleware.monitoring import monitoring_middleware, get_service_monitor
    app.middleware("http")(monitoring_middleware)
except Exception as e:
    logger.error(f"导入监控中间件失败: {e}")

# 导入图像处理服务
try:
    from src.services.processor.image_processor import process_single_image, process_batch_images, process_multi_role_image
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
    model_name: str = Form("resnet18_loli8"),
    use_coreml: bool = Form(False),
    use_model: bool = Form(True),
    use_attributes: bool = Form(True),
    cache_bypass: bool = Form(False),
    multi_role: bool = Form(False),
    use_deepdanbooru: bool = Form(True)
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
        multi_role: 是否使用多角色检测
        use_deepdanbooru: 是否使用DeepDanbooru标签提取
    
    Returns:
        dict: 分类结果
    """
    try:
        # 初始化缓存管理器
        init_cache_manager()
        
        # 加载模型
        load_models()
        
        # 处理图像
        if multi_role:
            result = await process_multi_role_image(file, model_name, cache_bypass, use_coreml, use_model, use_attributes, use_deepdanbooru)
        else:
            result = await process_single_image(file, model_name, cache_bypass, use_coreml, use_model, use_attributes, use_deepdanbooru)
        
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
        from src.core.error.error_handler import raise_app_error
        from src.core.error.error_codes import ErrorCode
        raise_app_error(
            error_code=ErrorCode.CLASSIFICATION_FAILED,
            status_code=500,
            detail=str(e)
        )


@app.post("/api/classify/multi-role")
async def multi_role_classify_image(
    file: UploadFile = File(...),
    model_name: str = Form("resnet18_loli8"),
    use_coreml: bool = Form(False),
    use_model: bool = Form(True),
    use_attributes: bool = Form(True),
    cache_bypass: bool = Form(False),
    use_deepdanbooru: bool = Form(True)
):
    """
    多角色检测
    
    Args:
        file: 上传的图像文件
        model_name: 模型名称
        use_coreml: 是否使用 CoreML 模型（Mac 平台）
        use_model: 是否使用专用模型
        use_attributes: 是否使用属性预测
        cache_bypass: 是否绕过缓存
        use_deepdanbooru: 是否使用DeepDanbooru标签提取
    
    Returns:
        dict: 多角色检测结果
    """
    return await classify_image(
        file=file,
        model_name=model_name,
        use_coreml=use_coreml,
        use_model=use_model,
        use_attributes=use_attributes,
        cache_bypass=cache_bypass,
        multi_role=True,
        use_deepdanbooru=use_deepdanbooru
    )


@app.post("/api/batch_classify")
async def batch_classify_images(
    files: list[UploadFile] = File(...),
    model_name: str = Form("resnet18_loli8"),
    use_coreml: bool = Form(False),
    use_model: bool = Form(False),
    use_attributes: bool = Form(False),
    cache_bypass: bool = Form(False),
    use_deepdanbooru: bool = Form(True),
    max_concurrency: int = Form(4)
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
        use_deepdanbooru: 是否使用DeepDanbooru标签提取
        max_concurrency: 最大并发数
    
    Returns:
        list: 分类结果列表
    """
    try:
        # 初始化缓存管理器
        init_cache_manager()
        
        # 加载模型
        load_models()
        
        # 限制并发数范围
        max_concurrency = max(1, min(max_concurrency, 10))  # 限制在1-10之间
        
        # 处理图像
        results = await process_batch_images(
            files, 
            model_name, 
            cache_bypass=cache_bypass, 
            use_coreml=use_coreml, 
            use_model=use_model, 
            use_attributes=use_attributes,
            use_deepdanbooru=use_deepdanbooru,
            max_concurrency=max_concurrency
        )
        
        # 构建响应
        response = {
            "success": True,
            "data": results,
            "message": f"成功处理 {len(results)} 个图像",
            "stats": {
                "total_files": len(files),
                "success_count": sum(1 for r in results if r.get('success', False)),
                "max_concurrency": max_concurrency
            }
        }
        
        return response
    except Exception as e:
        logger.error(f"批量分类图像失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        from src.core.error.error_handler import raise_app_error
        from src.core.error.error_codes import ErrorCode
        raise_app_error(
            error_code=ErrorCode.CLASSIFICATION_FAILED,
            status_code=500,
            detail=str(e)
        )


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
    # 从项目根目录开始计算路径：src/api -> src -> 项目根目录 -> models
    models_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    # 规范化路径
    models_path = os.path.normpath(models_path)
    logger.info(f"查找模型路径: {models_path}")
    logger.info(f"路径是否存在: {os.path.exists(models_path)}")
    
    if os.path.exists(models_path):
        model_dirs = [d for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]
        logger.info(f"找到模型目录: {model_dirs}")
    else:
        logger.warning(f"模型目录不存在: {models_path}")
    
    return {
        "success": True,
        "models": model_dirs,
        "default_model": "default"
    }


@app.get("/api/docs/info")
async def get_api_docs():
    """
    获取API文档信息
    
    Returns:
        dict: API文档信息
    """
    return {
        "success": True,
        "api_name": "Anime Role Detect API",
        "version": "1.0.0",
        "description": "用于检测和分类动画角色的API服务",
        "endpoints": [
            {
                "path": "/api/classify",
                "method": "POST",
                "description": "分类图像中的角色",
                "parameters": [
                    {"name": "file", "type": "UploadFile", "required": True, "description": "上传的图像文件"},
                    {"name": "model_name", "type": "string", "default": "resnet18_loli8", "description": "模型名称"},
                    {"name": "use_coreml", "type": "bool", "default": False, "description": "是否使用CoreML模型（Mac平台）"},
                    {"name": "use_model", "type": "bool", "default": True, "description": "是否使用专用模型"},
                    {"name": "use_attributes", "type": "bool", "default": True, "description": "是否使用属性预测"},
                    {"name": "cache_bypass", "type": "bool", "default": False, "description": "是否绕过缓存"},
                    {"name": "multi_role", "type": "bool", "default": False, "description": "是否使用多角色检测"},
                    {"name": "use_deepdanbooru", "type": "bool", "default": True, "description": "是否使用DeepDanbooru标签提取"}
                ],
                "response": {
                    "success": "bool",
                    "data": "object",
                    "message": "string"
                }
            },
            {
                "path": "/api/classify/multi-role",
                "method": "POST",
                "description": "多角色检测",
                "parameters": [
                    {"name": "file", "type": "UploadFile", "required": True, "description": "上传的图像文件"},
                    {"name": "model_name", "type": "string", "default": "resnet18_loli8", "description": "模型名称"},
                    {"name": "use_coreml", "type": "bool", "default": False, "description": "是否使用CoreML模型（Mac平台）"},
                    {"name": "use_model", "type": "bool", "default": True, "description": "是否使用专用模型"},
                    {"name": "use_attributes", "type": "bool", "default": True, "description": "是否使用属性预测"},
                    {"name": "cache_bypass", "type": "bool", "default": False, "description": "是否绕过缓存"},
                    {"name": "use_deepdanbooru", "type": "bool", "default": True, "description": "是否使用DeepDanbooru标签提取"}
                ],
                "response": {
                    "success": "bool",
                    "data": "object",
                    "message": "string"
                }
            },
            {
                "path": "/api/batch_classify",
                "method": "POST",
                "description": "批量分类图像中的角色",
                "parameters": [
                    {"name": "files", "type": "list[UploadFile]", "required": True, "description": "上传的图像文件列表"},
                    {"name": "model_name", "type": "string", "default": "resnet18_loli8", "description": "模型名称"},
                    {"name": "use_coreml", "type": "bool", "default": False, "description": "是否使用CoreML模型（Mac平台）"},
                    {"name": "use_model", "type": "bool", "default": False, "description": "是否使用专用模型"},
                    {"name": "use_attributes", "type": "bool", "default": False, "description": "是否使用属性预测"},
                    {"name": "cache_bypass", "type": "bool", "default": False, "description": "是否绕过缓存"},
                    {"name": "use_deepdanbooru", "type": "bool", "default": True, "description": "是否使用DeepDanbooru标签提取"},
                    {"name": "max_concurrency", "type": "int", "default": 4, "description": "最大并发数"}
                ],
                "response": {
                    "success": "bool",
                    "data": "list",
                    "message": "string",
                    "stats": "object"
                }
            },
            {
                "path": "/api/health",
                "method": "GET",
                "description": "健康检查",
                "response": {
                    "status": "string",
                    "service": "string",
                    "version": "string",
                    "timestamp": "number"
                }
            },
            {
                "path": "/api/monitoring",
                "method": "GET",
                "description": "获取监控信息",
                "response": {
                    "status": "string",
                    "monitoring": "object"
                }
            },
            {
                "path": "/api/models",
                "method": "GET",
                "description": "获取可用的模型列表",
                "response": {
                    "success": "bool",
                    "models": "list",
                    "default_model": "string"
                }
            },
            {
                "path": "/api/feedback",
                "method": "POST",
                "description": "提交用户反馈",
                "parameters": [
                    {"name": "feedback_type", "type": "string", "required": True, "description": "反馈类型: bug, suggestion, question, other"},
                    {"name": "message", "type": "string", "required": True, "description": "反馈内容"},
                    {"name": "image_id", "type": "string", "default": "", "description": "相关图像ID"},
                    {"name": "contact", "type": "string", "default": "", "description": "联系方式"}
                ],
                "response": {
                    "success": "bool",
                    "message": "string",
                    "feedback_id": "string"
                }
            },
            {
                "path": "/api/docs/info",
                "method": "GET",
                "description": "获取API文档信息",
                "response": {
                    "success": "bool",
                    "api_name": "string",
                    "version": "string",
                    "description": "string",
                    "endpoints": "list"
                }
            }
        ],
        "documentation": {
            "swagger_ui": "/api/docs",
            "redoc": "/api/redoc",
            "openapi_json": "/api/openapi.json"
        }
    }


@app.post("/api/feedback")
async def submit_feedback(
    feedback_type: str = Form(..., description="反馈类型: bug, suggestion, question, other"),
    message: str = Form(..., description="反馈内容"),
    image_id: str = Form("", description="相关图像ID"),
    contact: str = Form("", description="联系方式")
):
    """
    提交用户反馈
    
    Args:
        feedback_type: 反馈类型
        message: 反馈内容
        image_id: 相关图像ID
        contact: 联系方式
    
    Returns:
        dict: 反馈提交结果
    """
    try:
        # 验证反馈类型
        valid_types = ['bug', 'suggestion', 'question', 'other']
        if feedback_type not in valid_types:
            return {
                "success": False,
                "message": f"无效的反馈类型，有效值: {', '.join(valid_types)}"
            }
        
        # 生成反馈ID
        import uuid
        import time
        feedback_id = f"fb_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # 保存反馈
        feedback_data = {
            "id": feedback_id,
            "type": feedback_type,
            "message": message,
            "image_id": image_id,
            "contact": contact,
            "timestamp": time.time()
        }
        
        # 记录日志
        logger.info(f"收到用户反馈: {feedback_data}")
        
        # 这里可以添加保存到数据库或文件的逻辑
        # 暂时只记录日志
        
        return {
            "success": True,
            "message": "反馈提交成功",
            "feedback_id": feedback_id
        }
    except Exception as e:
        logger.error(f"提交反馈失败: {e}")
        return {
            "success": False,
            "message": "反馈提交失败，请稍后重试"
        }

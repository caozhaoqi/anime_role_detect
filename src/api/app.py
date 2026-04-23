import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import time

# 初始化日志记录器
from src.core.logging.global_logger import get_logger
logger = get_logger("api")

# 使用统一配置
from src.core.config.service_config import get_service_config
config = get_service_config()

# 设置Hugging Face和Keras缓存目录
os.environ['HF_HOME'] = config.HF_CACHE_DIR
os.environ['KERAS_HOME'] = config.KERAS_CACHE_DIR

# 创建缓存目录
os.makedirs(config.HF_CACHE_DIR, exist_ok=True)
os.makedirs(config.KERAS_CACHE_DIR, exist_ok=True)

# 从统一配置中读取（兼容环境变量）
USE_MODEL_SERVICE = config.USE_MODEL_SERVICE
MODEL_SERVICE_URL = config.MODEL_SERVICE_URL

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

# 导入认证服务
from src.services.auth_service import init_auth_service, authenticate_user, create_access_token, create_refresh_token, verify_token
from src.middleware.auth import auth_middleware, get_current_user, get_current_admin

# 导入认证中间件
try:
    app.middleware("http")(auth_middleware)
except Exception as e:
    logger.error(f"导入认证中间件失败: {e}")

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
from src.services.cache_service import init_cache_manager, get_cache_stats

# 导入监控服务
from src.services.monitoring_service import init_monitoring_service, get_monitoring_service, monitor_request

# 导入消息队列服务
from src.services.message_queue_service import init_message_queue_service, send_message

# 导入熔断器服务
from src.services.circuit_breaker_service import init_circuit_breaker_service, execute_with_fallback

# 导入模型版本管理服务
from src.services.model_version_service import init_model_version_service, get_model_versions, get_model_path, register_model, enable_ab_test, disable_ab_test, get_ab_test_config, select_model_for_ab_test, update_model_description, delete_model_version

# 导入多模型服务
from src.services.multi_model_service import init_multi_model_service, process_with_multiple_models, add_model, remove_model, set_fusion_strategy, get_model_configs, get_fusion_strategy, get_multi_model_service


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
        
        # 初始化监控服务
        init_monitoring_service()
        
        # 初始化消息队列服务
        init_message_queue_service()
        
        # 初始化熔断器服务
        init_circuit_breaker_service()
        
        # 初始化模型版本管理服务
        init_model_version_service()
        
        # 初始化多模型服务
        init_multi_model_service()
        
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
        
        # 初始化监控服务
        init_monitoring_service()
        
        # 初始化消息队列服务
        init_message_queue_service()
        
        # 初始化熔断器服务
        init_circuit_breaker_service()
        
        # 初始化模型版本管理服务
        init_model_version_service()
        
        # 初始化多模型服务
        init_multi_model_service()
        
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


@app.post("/api/auth/login")
async def login(
    username: str = Form(..., description="用户名"),
    password: str = Form(..., description="密码")
):
    """
    用户登录
    
    Args:
        username: 用户名
        password: 密码
    
    Returns:
        dict: 登录结果，包含访问令牌和刷新令牌
    """
    try:
        # 初始化认证服务
        init_auth_service()
        
        # 验证用户
        user = authenticate_user(username, password)
        if not user:
            return {
                "success": False,
                "message": "用户名或密码错误"
            }
        
        # 创建访问令牌和刷新令牌
        access_token = create_access_token(data={"sub": username, "role": user.get("role")})
        refresh_token = create_refresh_token(data={"sub": username})
        
        return {
            "success": True,
            "message": "登录成功",
            "data": {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "username": username,
                "role": user.get("role")
            }
        }
    except Exception as e:
        logger.error(f"登录失败: {e}")
        return {
            "success": False,
            "message": "登录失败，请稍后重试"
        }


@app.post("/api/auth/refresh")
async def refresh_token(
    refresh_token: str = Form(..., description="刷新令牌")
):
    """
    刷新访问令牌
    
    Args:
        refresh_token: 刷新令牌
    
    Returns:
        dict: 刷新结果，包含新的访问令牌
    """
    try:
        # 验证刷新令牌
        payload = verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return {
                "success": False,
                "message": "无效的刷新令牌"
            }
        
        # 获取用户名
        username = payload.get("sub")
        if not username:
            return {
                "success": False,
                "message": "无效的刷新令牌"
            }
        
        # 初始化认证服务
        from src.services.auth_service import get_auth_service
        auth_service = get_auth_service()
        
        # 获取用户信息
        user = auth_service.users.get(username)
        if not user:
            return {
                "success": False,
                "message": "用户不存在"
            }
        
        # 创建新的访问令牌
        new_access_token = create_access_token(data={"sub": username, "role": user.get("role")})
        
        return {
            "success": True,
            "message": "令牌刷新成功",
            "data": {
                "access_token": new_access_token
            }
        }
    except Exception as e:
        logger.error(f"刷新令牌失败: {e}")
        return {
            "success": False,
            "message": "刷新令牌失败，请稍后重试"
        }


@app.get("/api/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    获取当前用户信息
    
    Returns:
        dict: 当前用户信息
    """
    try:
        return {
            "success": True,
            "message": "获取用户信息成功",
            "data": {
                "username": current_user.get("sub"),
                "role": current_user.get("role")
            }
        }
    except Exception as e:
        logger.error(f"获取用户信息失败: {e}")
        return {
            "success": False,
            "message": "获取用户信息失败，请稍后重试"
        }


@app.get("/api/admin/test")
async def admin_test(current_admin: dict = Depends(get_current_admin)):
    """
    管理员测试端点
    
    Returns:
        dict: 测试结果
    """
    try:
        return {
            "success": True,
            "message": "管理员访问成功",
            "data": {
                "username": current_admin.get("sub"),
                "role": current_admin.get("role")
            }
        }
    except Exception as e:
        logger.error(f"管理员测试失败: {e}")
        return {
            "success": False,
            "message": "管理员测试失败，请稍后重试"
        }


@app.get("/api/model-versions")
async def get_model_versions_list(model_name: Optional[str] = None):
    """
    获取模型版本列表
    
    Args:
        model_name: 模型名称（可选）
    
    Returns:
        dict: 模型版本列表
    """
    try:
        versions = get_model_versions(model_name)
        return {
            "success": True,
            "message": "获取模型版本成功",
            "data": versions
        }
    except Exception as e:
        logger.error(f"获取模型版本失败: {e}")
        return {
            "success": False,
            "message": "获取模型版本失败，请稍后重试"
        }


@app.post("/api/model-versions/register")
async def register_model_version(
    model_name: str = Form(..., description="模型名称"),
    version: str = Form(..., description="模型版本"),
    path: str = Form(..., description="模型路径"),
    description: str = Form("", description="模型描述"),
    current_admin: dict = Depends(get_current_admin)
):
    """
    注册模型版本
    
    Args:
        model_name: 模型名称
        version: 模型版本
        path: 模型路径
        description: 模型描述
    
    Returns:
        dict: 注册结果
    """
    try:
        success = register_model(model_name, version, path, description)
        if success:
            return {
                "success": True,
                "message": "模型版本注册成功"
            }
        else:
            return {
                "success": False,
                "message": "模型版本注册失败"
            }
    except Exception as e:
        logger.error(f"注册模型版本失败: {e}")
        return {
            "success": False,
            "message": "注册模型版本失败，请稍后重试"
        }


@app.get("/api/model-versions/path")
async def get_model_version_path(
    model_name: str = Form(..., description="模型名称"),
    version: str = Form("latest", description="模型版本")
):
    """
    获取模型路径
    
    Args:
        model_name: 模型名称
        version: 模型版本（默认latest）
    
    Returns:
        dict: 模型路径
    """
    try:
        path = get_model_path(model_name, version)
        if path:
            return {
                "success": True,
                "message": "获取模型路径成功",
                "data": {
                    "path": path
                }
            }
        else:
            return {
                "success": False,
                "message": "模型路径不存在"
            }
    except Exception as e:
        logger.error(f"获取模型路径失败: {e}")
        return {
            "success": False,
            "message": "获取模型路径失败，请稍后重试"
        }


@app.post("/api/model-versions/ab-test/enable")
async def enable_ab_test_endpoint(
    test_models: str = Form(..., description="测试模型列表，格式: model1:v1,model2:v2"),
    weights: str = Form(..., description="权重列表，格式: 0.3,0.7"),
    control_model: str = Form(..., description="对照组模型"),
    current_admin: dict = Depends(get_current_admin)
):
    """
    启用A/B测试
    
    Args:
        test_models: 测试模型列表，格式: model1:v1,model2:v2
        weights: 权重列表，格式: 0.3,0.7
        control_model: 对照组模型
    
    Returns:
        dict: 启用结果
    """
    try:
        # 解析测试模型列表
        test_models_list = []
        for model_str in test_models.split(","):
            if ":" in model_str:
                model_name, version = model_str.split(":")
                test_models_list.append((model_name.strip(), version.strip()))
        
        # 解析权重列表
        weights_list = [float(w.strip()) for w in weights.split(",")]
        
        # 启用A/B测试
        enable_ab_test(test_models_list, weights_list, control_model)
        
        return {
            "success": True,
            "message": "A/B测试启用成功"
        }
    except Exception as e:
        logger.error(f"启用A/B测试失败: {e}")
        return {
            "success": False,
            "message": "启用A/B测试失败，请稍后重试"
        }


@app.post("/api/model-versions/ab-test/disable")
async def disable_ab_test_endpoint(current_admin: dict = Depends(get_current_admin)):
    """
    禁用A/B测试
    
    Returns:
        dict: 禁用结果
    """
    try:
        disable_ab_test()
        return {
            "success": True,
            "message": "A/B测试禁用成功"
        }
    except Exception as e:
        logger.error(f"禁用A/B测试失败: {e}")
        return {
            "success": False,
            "message": "禁用A/B测试失败，请稍后重试"
        }


@app.get("/api/model-versions/ab-test/config")
async def get_ab_test_config_endpoint():
    """
    获取A/B测试配置
    
    Returns:
        dict: A/B测试配置
    """
    try:
        config = get_ab_test_config()
        return {
            "success": True,
            "message": "获取A/B测试配置成功",
            "data": config
        }
    except Exception as e:
        logger.error(f"获取A/B测试配置失败: {e}")
        return {
            "success": False,
            "message": "获取A/B测试配置失败，请稍后重试"
        }


@app.get("/api/model-versions/ab-test/select")
async def select_model_for_ab_test_endpoint():
    """
    为A/B测试选择模型
    
    Returns:
        dict: 选择的模型
    """
    try:
        model_name, version = select_model_for_ab_test()
        return {
            "success": True,
            "message": "选择模型成功",
            "data": {
                "model_name": model_name,
                "version": version
            }
        }
    except Exception as e:
        logger.error(f"选择模型失败: {e}")
        return {
            "success": False,
            "message": "选择模型失败，请稍后重试"
        }


@app.post("/api/model-versions/update-description")
async def update_model_description_endpoint(
    model_name: str = Form(..., description="模型名称"),
    version: str = Form(..., description="模型版本"),
    description: str = Form(..., description="模型描述"),
    current_admin: dict = Depends(get_current_admin)
):
    """
    更新模型描述
    
    Args:
        model_name: 模型名称
        version: 模型版本
        description: 模型描述
    
    Returns:
        dict: 更新结果
    """
    try:
        success = update_model_description(model_name, version, description)
        if success:
            return {
                "success": True,
                "message": "模型描述更新成功"
            }
        else:
            return {
                "success": False,
                "message": "模型描述更新失败"
            }
    except Exception as e:
        logger.error(f"更新模型描述失败: {e}")
        return {
            "success": False,
            "message": "更新模型描述失败，请稍后重试"
        }


@app.post("/api/model-versions/delete")
async def delete_model_version_endpoint(
    model_name: str = Form(..., description="模型名称"),
    version: str = Form(..., description="模型版本"),
    current_admin: dict = Depends(get_current_admin)
):
    """
    删除模型版本
    
    Args:
        model_name: 模型名称
        version: 模型版本
    
    Returns:
        dict: 删除结果
    """
    try:
        success = delete_model_version(model_name, version)
        if success:
            return {
                "success": True,
                "message": "模型版本删除成功"
            }
        else:
            return {
                "success": False,
                "message": "模型版本删除失败"
            }
    except Exception as e:
        logger.error(f"删除模型版本失败: {e}")
        return {
            "success": False,
            "message": "删除模型版本失败，请稍后重试"
        }


@app.post("/api/classify/multi-model")
async def multi_model_classify_image(
    file: UploadFile = File(...),
    multi_role: bool = Form(False)
):
    """
    使用多个模型进行分类
    
    Args:
        file: 上传的图像文件
        multi_role: 是否使用多角色检测
    
    Returns:
        dict: 多模型融合的分类结果
    """
    try:
        # 读取文件内容
        content = await file.read()
        
        # 使用多模型处理
        result = await get_multi_model_service().process_with_multiple_models(file, content, multi_role)
        
        # 构建响应
        response = {
            "success": True,
            "data": result,
            "message": "多模型分类成功"
        }
        
        return response
    except Exception as e:
        logger.error(f"多模型分类失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        from src.core.error.error_handler import raise_app_error
        from src.core.error.error_codes import ErrorCode
        raise_app_error(
            error_code=ErrorCode.CLASSIFICATION_FAILED,
            status_code=500,
            detail=str(e)
        )


@app.get("/api/multi-model/config")
async def get_multi_model_config():
    """
    获取多模型配置
    
    Returns:
        dict: 多模型配置
    """
    try:
        configs = get_model_configs()
        strategy = get_fusion_strategy()
        return {
            "success": True,
            "message": "获取多模型配置成功",
            "data": {
                "models": configs,
                "fusion_strategy": strategy
            }
        }
    except Exception as e:
        logger.error(f"获取多模型配置失败: {e}")
        return {
            "success": False,
            "message": "获取多模型配置失败，请稍后重试"
        }


@app.post("/api/multi-model/add")
async def add_multi_model(
    model_name: str = Form(..., description="模型名称"),
    model_type: str = Form(..., description="模型类型 (local 或 service)"),
    weight: float = Form(..., description="模型权重"),
    current_admin: dict = Depends(get_current_admin)
):
    """
    添加模型到多模型集成
    
    Args:
        model_name: 模型名称
        model_type: 模型类型 (local 或 service)
        weight: 模型权重
    
    Returns:
        dict: 添加结果
    """
    try:
        add_model(model_name, model_type, weight)
        return {
            "success": True,
            "message": "模型添加成功"
        }
    except Exception as e:
        logger.error(f"添加模型失败: {e}")
        return {
            "success": False,
            "message": "添加模型失败，请稍后重试"
        }


@app.post("/api/multi-model/remove")
async def remove_multi_model(
    model_name: str = Form(..., description="模型名称"),
    current_admin: dict = Depends(get_current_admin)
):
    """
    从多模型集成中移除模型
    
    Args:
        model_name: 模型名称
    
    Returns:
        dict: 移除结果
    """
    try:
        remove_model(model_name)
        return {
            "success": True,
            "message": "模型移除成功"
        }
    except Exception as e:
        logger.error(f"移除模型失败: {e}")
        return {
            "success": False,
            "message": "移除模型失败，请稍后重试"
        }


@app.post("/api/multi-model/strategy")
async def set_multi_model_strategy(
    strategy: str = Form(..., description="融合策略 (weighted_average, majority_vote, max_confidence)"),
    current_admin: dict = Depends(get_current_admin)
):
    """
    设置多模型融合策略
    
    Args:
        strategy: 融合策略 (weighted_average, majority_vote, max_confidence)
    
    Returns:
        dict: 设置结果
    """
    try:
        set_fusion_strategy(strategy)
        return {
            "success": True,
            "message": "融合策略设置成功"
        }
    except Exception as e:
        logger.error(f"设置融合策略失败: {e}")
        return {
            "success": False,
            "message": "设置融合策略失败，请稍后重试"
        }

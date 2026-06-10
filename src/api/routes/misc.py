"""
反馈、API文档信息和配置路由
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

import time
import uuid
import json
from fastapi import APIRouter, Form, Request

from src.core.logging.global_logger import get_logger

logger = get_logger("api.routes.feedback")

router = APIRouter()


@router.post("/api/feedback")
async def submit_feedback(
    feedback_type: str = Form(...),
    message: str = Form(...),
    image_id: str = Form(""),
    contact: str = Form(""),
):
    """提交用户反馈"""
    try:
        valid_types = ["bug", "suggestion", "question", "other"]
        if feedback_type not in valid_types:
            return {"success": False, "message": f"无效的反馈类型，有效值: {', '.join(valid_types)}"}
        feedback_id = f"fb_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        feedback_data = {
            "id": feedback_id, "type": feedback_type, "message": message,
            "image_id": image_id, "contact": contact, "timestamp": time.time(),
        }
        logger.info(f"收到用户反馈: {feedback_data}")
        return {"success": True, "message": "反馈提交成功", "feedback_id": feedback_id}
    except Exception as e:
        logger.error(f"提交反馈失败: {e}")
        return {"success": False, "message": "反馈提交失败，请稍后重试"}


@router.get("/api/docs/info")
async def get_api_docs():
    """获取API文档信息"""
    return {
        "success": True,
        "api_name": "Anime Role Detect API",
        "version": "1.0.0",
        "description": "用于检测和分类动画角色的API服务",
        "endpoints": [
            {"path": "/api/classify", "method": "POST", "description": "分类图像中的角色"},
            {"path": "/api/classify/async", "method": "POST", "description": "异步分类图像中的角色"},
            {"path": "/api/classify/multi-role", "method": "POST", "description": "多角色检测"},
            {"path": "/api/batch_classify", "method": "POST", "description": "批量分类图像中的角色"},
            {"path": "/api/classify/multi-model", "method": "POST", "description": "多模型融合分类"},
            {"path": "/api/health", "method": "GET", "description": "健康检查"},
            {"path": "/api/health/detailed", "method": "GET", "description": "详细健康检查"},
            {"path": "/api/models", "method": "GET", "description": "获取可用的模型列表"},
            {"path": "/api/monitoring", "method": "GET", "description": "获取监控信息"},
            {"path": "/api/model-versions", "method": "GET", "description": "获取模型版本列表"},
            {"path": "/api/auth/login", "method": "POST", "description": "用户登录"},
            {"path": "/api/auth/refresh", "method": "POST", "description": "刷新访问令牌"},
            {"path": "/api/auth/me", "method": "GET", "description": "获取当前用户信息"},
            {"path": "/api/history", "method": "GET", "description": "获取识别历史记录"},
            {"path": "/api/feedback", "method": "POST", "description": "提交用户反馈"},
        ],
        "documentation": {
            "swagger_ui": "/api/docs",
            "redoc": "/api/redoc",
            "openapi_json": "/api/openapi.json",
        },
    }


@router.get("/api/config")
async def get_config(request: Request):
    """获取前端配置信息"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "../../frontend/app/config/config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                frontend_config = json.load(f)
        else:
            frontend_config = {
                "ui": {
                    "theme": "light", "enableDarkMode": True,
                    "animateTransitions": True, "showPlatformInfo": True,
                    "enableNotifications": True,
                },
                "features": {
                    "enableModelSelection": True, "enableCoremlSwitch": True,
                    "enableAttributesSwitch": True, "enableMultiRoleSwitch": True,
                    "enableHistoryPanel": True, "enableDragDrop": True,
                    "enableCopyDownload": True, "enableImagePreview": True,
                },
                "api": {"baseUrl": "/api", "timeout": 30000, "retryCount": 3, "retryDelay": 1000},
                "messages": {
                    "welcomeMessage": "你好！我是动漫角色识别助手。请上传一张动漫角色图片，我将尝试识别出这个角色。",
                    "processingMessage": "正在识别...",
                    "errorMessage": "识别过程中出现错误，请重试。",
                    "successMessage": "识别完成！",
                    "loginSuccessMessage": "登录成功！",
                    "loginErrorMessage": "登录失败，请检查用户名和密码。",
                },
                "validation": {
                    "maxImageSize": 10485760,
                    "allowedFormats": ["image/jpeg", "image/png", "image/gif", "image/webp"],
                    "minImageDimension": 100,
                },
            }
        return {"success": True, "message": "获取配置成功", "data": frontend_config}
    except Exception as e:
        logger.error(f"获取配置失败: {e}")
        return {"success": False, "message": "获取配置失败，请稍后重试"}
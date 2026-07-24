"""
历史记录路由
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

from fastapi import APIRouter, Depends
from typing import Optional

from src.core.logging import get_enhanced_logger as get_logger
from src.middleware.auth_enhanced import get_optional_current_user
from src.services.model.recognition_service import get_recognition_service

logger = get_logger("api.routes.history")

router = APIRouter()


@router.get("/api/history")
async def get_recognition_history(current_user: Optional[dict] = Depends(get_optional_current_user)):
    """获取用户的识别历史记录"""
    try:
        user_id = current_user.get("sub", "anonymous") if current_user else "anonymous"
        recognition_service = get_recognition_service()
        records = recognition_service.get_records_by_user(user_id)
        records_data = []
        for record in records:
            records_data.append({
                "id": record.id,
                "image_filename": record.image_filename,
                "model_used": record.model_used,
                "processing_time": record.processing_time,
                "is_multi_role": record.is_multi_role,
                "nsfw_status": record.nsfw_status,
                "detected_text": record.detected_text,
                "recognition_result": record.recognition_result,
                "timestamp": record.timestamp.isoformat() if record.timestamp else None,
            })
        return {"success": True, "message": "获取识别历史成功", "data": records_data}
    except Exception as e:
        logger.error(f"获取识别历史失败: {e}")
        return {"success": False, "message": "获取识别历史失败，请稍后重试"}


@router.delete("/api/history/{record_id}")
async def delete_recognition_record(record_id: str, current_user: Optional[dict] = Depends(get_optional_current_user)):
    """删除识别记录"""
    try:
        recognition_service = get_recognition_service()
        success = recognition_service.delete_record(record_id)
        return {"success": True, "message": "删除记录成功"} if success else {"success": False, "message": "记录不存在"}
    except Exception as e:
        logger.error(f"删除记录失败: {e}")
        return {"success": False, "message": "删除记录失败，请稍后重试"}

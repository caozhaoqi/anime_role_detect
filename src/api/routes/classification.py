"""
分类相关路由
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

import time
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional

from src.core.logging.global_logger import get_logger
from src.middleware.auth_enhanced import get_current_user
from src.services.processor.image_processor import (
    process_single_image,
    process_batch_images,
)
from src.services.model.recognition_service import get_recognition_service
from src.models.recognition_record import RecognitionRecordCreate
from src.services.processor.feature_processor import generate_image_summary, convert_numpy_types

logger = get_logger("api.routes.classification")

router = APIRouter()


@router.post("/api/classify")
async def classify_image(
    file: UploadFile = File(...),
    model_name: str = Form("resnet18_loli8"),
    use_coreml: bool = Form(False),
    use_model: bool = Form(True),
    use_attributes: bool = Form(True),
    cache_bypass: bool = Form(False),
    multi_role: bool = Form(False),
    use_deepdanbooru: bool = Form(True),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """分类图像中的角色"""
    try:
        start_time = time.time()

        if multi_role:
            from src.services.processor.image_processor import process_multi_role_image
            result = await process_multi_role_image(
                file, model_name, cache_bypass, use_coreml,
                use_model, use_attributes, use_deepdanbooru,
            )
        else:
            result = await process_single_image(
                file, model_name, cache_bypass, use_coreml,
                use_model, use_attributes, use_deepdanbooru,
            )

        processing_time = time.time() - start_time

        summary = generate_image_summary(
            text_detections=result.get("text_detections", []),
            tags=result.get("tags", []),
            role_info=result.get("role_info")
                or result.get("classification")
                or result.get("ai_predicted_role"),
            attributes=result.get("attributes", []),
            nsfw_status=result.get("nsfw"),
        )
        result["summary"] = summary
        result = convert_numpy_types(result)

        try:
            recognition_service = get_recognition_service()
            user_id = current_user.get("sub") if current_user else "anonymous"
            username = current_user.get("username") if current_user else "anonymous"
            record = RecognitionRecordCreate(
                user_id=user_id,
                username=username,
                image_filename=file.filename,
                image_path="",
                recognition_result=result,
                model_used=model_name,
                processing_time=processing_time,
                is_multi_role=multi_role,
                nsfw_status=result.get("nsfw", {}).get("is_nsfw", False),
                detected_text=len(result.get("text_detections", [])) > 0,
            )
            recognition_service.create_record(record)
        except Exception as e:
            logger.warning(f"存储识别记录失败: {e}")

        return {"success": True, "data": result, "message": "图像分类成功"}
    except Exception as e:
        logger.error(f"分类图像失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        from src.core.error.error_handler import raise_app_error
        from src.core.error.error_codes import ErrorCode
        raise_app_error(error_code=ErrorCode.CLASSIFICATION_FAILED, status_code=500, detail=str(e))


@router.post("/api/classify/async")
async def classify_image_async(
    file: UploadFile = File(...),
    model_name: str = Form("resnet18_loli8"),
    use_coreml: bool = Form(False),
    use_model: bool = Form(True),
    use_attributes: bool = Form(True),
    cache_bypass: bool = Form(False),
    multi_role: bool = Form(False),
    use_deepdanbooru: bool = Form(True),
    current_user: dict = Depends(get_current_user),
):
    """异步分类图像中的角色"""
    try:
        image_content = await file.read()
        try:
            from src.tasks.classify_tasks import classify_image_task
            task = classify_image_task.delay(
                image_content=image_content,
                image_filename=file.filename,
                model_name=model_name,
                use_coreml=use_coreml,
                use_model=use_model,
                use_attributes=use_attributes,
                use_deepdanbooru=use_deepdanbooru,
                multi_role=multi_role,
            )
            logger.info(f"异步分类任务已提交: task_id={task.id}")
            return {
                "success": True,
                "task_id": task.id,
                "status": "pending",
                "message": "任务已提交，请通过 /api/task/{task_id} 查询结果",
            }
        except Exception as e:
            logger.warning(f"Celery任务提交失败，使用同步处理: {e}")
            return {"success": False, "message": "异步任务队列不可用，请使用同步接口", "error": str(e)}
    except Exception as e:
        logger.error(f"提交异步分类任务失败: {e}")
        return {"success": False, "message": "任务提交失败", "error": str(e)}


@router.get("/api/task/{task_id}")
async def get_task_result(task_id: str, current_user: dict = Depends(get_current_user)):
    """获取异步任务结果"""
    try:
        from src.tasks.celery_app import celery_app
        task = celery_app.AsyncResult(task_id)
        response = {"task_id": task_id, "status": task.state}
        if task.state == "SUCCESS":
            response.update({"success": True, "result": task.result, "message": "任务已完成"})
        elif task.state == "FAILURE":
            response.update({"success": False, "error": str(task.info), "message": "任务失败"})
        elif task.state == "PENDING":
            response.update({"success": True, "message": "任务正在排队或运行中"})
        elif task.state == "STARTED":
            response.update({"success": True, "message": "任务正在执行", "info": str(task.info) if task.info else None})
        else:
            response.update({"success": True, "message": f"任务状态: {task.state}", "info": str(task.info) if task.info else None})
        return response
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        return {"success": False, "task_id": task_id, "error": str(e), "message": "获取任务状态失败"}


@router.get("/api/task/{task_id}/result")
async def get_task_result_data(task_id: str, current_user: dict = Depends(get_current_user)):
    """获取异步任务结果数据"""
    from src.tasks.celery_app import celery_app
    task = celery_app.AsyncResult(task_id)
    if task.state == "SUCCESS":
        return {"success": True, "data": task.result}
    elif task.state == "FAILURE":
        raise HTTPException(status_code=500, detail=str(task.info))
    else:
        raise HTTPException(status_code=202, detail="Task not yet completed")


@router.post("/api/classify/multi-role")
async def multi_role_classify_image(
    file: UploadFile = File(...),
    model_name: str = Form("resnet18_loli8"),
    use_coreml: bool = Form(False),
    use_model: bool = Form(True),
    use_attributes: bool = Form(True),
    cache_bypass: bool = Form(False),
    use_deepdanbooru: bool = Form(True),
    current_user: dict = Depends(get_current_user),
):
    """多角色检测"""
    return await classify_image(
        file=file, model_name=model_name, use_coreml=use_coreml,
        use_model=use_model, use_attributes=use_attributes,
        cache_bypass=cache_bypass, multi_role=True,
        use_deepdanbooru=use_deepdanbooru, current_user=current_user,
    )


@router.post("/api/batch_classify")
async def batch_classify_images(
    files: list[UploadFile] = File(...),
    model_name: str = Form("resnet18_loli8"),
    use_coreml: bool = Form(False),
    use_model: bool = Form(False),
    use_attributes: bool = Form(False),
    cache_bypass: bool = Form(False),
    use_deepdanbooru: bool = Form(True),
    max_concurrency: int = Form(4),
):
    """批量分类图像中的角色"""
    try:
        # 服务已在 startup 事件中初始化，无需每次请求重复初始化 7 个服务
        # 仅检查模型是否就绪作为轻量级保活检查
        from src.services.processor.model_loader import _model_cache
        if not _model_cache:
            logger.warning("模型缓存为空，尝试加载")
            from src.services.processor.model_loader import load_models
            load_models()

        max_concurrency = max(1, min(max_concurrency, 10))
        results = await process_batch_images(
            files, model_name, cache_bypass=cache_bypass, use_coreml=use_coreml,
            use_model=use_model, use_attributes=use_attributes,
            use_deepdanbooru=use_deepdanbooru, max_concurrency=max_concurrency,
        )

        return {
            "success": True,
            "data": results,
            "message": f"成功处理 {len(results)} 个图像",
            "stats": {
                "total_files": len(files),
                "success_count": sum(1 for r in results if r.get("success", False)),
                "max_concurrency": max_concurrency,
            },
        }
    except Exception as e:
        logger.error(f"批量分类图像失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        from src.core.error.error_handler import raise_app_error
        from src.core.error.error_codes import ErrorCode
        raise_app_error(error_code=ErrorCode.CLASSIFICATION_FAILED, status_code=500, detail=str(e))


@router.post("/api/classify/multi-model")
async def multi_model_classify_image(
    file: UploadFile = File(...),
    multi_role: bool = Form(False),
    current_user: dict = Depends(get_current_user),
):
    """使用多个模型进行分类"""
    try:
        from src.services.model.multi_model_service import get_multi_model_service
        content = await file.read()
        result = await get_multi_model_service().process_with_multiple_models(file, content, multi_role)
        return {"success": True, "data": result, "message": "多模型分类成功"}
    except Exception as e:
        logger.error(f"多模型分类失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        from src.core.error.error_handler import raise_app_error
        from src.core.error.error_codes import ErrorCode
        raise_app_error(error_code=ErrorCode.CLASSIFICATION_FAILED, status_code=500, detail=str(e))
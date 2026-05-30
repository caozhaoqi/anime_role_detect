#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分类任务
定义异步图像分类Celery任务
"""

import os
import io
from typing import Dict, Any
from fastapi import UploadFile
from src.tasks.celery_app import celery_app
from src.core.logging.global_logger import get_logger

logger = get_logger("celery.tasks")


class ClassificationError(Exception):
    """分类任务错误"""

    pass


@celery_app.task(bind=True, max_retries=3, name="src.tasks.classify_tasks.classify_image_task")
def classify_image_task(
    self,
    image_content: bytes,
    image_filename: str,
    model_name: str = "resnet18_loli8",
    use_coreml: bool = False,
    use_model: bool = True,
    use_attributes: bool = True,
    use_deepdanbooru: bool = True,
    multi_role: bool = False,
) -> Dict[str, Any]:
    """
    异步图像分类任务

    Args:
        self: Celery task实例
        image_content: 图像二进制内容
        image_filename: 图像文件名
        model_name: 模型名称
        use_coreml: 是否使用CoreML
        use_model: 是否使用模型
        use_attributes: 是否使用属性
        use_deepdanbooru: 是否使用DeepDanbooru标签提取
        multi_role: 是否多角色检测

    Returns:
        Dict containing classification result
    """
    try:
        logger.info(f"开始异步分类任务: {image_filename}, model={model_name}")

        from src.services.processor.image_processor import (
            process_single_image,
            process_multi_role_image,
        )
        from fastapi import UploadFile
        from starlette.datastructures import UploadFile as StarletteUploadFile

        class BytesUploadFile(UploadFile):
            def __init__(self, content: bytes, filename: str):
                super().__init__(filename)
                self._content = content

            async def read(self, size: int = -1) -> bytes:
                if size == -1:
                    return self._content
                return self._content[:size]

            async def seek(self, offset: int, whence: int = 0) -> int:
                return 0

            async def write(self, content: bytes) -> None:
                pass

        file = BytesUploadFile(image_content, image_filename)

        if multi_role:
            result = process_multi_role_image(
                file=file,
                model_name=model_name,
                cache_bypass=False,
                use_coreml=use_coreml,
                use_model=use_model,
                use_attributes=use_attributes,
                use_deepdanbooru=use_deepdanbooru,
            )
        else:
            result = process_single_image(
                file=file,
                model_name=model_name,
                cache_bypass=False,
                use_coreml=use_coreml,
                use_model=use_model,
                use_attributes=use_attributes,
                use_deepdanbooru=use_deepdanbooru,
            )

        logger.info(f"异步分类任务完成: {image_filename}")
        return {"status": "success", "result": result, "filename": image_filename}

    except Exception as e:
        logger.error(f"异步分类任务失败: {e}")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(bind=True, max_retries=3, name="src.tasks.classify_tasks.batch_classify_task")
def batch_classify_task(
    self,
    image_contents: list,
    image_filenames: list,
    model_name: str = "resnet18_loli8",
    use_coreml: bool = False,
    use_model: bool = False,
    use_attributes: bool = False,
    use_deepdanbooru: bool = True,
    max_concurrency: int = 4,
) -> Dict[str, Any]:
    """
    批量异步图像分类任务

    Args:
        self: Celery task实例
        image_contents: 图像二进制内容列表
        image_filenames: 图像文件名列表
        model_name: 模型名称
        use_coreml: 是否使用CoreML
        use_model: 是否使用模型
        use_attributes: 是否使用属性
        use_deepdanbooru: 是否使用DeepDanbooru标签提取
        max_concurrency: 最大并发数

    Returns:
        Dict containing batch classification results
    """
    try:
        logger.info(f"开始批量异步分类任务: {len(image_filenames)}张图像")

        results = []
        for i, (content, filename) in enumerate(zip(image_contents, image_filenames)):
            sub_task = classify_image_task.delay(
                image_content=content,
                image_filename=filename,
                model_name=model_name,
                use_coreml=use_coreml,
                use_model=use_model,
                use_attributes=use_attributes,
                use_deepdanbooru=use_deepdanbooru,
            )
            results.append(
                {"index": i, "filename": filename, "task_id": sub_task.id, "status": "pending"}
            )

        logger.info(f"批量任务已提交: {len(image_filenames)}张图像")
        return {
            "status": "submitted",
            "batch_id": self.request.id,
            "total": len(image_filenames),
            "tasks": results,
        }

    except Exception as e:
        logger.error(f"批量分类任务失败: {e}")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(name="src.tasks.classify_tasks.get_task_status")
def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    获取任务状态

    Args:
        task_id: Celery任务ID

    Returns:
        Dict containing task status
    """
    from src.tasks.celery_app import celery_app

    task = celery_app.AsyncResult(task_id)

    response = {"task_id": task_id, "status": task.state}

    if task.state == "SUCCESS":
        response["result"] = task.result
    elif task.state == "FAILURE":
        response["error"] = str(task.info)
    elif task.state == "PENDING":
        pass
    else:
        response["info"] = task.info

    return response

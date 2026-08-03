#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像处理器

负责处理图像的主逻辑
"""

import os
import hashlib
import numpy as np
from src.core.logging import get_enhanced_logger as get_logger
from src.services.cache_service import get_cache_manager
from .model_processor import (
    process_with_model_service,
    process_with_local_model,
    process_with_trained_model,
    process_with_traditional_model,
)
from src.core.classification.deepdanbooru_inference import DeepDanbooruInference
from .utils import with_temp_file, validate_file, get_file_extension

logger = get_logger("image_processor")


class_names = ["unknown", "plana", "other"]


def _generate_cache_key(content, model_name, options=None):
    """生成缓存键（委托给缓存助手）。"""
    from src.services.cache_service.cache_helper import get_classify_cache_key

    return get_classify_cache_key(content, model_name, options)


def _extract_deepdanbooru_tags(content, filename, result):
    """使用 DeepDanbooru 提取标签，原地更新 result。"""

    def process(temp_path):
        try:
            deepdanbooru = DeepDanbooruInference()
            tags = deepdanbooru.predict(temp_path, top_k=30, threshold=0.3)
            if tags:
                result["deepdanbooru_tags"] = tags
                if "tags" not in result:
                    tag_list = [tag["tag"] for tag in tags[:30]]
                    if len(tag_list) < 10:
                        more_tags = deepdanbooru.predict(temp_path, top_k=50, threshold=0.2)
                        more_tag_list = [
                            tag["tag"] for tag in more_tags[:50] if tag["tag"] not in tag_list
                        ]
                        tag_list.extend(more_tag_list)
                    result["tags"] = tag_list[:30]
                else:
                    existing_tags = set(result["tags"])
                    new_tags = [tag["tag"] for tag in tags if tag["tag"] not in existing_tags]
                    result["tags"].extend(new_tags)
                    if len(result["tags"]) < 10:
                        more_tags = deepdanbooru.predict(temp_path, top_k=50, threshold=0.2)
                        more_tag_list = [
                            tag["tag"] for tag in more_tags[:50] if tag["tag"] not in result["tags"]
                        ]
                        result["tags"].extend(more_tag_list)
                    result["tags"] = result["tags"][:30]
                logger.info(f"DeepDanbooru提取到 {len(tags)} 个标签")
        except Exception as e:
            logger.error(f"DeepDanbooru标签提取失败: {e}")
        return result

    suffix = get_file_extension(filename)
    return with_temp_file(content, suffix, process)


def _build_image_cache_key(content, model_name, use_coreml, use_model,
                           use_attributes, use_deepdanbooru, multi_role=False):
    """生成单/多角色图像处理的缓存键（含影响结果的所有选项）。"""
    cache_options = {
        "use_coreml": use_coreml,
        "use_model": use_model,
        "use_attributes": use_attributes,
        "use_deepdanbooru": use_deepdanbooru,
    }
    if multi_role:
        cache_options["multi_role"] = True
    key_model = f"multi_role_{model_name}" if multi_role else model_name
    return _generate_cache_key(content, key_model, cache_options)


def _detect_multi_role(temp_path, model_name):
    """多角色检测：初始化检测器并构建标准结果结构（bbox 转原生类型）。"""
    from src.core.detection.multi_role_detection import MultiRoleDetector

    detector = MultiRoleDetector(model_name=model_name)
    detections = detector.detect_roles(temp_path)

    processed_results = []
    for det in detections:
        bbox = det.get("bbox", {})
        processed_bbox = {}
        for key, value in bbox.items():
            if isinstance(value, (np.integer, np.floating)):
                processed_bbox[key] = (
                    float(value) if isinstance(value, np.floating) else int(value)
                )
            else:
                processed_bbox[key] = value
        processed_results.append(
            {
                "role": det.get("role", "unknown"),
                "similarity": float(det.get("similarity", 0.0)),
                "tags": det.get("attributes", []),
                "bbox": processed_bbox,
                "confidence": float(det.get("confidence", 0.0)),
            }
        )

    return {
        "roles": processed_results,
        "count": len(processed_results),
        "nsfw": {"is_nsfw": False, "details": {}},
    }


def _run_multi_role_local(content, filename, model_name):
    """本地多角色检测（临时文件封装 + 日志）。"""
    suffix = get_file_extension(filename)
    logger.info(f"执行本地多角色检测，文件后缀: {suffix}")
    local_result = with_temp_file(content, suffix, lambda tp: _detect_multi_role(tp, model_name))
    logger.info(f"本地多角色检测完成，结果: {local_result}")
    return local_result


async def _run_multi_role_model_service(file, content, model_name):
    """经模型服务处理多角色图像；失败或返回错误时回退本地检测。"""
    use_model_service = os.environ.get("USE_MODEL_SERVICE", "True").lower() == "true"
    logger.info(f"USE_MODEL_SERVICE: {use_model_service}")

    if use_model_service:
        try:
            logger.info("使用模型服务处理多角色图像")
            model_result = await process_with_model_service(
                file, content, model_name, multi_role=True
            )
            if "error" in model_result:
                logger.error(f"模型服务返回错误: {model_result['error']}")
                use_model_service = False
            else:
                logger.info("模型服务处理成功")
                return model_result
        except Exception as e:
            logger.error(f"模型服务处理失败，回退到本地检测: {e}")
            use_model_service = False

    logger.info(f"检查use_model_service状态: {use_model_service}")
    if not use_model_service:
        logger.info("使用本地多角色检测")
        return _run_multi_role_local(content, file.filename, model_name)


def _run_single_image_model_path(file, content, temp_path, use_coreml, model_name):
    """
    单图模型路径：CoreML 优先，命中 unknown/0.0 时回退本地模型；否则走传统模型。
    注：保持与原实现一致——本地模型回退调用不 await（原闭包上下文为同步）。
    """
    if use_coreml:
        result = process_with_trained_model(file, temp_path, model_name)
        if result.get("role") == "unknown" and result.get("similarity") == 0.0:
            logger.info("CoreML模型不存在，回退到本地模型")
            return process_with_local_model(file, content, model_name)
        return result
    return process_with_traditional_model(file, temp_path, model_name)


async def _process_one_in_batch(file, model_name, cache_bypass, use_coreml, use_model,
                                use_attributes, use_deepdanbooru):
    """批量处理中的单文件封装：调用单图处理并统一返回结构。"""
    try:
        result = await process_single_image(
            file,
            model_name,
            cache_bypass=cache_bypass,
            use_coreml=use_coreml,
            use_model=use_model,
            use_attributes=use_attributes,
            use_deepdanbooru=use_deepdanbooru,
        )
        return {"filename": file.filename, "result": result, "success": True}
    except Exception as e:
        logger.error(f"处理文件 {file.filename} 失败: {e}")
        return {"filename": file.filename, "result": {"error": str(e)}, "success": False}


async def _select_single_image_backend(file, content, model_name):
    """单图后端选择：USE_MODEL_SERVICE 为真走模型服务，否则本地模型。"""
    use_model_service = os.environ.get("USE_MODEL_SERVICE", "True").lower() == "true"
    logger.info(f"USE_MODEL_SERVICE: {use_model_service}")
    if use_model_service:
        logger.info("使用模型服务处理图像")
        return await process_with_model_service(file, content, model_name)
    logger.info("使用本地模型处理图像")
    return await process_with_local_model(file, content, model_name)


async def process_single_image(
    file,
    model_name,
    cache_bypass=False,
    use_coreml=False,
    use_model=False,
    use_attributes=False,
    use_deepdanbooru=True,
):
    """处理单个图像：缓存优先，按模型后端选择处理路径。"""
    try:
        content = await file.read()
        if not validate_file(content):
            return {"error": "文件为空"}

        cache_key = _build_image_cache_key(
            content, model_name, use_coreml, use_model, use_attributes, use_deepdanbooru
        )
        cache_manager = get_cache_manager()
        if not cache_bypass:
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"从缓存获取结果: {cache_key}")
                return cached_result

        logger.info(
            f"process_single_image 参数: use_model={use_model}, use_coreml={use_coreml}, "
            f"model_name={model_name}, use_deepdanbooru={use_deepdanbooru}"
        )

        if use_model and not use_coreml:
            result = await _select_single_image_backend(file, content, model_name)
        else:
            suffix = get_file_extension(file.filename)
            result = with_temp_file(
                content, suffix,
                lambda tp: _run_single_image_model_path(file, content, tp, use_coreml, model_name),
            )

        if use_deepdanbooru:
            logger.info("使用DeepDanbooru提取标签")
            result = _extract_deepdanbooru_tags(content, file.filename, result)

        cache_manager.set(cache_key, result, ttl=3600)
        return result

    except Exception as e:
        logger.error(f"处理图像失败: {e}")
        return {"error": str(e)}


async def process_batch_images(
    files,
    model_name,
    cache_bypass=False,
    use_coreml=False,
    use_model=False,
    use_attributes=False,
    use_deepdanbooru=True,
    max_concurrency=4,
):
    """批量处理图像：并发调用单图处理并统计耗时与成功率。"""
    import asyncio
    import time

    start_time = time.time()

    semaphore = asyncio.Semaphore(max_concurrency)
    async def process_file(file):
        async with semaphore:
            return await _process_one_in_batch(
                file, model_name, cache_bypass, use_coreml, use_model,
                use_attributes, use_deepdanbooru,
            )

    tasks = [process_file(file) for file in files]
    results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time
    avg_time_per_file = total_time / len(files) if files else 0
    logger.info(
        f"批量处理完成: {len(files)}个文件，总耗时: {total_time:.2f}秒，平均每个文件: {avg_time_per_file:.2f}秒"
    )

    success_count = sum(1 for r in results if r.get("success", False))
    success_rate = (success_count / len(files)) * 100 if files else 0
    logger.info(f"批量处理成功率: {success_rate:.2f}% ({success_count}/{len(files)})")

    return results


async def process_multi_role_image(
    file,
    model_name,
    cache_bypass=False,
    use_coreml=False,
    use_model=False,
    use_attributes=False,
    use_deepdanbooru=True,
):
    """处理多角色图像：缓存优先，模型服务优先、失败回退本地检测。"""
    try:
        content = await file.read()
        if not validate_file(content):
            return {"error": "文件为空"}

        cache_key = _build_image_cache_key(
            content, model_name, use_coreml, use_model, use_attributes,
            use_deepdanbooru, multi_role=True,
        )
        cache_manager = get_cache_manager()
        if not cache_bypass:
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"从缓存获取多角色检测结果: {cache_key}")
                return cached_result

        logger.info(
            f"process_multi_role_image 参数: use_model={use_model}, use_coreml={use_coreml}, "
            f"model_name={model_name}, use_deepdanbooru={use_deepdanbooru}"
        )

        if use_model and not use_coreml:
            result = await _run_multi_role_model_service(file, content, model_name)
        else:
            result = _run_multi_role_local(content, file.filename, model_name)

        if use_deepdanbooru:
            logger.info("使用DeepDanbooru提取标签")
            result = _extract_deepdanbooru_tags(content, file.filename, result)

        cache_manager.set(cache_key, result, ttl=3600)
        return result

    except Exception as e:
        logger.error(f"处理多角色图像失败: {e}")
        return {"error": str(e)}

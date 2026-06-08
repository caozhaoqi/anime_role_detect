#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型处理器

负责各种模型处理操作
"""

import os
import time
import asyncio
import aiohttp
import numpy as np
from src.core.logging.global_logger import get_logger
from src.services.processor.model_loader import (
    get_preprocessor,
    get_keypoint_detector,
    get_tagger,
    get_role_predictor,
)
from src.services.nsfw_detector import detect_nsfw
from src.services.circuit_breaker_service import execute_with_fallback_async
from .preprocessor import preprocess_image
from .model_loader import load_trained_model
from .feature_processor import process_image_features

logger = get_logger("image_processor")

# 使用统一配置
from src.core.config.service_config import get_service_config

config = get_service_config()

USE_MODEL_SERVICE = config.USE_MODEL_SERVICE
MODEL_SERVICE_URL = config.MODEL_SERVICE_URL

os.makedirs("temp", exist_ok=True)


async def _call_model_service(file, content, model_name, multi_role=False):
    """
    调用模型服务

    Args:
        file: 上传的文件
        content: 文件内容
        model_name: 模型名称
        multi_role: 是否使用多角色检测

    Returns:
        dict: 处理结果
    """
    temp_path = None

    try:
        logger.info(f"使用模型服务: {MODEL_SERVICE_URL}, 多角色: {multi_role}")

        # 确定文件类型
        content_type = file.content_type
        if content_type is None:
            ext = os.path.splitext(file.filename)[1].lower()
            ext_to_content_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".bmp": "image/bmp",
                ".svg": "image/svg+xml",
            }
            content_type = ext_to_content_type.get(ext, "application/octet-stream")

        # 发送请求到模型服务
        if multi_role:
            endpoint = f"{MODEL_SERVICE_URL}/api/model/detect-multiple"
            logger.info(f"开始调用模型服务(多角色): {endpoint}")
        else:
            endpoint = f"{MODEL_SERVICE_URL}/api/model/predict"
            logger.info(f"开始调用模型服务: {endpoint}")

        logger.info(f"请求文件: {file.filename}, 大小: {len(content)}字节, 类型: {content_type}")

        # 使用aiohttp进行异步HTTP调用
        async with aiohttp.ClientSession() as session:
            # 构建multipart/form-data请求
            form = aiohttp.FormData()
            form.add_field("file", content, filename=file.filename, content_type=content_type)
            form.add_field("model_name", model_name)
            form.add_field("use_attributes", "true")

            logger.info(f"准备发送请求到模型服务")
            async with session.post(endpoint, data=form, timeout=30) as response:
                logger.info(f"模型服务响应状态码: {response.status}")
                response.raise_for_status()
                model_result = await response.json()
                logger.info(f"模型服务返回数据: {model_result}")

        if multi_role:
            # 处理多角色结果
            roles = model_result.get("roles", [])
            count = model_result.get("count", 0)
            nsfw = model_result.get("nsfw", {"is_nsfw": False, "details": {}})

            # 处理角色信息，确保numpy类型转换为Python原生类型
            processed_roles = []
            for role in roles:
                # 确保bbox中的值都是Python原生类型
                bbox = role.get("bbox", {})
                processed_bbox = {}
                for key, value in bbox.items():
                    if isinstance(value, (np.integer, np.floating)):
                        processed_bbox[key] = (
                            float(value) if isinstance(value, np.floating) else int(value)
                        )
                    else:
                        processed_bbox[key] = value

                processed_roles.append(
                    {
                        "role": role.get("role", "unknown"),
                        "similarity": float(role.get("similarity", 0.0)),
                        "tags": role.get("tags", []),
                        "bbox": processed_bbox,
                        "confidence": float(role.get("confidence", 0.0)),
                    }
                )

            # 保存临时文件用于其他处理
            temp_path = f"temp/temp_{int(time.time())}_{file.filename}"
            with open(temp_path, "wb") as f:
                f.write(content)

            # 处理图像特征
            text_detections, keypoints, ai_predicted_role = process_image_features(
                temp_path, file.content_type, []
            )

            # 执行NSFW检测
            nsfw_result = detect_nsfw(temp_path)

            # 构建结果
            result = {
                "roles": processed_roles,
                "count": count,
                "text_detections": text_detections,
                "keypoints": keypoints,
                "ai_predicted_role": _get_chinese_role_name(ai_predicted_role),
                "nsfw": nsfw_result,
            }
        else:
            # 处理单角色结果
            # 处理结果
            role = model_result.get("role", "unknown")
            similarity = model_result.get("similarity", 0.0)
            attributes = model_result.get("attributes", [])
            tags = model_result.get("tags", [])
            feature = model_result.get("feature", None)

            logger.info(
                f"模型服务返回结果: role={role}, similarity={similarity}, has_feature={feature is not None}"
            )

            # 保存临时文件用于其他处理
            temp_path = f"temp/temp_{int(time.time())}_{file.filename}"
            with open(temp_path, "wb") as f:
                f.write(content)

            # 如果模型服务返回unknown且提供了特征向量，使用本地模型进行分类
            if role == "unknown" and feature is not None:
                logger.info(
                    f"模型服务返回unknown且提供了特征向量，role={role}, feature长度={len(feature) if feature else 'None'}"
                )
                # 加载训练好的模型

                model_info = load_trained_model(model_name)
                logger.info(f"load_trained_model返回: {model_info}")
                if model_info is not None:
                    model, class_to_idx = model_info
                    idx_to_class = {v: k for k, v in class_to_idx.items()}

                    # 预处理图像
                    img = preprocess_image(temp_path)

                    # 预测
                    import torch

                    with torch.no_grad():
                        outputs = model(img)
                        _, predicted = torch.max(outputs, 1)
                        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][
                            predicted.item()
                        ].item()

                    # 获取预测结果
                    role = idx_to_class.get(predicted.item(), "unknown")
                    similarity = float(confidence)

                    # 转换为中文角色名
                    chinese_role = _get_chinese_role_name(role)

                    logger.info(
                        f"本地模型分类结果: {role} -> {chinese_role}, 相似度: {similarity:.4f}"
                    )

            # 处理图像特征
            text_detections, keypoints, ai_predicted_role = process_image_features(
                temp_path, file.content_type, attributes
            )

            # 执行NSFW检测
            nsfw_result = detect_nsfw(temp_path)

            # 使用AI预测的角色名作为主要角色（如果模型服务返回unknown）
            final_role = role if role != "unknown" else ai_predicted_role or "unknown"

            # 构建结果
            result = {
                "role": final_role,
                "similarity": similarity,
                "possible_roles": [],
                "attributes": attributes,
                "tags": tags,
                "text_detections": text_detections,
                "keypoints": keypoints,
                "ai_predicted_role": ai_predicted_role or final_role,
                "nsfw": nsfw_result,
            }

        return result

    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"清理临时文件失败: {e}")


async def _model_service_fallback(file, content, model_name, multi_role=False):
    """
    模型服务降级策略

    Args:
        file: 上传的文件
        content: 文件内容
        model_name: 模型名称
        multi_role: 是否使用多角色检测

    Returns:
        dict: 降级处理结果
    """
    logger.warning(f"模型服务不可用，使用本地模型处理: {model_name}")
    return await process_with_local_model(file, content, model_name)


async def process_with_model_service(file, content, model_name, multi_role=False):
    """
    使用模型服务处理图像

    Args:
        file: 上传的文件
        content: 文件内容
        model_name: 模型名称
        multi_role: 是否使用多角色检测

    Returns:
        dict: 处理结果
    """
    try:
        # 使用熔断器调用模型服务
        result = await execute_with_fallback_async(
            name=f"model_service_{'multi' if multi_role else 'single'}",
            func=_call_model_service,
            fallback=_model_service_fallback,
            file=file,
            content=content,
            model_name=model_name,
            multi_role=multi_role,
        )
        return result
    except Exception as e:
        logger.error(f"使用模型服务处理图像失败: {e}")
        # 最终降级到本地模型
        return await process_with_local_model(file, content, model_name)


async def process_with_local_model(file, content, model_name):
    """
    使用本地模型处理图像

    Args:
        file: 上传的文件
        content: 文件内容
        model_name: 模型名称

    Returns:
        dict: 处理结果
    """
    temp_path = None

    try:
        logger.info(f"使用本地模型: {model_name}")

        # 保存临时文件
        temp_path = f"temp/temp_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)

        # 加载训练好的模型
        model_info = load_trained_model(model_name)

        if model_info is None:
            logger.warning(f"未找到训练好的模型: {model_name}")
            # 如果模型不存在，使用传统模型处理
            text_detections, keypoints, ai_predicted_role = process_image_features(
                temp_path, file.content_type, []
            )
            nsfw_result = detect_nsfw(temp_path)
            logger.info(f"传统模型分类结果: {ai_predicted_role or 'unknown'}")
            # 使用AI预测的角色名作为主要角色
            final_role = ai_predicted_role or "unknown"
            return {
                "role": final_role,
                "similarity": 0.0,
                "possible_roles": [],
                "attributes": [],
                "text_detections": text_detections,
                "keypoints": keypoints,
                "ai_predicted_role": final_role,
                "nsfw": nsfw_result,
            }
        else:
            model, class_to_idx = model_info
            idx_to_class = {v: k for k, v in class_to_idx.items()}

            # 预处理图像
            img = preprocess_image(temp_path)

            # 预测
            import torch

            with torch.no_grad():
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()

            # 获取预测结果
            role = idx_to_class.get(predicted.item(), "unknown")
            similarity = float(confidence)

            # 转换为中文角色名
            chinese_role = _get_chinese_role_name(role)

        logger.info(f"本地模型分类结果: {role} -> {chinese_role}, 相似度: {similarity:.4f}")

        # 处理图像特征
        text_detections, keypoints, ai_predicted_role = process_image_features(
            temp_path, file.content_type, []
        )

        # 执行NSFW检测
        nsfw_result = detect_nsfw(temp_path)

        # 构建结果
        result = {
            "role": chinese_role,
            "similarity": similarity,
            "possible_roles": [],
            "attributes": [],
            "tags": ["digital art", "anime", "character"],  # 默认标签
            "text_detections": text_detections,
            "keypoints": keypoints,
            "ai_predicted_role": _get_chinese_role_name(ai_predicted_role),
            "nsfw": nsfw_result,
        }

        return result

    except Exception as e:
        logger.error(f"使用本地模型处理图像失败: {e}")
        raise
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"清理临时文件失败: {e}")


def _get_chinese_role_name(role_name):
    """
    将英文/拼音角色名转换为中文

    Args:
        role_name: 英文/拼音角色名

    Returns:
        str: 中文角色名
    """
    role_mapping = {
        "a1luo2na4": "阿罗娜",
        "ri4nai4": "日奈",
        "unknown": "未知",
        "plana": "普拉娜",
        "other": "其他",
    }
    return role_mapping.get(role_name, role_name)


def process_with_trained_model(file, image_source, model_name):
    """
    使用训练好的模型处理图像

    Args:
        file: 上传的文件
        image_source: 图像来源
        model_name: 模型名称

    Returns:
        dict: 处理结果
    """
    try:
        logger.info(f"使用训练好的模型: {model_name}")

        # 加载训练好的模型
        model_info = load_trained_model(model_name)

        if model_info is None:
            logger.warning(f"未找到训练好的模型: {model_name}")
            # 如果模型不存在，使用传统模型处理
            text_detections, keypoints, ai_predicted_role = process_image_features(
                image_source, file.content_type, []
            )
            nsfw_result = detect_nsfw(image_source)
            logger.info(f"传统模型分类结果: {ai_predicted_role or 'unknown'}")
            # 使用AI预测的角色名作为主要角色
            final_role = ai_predicted_role or "unknown"
            return {
                "role": final_role,
                "similarity": 0.0,
                "possible_roles": [],
                "attributes": [],
                "text_detections": text_detections,
                "keypoints": keypoints,
                "ai_predicted_role": final_role,
                "nsfw": nsfw_result,
            }

        model, class_to_idx = model_info
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        # 预处理图像
        img = preprocess_image(image_source)

        # 预测
        import torch

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()

        # 获取预测结果
        role = idx_to_class.get(predicted.item(), "unknown")
        similarity = float(confidence)

        # 转换为中文角色名
        chinese_role = _get_chinese_role_name(role)

        logger.info(f"训练模型分类结果: {role} -> {chinese_role}, 相似度: {similarity:.4f}")

        # 处理图像特征
        text_detections, keypoints, ai_predicted_role = process_image_features(
            image_source, file.content_type, []
        )

        # 执行NSFW检测
        nsfw_result = detect_nsfw(image_source)

        # 构建结果
        result = {
            "role": chinese_role,
            "similarity": similarity,
            "possible_roles": [],
            "attributes": [],
            "tags": ["digital art", "anime", "character"],  # 默认标签
            "text_detections": text_detections,
            "keypoints": keypoints,
            "ai_predicted_role": _get_chinese_role_name(ai_predicted_role),
            "nsfw": nsfw_result,
        }

        return result

    except Exception as e:
        logger.error(f"使用训练模型处理图像失败: {e}")
        raise


def process_with_traditional_model(file, image_source, model_name):
    """
    使用传统模型处理图像

    Args:
        file: 上传的文件
        image_source: 图像来源
        model_name: 模型名称

    Returns:
        dict: 处理结果
    """
    try:
        logger.info(f"使用传统模型: {model_name}")

        # 处理图像特征
        text_detections, keypoints, ai_predicted_role = process_image_features(
            image_source, file.content_type, []
        )

        # 执行NSFW检测
        nsfw_result = detect_nsfw(image_source)

        # 构建结果
        result = {
            "role": "unknown",
            "similarity": 0.0,
            "possible_roles": [],
            "attributes": [],
            "text_detections": text_detections,
            "keypoints": keypoints,
            "ai_predicted_role": ai_predicted_role,
            "nsfw": nsfw_result,
        }

        return result

    except Exception as e:
        logger.error(f"使用传统模型处理图像失败: {e}")
        raise

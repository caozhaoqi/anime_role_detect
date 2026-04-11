#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像处理服务
负责处理图像相关的操作
"""

import os
import time
import hashlib
import asyncio
import aiohttp
from PIL import Image
from src.core.logging.global_logger import get_logger
from src.utils.image_utils import ImageUtils
from src.backend.services.model_loader import get_preprocessor, get_keypoint_detector, get_tagger, get_role_predictor
from src.backend.services.nsfw_detector import detect_nsfw
from src.backend.services.cache_service import get_cache_manager, get_image_transform, model_cache, model_versions

logger = get_logger("image_processor")

# 从环境变量中读取配置
# 禁用模型服务，因为启动失败
USE_MODEL_SERVICE = False
MODEL_SERVICE_URL = os.environ.get('MODEL_SERVICE_URL', 'http://localhost:8001')

# 类名列表
class_names = [
    "unknown", "plana", "other"
]


def preprocess_image(image_path):
    """
    预处理图像
    
    Args:
        image_path: 图像路径
    
    Returns:
        预处理后的图像张量
    """
    # 延迟导入PyTorch
    import torch
    
    try:
        # 获取图像变换
        transform = get_image_transform()
        
        # 加载图像并转换
        img = Image.open(image_path).convert('RGB')
        
        # 限制图像大小，避免内存占用过高
        max_size = 1024
        width, height = img.size
        if width > max_size or height > max_size:
            # 计算缩放比例
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"图像已缩放: {width}x{height} -> {new_width}x{new_height}")
        
        img = transform(img)
        img = img.unsqueeze(0)  # 添加批次维度
        
        return img
    except Exception as e:
        logger.error(f"预处理图像失败: {e}")
        raise


def _generate_cache_key(content, model_name):
    """
    生成缓存键
    
    Args:
        content: 文件内容
        model_name: 模型名称
    
    Returns:
        str: 缓存键
    """
    file_hash = hashlib.md5(content).hexdigest()
    return f"image_processing_{file_hash}_{model_name}"


async def _process_with_model_service(file, content, model_name):
    """
    使用模型服务处理图像
    
    Args:
        file: 上传的文件
        content: 文件内容
        model_name: 模型名称
    
    Returns:
        dict: 处理结果或None
    """
    import os
    
    temp_path = None
    
    try:
        logger.info(f"使用模型服务: {MODEL_SERVICE_URL}")
        
        # 确定文件类型
        content_type = file.content_type
        if content_type is None:
            ext = os.path.splitext(file.filename)[1].lower()
            ext_to_content_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.svg': 'image/svg+xml'
            }
            content_type = ext_to_content_type.get(ext, 'application/octet-stream')
        
        # 发送请求到模型服务
        logger.info(f"开始调用模型服务: {MODEL_SERVICE_URL}/api/model/predict")
        logger.info(f"请求文件: {file.filename}, 大小: {len(content)}字节, 类型: {content_type}")
        
        # 使用aiohttp进行异步HTTP调用
        async with aiohttp.ClientSession() as session:
            # 构建multipart/form-data请求
            form = aiohttp.FormData()
            form.add_field('file', content, filename=file.filename, content_type=content_type)
            form.add_field('model_name', model_name)
            form.add_field('use_attributes', 'true')
            
            logger.info(f"准备发送请求到模型服务")
            async with session.post(
                f"{MODEL_SERVICE_URL}/api/model/predict",
                data=form,
                timeout=30
            ) as response:
                logger.info(f"模型服务响应状态码: {response.status}")
                response.raise_for_status()
                model_result = await response.json()
                logger.info(f"模型服务返回数据: {model_result}")
        
        # 处理结果
        role = model_result.get('role', 'unknown')
        similarity = model_result.get('similarity', 0.0)
        attributes = model_result.get('attributes', [])
        feature = model_result.get('feature', None)
        
        logger.info(f"模型服务返回结果: role={role}, similarity={similarity}, has_feature={feature is not None}")
        
        # 保存临时文件用于其他处理
        temp_path = f"temp/temp_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # 如果模型服务返回unknown且提供了特征向量，使用本地模型进行分类
        if role == 'unknown' and feature is not None:
            logger.info(f"模型服务返回unknown且提供了特征向量，role={role}, feature长度={len(feature) if feature else 'None'}")
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
                    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
                
                # 获取预测结果
                role = idx_to_class.get(predicted.item(), "unknown")
                similarity = float(confidence)
                logger.info(f"本地模型分类结果: {role}, 相似度: {similarity:.4f}")
        
        # 处理图像特征
        text_detections, keypoints, ai_predicted_role = _process_image_features(temp_path, file.content_type, attributes)
        
        # 执行NSFW检测
        nsfw_result = detect_nsfw(temp_path)
        
        # 构建结果
        result = {
            "role": role,
            "similarity": similarity,
            "possible_roles": possible_roles if 'possible_roles' in locals() else [],
            "attributes": attributes,
            "text_detections": text_detections,
            "keypoints": keypoints,
            "ai_predicted_role": ai_predicted_role,
            "nsfw": nsfw_result
        }
        
        return result
    except Exception as e:
        logger.error(f"调用模型服务失败: {e}")
        logger.error(f"异常类型: {type(e).__name__}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        # 回退到本地处理
        return None
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"清理临时文件: {temp_path}")
            except Exception as e:
                logger.error(f"清理临时文件失败: {e}")


async def _process_with_local_model(file, content, model_name):
    """
    使用本地模型处理图像
    
    Args:
        file: 上传的文件
        content: 文件内容
        model_name: 模型名称
    
    Returns:
        dict: 处理结果
    """
    import os
    
    temp_path = None
    
    try:
        # 验证图像
        validate_start = time.time()
        is_valid = ImageUtils.validate_image(content)
        validate_time = time.time() - validate_start
        logger.debug(f"验证图像耗时: {validate_time:.4f}秒, 结果: {is_valid}")
        
        # 保存临时文件
        temp_path = f"temp/temp_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # 检查是否使用新训练的模型
        trained_model_names = ["mobilenet_v2", "efficientnet_b0", "efficientnet_b3", "resnet50", "incremental"]
        
        if model_name in trained_model_names:
            return await _process_with_trained_model(file, temp_path, model_name)
        else:
            return _process_with_traditional_model(file, temp_path, model_name)
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"清理临时文件: {temp_path}")
            except Exception as e:
                logger.error(f"清理临时文件失败: {e}")


async def _process_with_trained_model(file, temp_path, model_name):
    """
    使用训练好的模型处理图像
    
    Args:
        file: 上传的文件
        temp_path: 临时文件路径
        model_name: 模型名称
    
    Returns:
        dict: 处理结果
    """
    logger.info(f"使用新训练的模型: {model_name}")
    
    # 加载训练好的模型
    model_info = load_trained_model(model_name)
    if model_info is None:
        result = {"role": "unknown", "similarity": 0.0, "attributes": []}
        return result
    
    model, class_to_idx = model_info
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # 预处理图像
    try:
        logger.info(f"加载图像: {temp_path}")
        img = preprocess_image(temp_path)
        logger.info(f"图像预处理成功，形状: {img.shape}")
        
        # 预测
        import torch
        with torch.no_grad():
            logger.info("开始模型预测...")
            outputs = model(img)
            logger.info(f"模型输出形状: {outputs.shape}")
            
            # 计算所有类别的概率
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # 获取最高概率的类别
            _, predicted = torch.max(outputs, 1)
            confidence = probabilities[predicted.item()].item()
        logger.info(f"预测完成，预测类别: {predicted.item()}, 置信度: {confidence}")
        
        # 获取预测结果
        role = idx_to_class.get(predicted.item(), "unknown")
        similarity = float(confidence)
        logger.info(f"预测角色: {role}, 相似度: {similarity}")
        
        # 获取高概率的可能结果
        possible_roles = []
        sorted_indices = torch.argsort(probabilities, descending=True)
        for idx in sorted_indices[:3]:  # 取前3个高概率结果
            class_idx = idx.item()
            class_name = idx_to_class.get(class_idx, "unknown")
            prob = probabilities[idx].item()
            if prob > 0.1:  # 只保留概率大于0.1的结果
                possible_roles.append({
                    "role": class_name,
                    "probability": float(prob)
                })
        logger.info(f"高概率可能结果: {possible_roles}")
        
        # 标签生成
        attributes = []
        try:
            logger.info("开始标签生成...")
            tagger = get_tagger()
            if tagger is not None:
                attributes = tagger.generate_tags(temp_path)
                logger.info(f"标签生成完成，生成 {len(attributes)} 个标签")
            else:
                logger.info("标签生成模块未初始化，跳过标签生成")
        except Exception as e:
            logger.error(f"标签生成失败: {e}")
            attributes = []
        
        # 处理图像特征
        text_detections, keypoints, ai_predicted_role = _process_image_features(temp_path, file.content_type, attributes)
        
        # 执行NSFW检测
        nsfw_result = detect_nsfw(temp_path)
        
        # 构建结果
        result = {
            "role": role,
            "similarity": similarity,
            "possible_roles": possible_roles,
            "attributes": attributes,
            "text_detections": text_detections,
            "keypoints": keypoints,
            "ai_predicted_role": ai_predicted_role,
            "nsfw": nsfw_result,
            "detection_process": {
                "content_detection": {
                    "nsfw": nsfw_result,
                    "ocr": {
                        "status": "成功" if text_detections is not None else "失败",
                        "text_detections": text_detections
                    }
                },
                "classification": {
                    "status": "成功" if role != "unknown" or similarity > 0 else "失败",
                    "role": role,
                    "similarity": similarity,
                    "possible_roles": possible_roles
                },
                "tag_output": {
                    "status": "成功" if len(attributes) > 0 else "未初始化，跳过",
                    "attributes": attributes
                },
                "keypoint_detection": {
                    "status": "成功" if len(keypoints) > 0 else "未初始化，跳过",
                    "keypoints": keypoints
                },
                "role_prediction": {
                    "status": "成功" if ai_predicted_role is not None else "失败",
                    "predicted_role": ai_predicted_role
                }
            }
        }
        
        return result
    except Exception as e:
        logger.error(f"处理图像失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        result = {"role": "unknown", "similarity": 0.0, "attributes": []}
        return result


def _process_with_traditional_model(file, temp_path, model_name):
    """
    使用传统模型处理图像
    
    Args:
        file: 上传的文件
        temp_path: 临时文件路径
        model_name: 模型名称
    
    Returns:
        dict: 处理结果
    """
    import torch
    import torch.nn as nn
    import torchvision.models as models
    
    model = None
    img = None
    outputs = None
    probabilities = None
    predicted = None
    
    try:
        logger.info("创建默认预训练模型: mobilenet_v2")
        
        # 创建一个mobilenet_v2模型
        model = models.mobilenet_v2(pretrained=True)
        # 修改分类层，使其输出3个类别
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        model.eval()
        logger.info("默认预训练模型创建成功")
        
        # 预处理图像
        img = preprocess_image(temp_path)
        
        # 预测
        with torch.no_grad():
            outputs = model(img)
            
            # 计算所有类别的概率
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # 获取最高概率的类别
            _, predicted = torch.max(outputs, 1)
            confidence = probabilities[predicted.item()].item()
        
        # 获取高概率的可能结果
        possible_roles = []
        sorted_indices = torch.argsort(probabilities, descending=True)
        for idx in sorted_indices[:3]:  # 取前3个高概率结果
            class_idx = idx.item()
            class_name = class_names[class_idx]
            prob = probabilities[idx].item()
            if prob > 0.1:  # 只保留概率大于0.1的结果
                possible_roles.append({
                    "role": class_name,
                    "probability": float(prob)
                })
        logger.info(f"高概率可能结果: {possible_roles}")
        
        # 获取预测结果
        role = class_names[predicted.item()]
        similarity = float(confidence)
        
        # 标签生成
        attributes = []
        try:
            logger.info("开始标签生成...")
            tagger = get_tagger()
            if tagger is not None:
                attributes = tagger.generate_tags(temp_path)
                logger.info(f"标签生成完成，生成 {len(attributes)} 个标签")
            else:
                logger.info("标签生成模块未初始化，跳过标签生成")
        except Exception as e:
            logger.error(f"标签生成失败: {e}")
            attributes = []
        
        # 处理图像特征
        text_detections, keypoints, ai_predicted_role = _process_image_features(temp_path, file.content_type, attributes)
        
        # 执行NSFW检测
        nsfw_result = detect_nsfw(temp_path)
        
        # 构建结果
        result = {
            "role": role,
            "similarity": similarity,
            "possible_roles": possible_roles,
            "attributes": attributes,
            "text_detections": text_detections,
            "keypoints": keypoints,
            "ai_predicted_role": ai_predicted_role,
            "nsfw": nsfw_result,
            "detection_process": {
                "content_detection": {
                    "nsfw": nsfw_result,
                    "ocr": {
                        "status": "成功" if text_detections is not None else "失败",
                        "text_detections": text_detections
                    }
                },
                "classification": {
                    "status": "成功" if role != "unknown" or similarity > 0 else "失败",
                    "role": role,
                    "similarity": similarity,
                    "possible_roles": possible_roles
                },
                "tag_output": {
                    "status": "成功" if len(attributes) > 0 else "未初始化，跳过",
                    "attributes": attributes
                },
                "keypoint_detection": {
                    "status": "成功" if len(keypoints) > 0 else "未初始化，跳过",
                    "keypoints": keypoints
                },
                "role_prediction": {
                    "status": "成功" if ai_predicted_role is not None else "失败",
                    "predicted_role": ai_predicted_role
                }
            }
        }
        
        return result
    except Exception as e:
        logger.error(f"创建模型失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        result = {"role": "unknown", "similarity": 0.0, "attributes": []}
        return result
    finally:
        # 释放内存
        if model is not None:
            del model
        if img is not None:
            del img
        if outputs is not None:
            del outputs
        if probabilities is not None:
            del probabilities
        if predicted is not None:
            del predicted
        import gc
        gc.collect()
        logger.info("已释放模型和相关内存")


def _process_image_features(temp_path, content_type, attributes):
    """
    处理图像特征
    
    Args:
        temp_path: 临时文件路径
        content_type: 文件类型
        attributes: 标签属性
    
    Returns:
        tuple: (text_detections, keypoints, ai_predicted_role)
    """
    # 文本检测
    text_detections = []
    try:
        if content_type != "image/svg+xml":
            logger.info("开始文本检测...")
            preprocessor = get_preprocessor()
            if preprocessor is not None:
                text_detections = preprocessor.detect_text(temp_path)
                logger.info(f"文本检测完成，检测到 {len(text_detections)} 个文本")
            else:
                logger.info("预处理模块未初始化，跳过文本检测")
        else:
            text_detections = []
            logger.info("SVG图像，跳过文本检测")
    except Exception as e:
        logger.error(f"文本检测失败: {e}")
        text_detections = []
    
    # 关键点检测
    keypoints = []
    try:
        if content_type != "image/svg+xml":
            logger.info("开始关键点检测...")
            keypoint_detector = get_keypoint_detector()
            if keypoint_detector is not None:
                keypoints = keypoint_detector.detect_keypoints(temp_path)
                logger.info(f"关键点检测完成，检测到 {len(keypoints)} 个关键点")
            else:
                logger.info("关键点检测模块未初始化，跳过关键点检测")
        else:
            keypoints = []
            logger.info("SVG图像，跳过关键点检测")
    except Exception as e:
        logger.error(f"关键点检测失败: {e}")
        keypoints = []
    
    # 角色预测
    ai_predicted_role = None
    try:
        logger.info("开始角色预测...")
        role_predictor = get_role_predictor()
        if role_predictor is not None:
            ai_predicted_role = role_predictor.predict_role(attributes)
            logger.info(f"角色预测完成，预测角色: {ai_predicted_role}")
        else:
            logger.info("角色预测模块未初始化，跳过角色预测")
    except Exception as e:
        logger.error(f"角色预测失败: {e}")
        ai_predicted_role = None
    
    return text_detections, keypoints, ai_predicted_role


def load_trained_model(model_name):
    """
    加载训练好的模型，支持热更新
    
    Args:
        model_name: 模型名称
    
    Returns:
        模型和类别映射
    """
    global model_cache, model_versions
    
    # 构建模型路径
    model_dir = os.path.join("models", model_name)
    model_path = os.path.join(model_dir, "model_best.pth")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        # 检查模型目录是否存在
        if not os.path.exists(model_dir):
            logger.warning(f"模型目录不存在: {model_dir}")
        # 尝试使用 incremental 模型作为后备
        logger.info(f"尝试使用 incremental 模型作为后备")
        return load_trained_model("incremental")
    
    # 获取模型文件的修改时间作为版本标识
    model_mtime = os.path.getmtime(model_path)
    
    # 检查模型是否在缓存中且版本未更新
    if model_name in model_cache and model_name in model_versions:
        if model_versions[model_name] == model_mtime:
            logger.info(f"从缓存加载模型: {model_name} (版本未更新)")
            return model_cache[model_name]
        else:
            logger.info(f"模型文件已更新，重新加载: {model_name}")
            # 释放旧模型内存
            if model_name in model_cache:
                del model_cache[model_name]
                logger.info(f"已释放旧模型内存: {model_name}")
    
    try:
        logger.info(f"加载训练好的模型: {model_name}")
        
        # 确定模型类型
        if model_name == "incremental":
            model_type = "mobilenet_v2_incremental"
        else:
            model_type = model_name
        
        # 加载模型权重
        import torch
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 获取类别映射
        class_to_idx = checkpoint.get('class_to_idx', {})
        num_classes = len(class_to_idx)
        
        # 导入get_model函数
        from src.core.classification.models import get_model
        
        # 创建模型
        model = get_model(model_type, num_classes)
        
        # 加载权重，使用strict=False以忽略不匹配的键
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        logger.info(f"模型加载成功: {model_name}, 类别数: {num_classes}")
        
        # 更新模型缓存和版本信息
        model_cache[model_name] = (model, class_to_idx)
        model_versions[model_name] = model_mtime
        logger.info(f"模型已更新到缓存: {model_name}, 版本: {model_mtime}")
        
        return model, class_to_idx
    except Exception as e:
        logger.error(f"加载训练好的模型失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        return None


async def process_single_image(file, model_name, cache_bypass=False):
    """
    处理单个图像
    
    Args:
        file: 上传的文件
        model_name: 模型名称
        cache_bypass: 是否绕过缓存
    
    Returns:
        dict: 处理结果
    """
    start_time = time.time()
    content = await file.read()
    
    # 生成缓存键
    cache_key = _generate_cache_key(content, model_name)
    
    # 尝试从缓存获取结果
    if not cache_bypass:
        cache_manager = get_cache_manager()
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            logger.info(f"从缓存获取结果，缓存键: {cache_key}")
            cached_result["processing_time"] = time.time() - start_time
            return cached_result
    
    # 调用模型服务或本地模型处理图像
    if USE_MODEL_SERVICE:
        result = await _process_with_model_service(file, content, model_name)
    else:
        result = None
    
    # 如果模型服务失败或未启用，使用本地模型
    if result is None:
        logger.info("使用本地模型处理图像")
        result = await _process_with_local_model(file, content, model_name)
    
    # 计算处理时间
    processing_time = time.time() - start_time
    result["processing_time"] = processing_time
    logger.info(f"处理图像耗时: {processing_time:.4f}秒")
    
    # 缓存结果
    if not cache_bypass:
        cache_manager = get_cache_manager()
        cache_manager.set(value=result, key=cache_key)
        logger.info(f"结果已缓存，缓存键: {cache_key}")
    
    return result


async def process_batch_images(files, model_name, cache_bypass=False):
    """
    处理批量图像
    
    Args:
        files: 上传的文件列表
        model_name: 模型名称
        cache_bypass: 是否绕过缓存
    
    Returns:
        list: 处理结果列表
    """
    tasks = []
    for file in files:
        task = process_single_image(file, model_name, cache_bypass)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

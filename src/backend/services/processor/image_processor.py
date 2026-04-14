#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像处理器

负责处理图像的主逻辑
"""

import os
import hashlib
from src.core.logging.global_logger import get_logger
from src.backend.services.cache_service import get_cache_manager
from .model_processor import process_with_model_service, process_with_local_model, process_with_trained_model, process_with_traditional_model

logger = get_logger("image_processor")


class_names = [
    "unknown", "plana", "other"
]


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


async def process_single_image(file, model_name, cache_bypass=False, use_coreml=False, use_model=False, use_attributes=False):
    """
    处理单个图像
    
    Args:
        file: 上传的文件
        model_name: 模型名称
        cache_bypass: 是否绕过缓存
        use_coreml: 是否使用CoreML
        use_model: 是否使用模型
        use_attributes: 是否使用属性
    
    Returns:
        dict: 处理结果
    """
    try:
        # 读取文件内容
        content = await file.read()
        
        # 检查文件大小
        if len(content) == 0:
            return {"error": "文件为空"}
        
        # 生成缓存键
        cache_key = _generate_cache_key(content, model_name)
        
        # 尝试从缓存获取结果
        cache_manager = get_cache_manager()
        if not cache_bypass:
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"从缓存获取结果: {cache_key}")
                return cached_result
        
        # 处理图像
        if use_model and not use_coreml:
            if os.environ.get('USE_MODEL_SERVICE', 'False').lower() == 'true':
                result = await process_with_model_service(file, content, model_name)
            else:
                result = await process_with_local_model(file, content, model_name)
        else:
            # 保存临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            try:
                if use_coreml:
                    # CoreML处理逻辑
                    result = process_with_trained_model(file, temp_path, model_name)
                    # 如果CoreML模型不存在，回退到本地模型
                    if result.get('role') == 'unknown' and result.get('similarity') == 0.0:
                        logger.info("CoreML模型不存在，回退到本地模型")
                        result = await process_with_local_model(file, content, model_name)
                else:
                    # 传统处理逻辑
                    result = process_with_traditional_model(file, temp_path, model_name)
            finally:
                # 清理临时文件
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        logger.error(f"清理临时文件失败: {e}")
        
        # 缓存结果
        cache_manager.set(result, cache_key, ttl=3600)
        
        return result
        
    except Exception as e:
        logger.error(f"处理图像失败: {e}")
        return {"error": str(e)}


async def process_batch_images(files, model_name, cache_bypass=False, use_coreml=False, use_model=False, use_attributes=False):
    """
    批量处理图像
    
    Args:
        files: 上传的文件列表
        model_name: 模型名称
        cache_bypass: 是否绕过缓存
        use_coreml: 是否使用CoreML
        use_model: 是否使用模型
        use_attributes: 是否使用属性
    
    Returns:
        list: 处理结果列表
    """
    results = []
    
    for file in files:
        result = await process_single_image(
            file, 
            model_name, 
            cache_bypass=cache_bypass, 
            use_coreml=use_coreml, 
            use_model=use_model, 
            use_attributes=use_attributes
        )
        results.append({
            "filename": file.filename,
            "result": result
        })
    
    return results

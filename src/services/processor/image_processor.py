#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像处理器

负责处理图像的主逻辑
"""

import os
import hashlib
from src.core.logging.global_logger import get_logger
from src.services.cache_service import get_cache_manager
from .model_processor import process_with_model_service, process_with_local_model, process_with_trained_model, process_with_traditional_model
from src.core.classification.deepdanbooru_inference import DeepDanbooruInference

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


async def process_single_image(file, model_name, cache_bypass=False, use_coreml=False, use_model=False, use_attributes=False, use_deepdanbooru=True):
    """
    处理单个图像
    
    Args:
        file: 上传的文件
        model_name: 模型名称
        cache_bypass: 是否绕过缓存
        use_coreml: 是否使用CoreML
        use_model: 是否使用模型
        use_attributes: 是否使用属性
        use_deepdanbooru: 是否使用DeepDanbooru标签提取
    
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
        logger.info(f"process_single_image 参数: use_model={use_model}, use_coreml={use_coreml}, model_name={model_name}, use_deepdanbooru={use_deepdanbooru}")
        if use_model and not use_coreml:
            use_model_service = os.environ.get('USE_MODEL_SERVICE', 'True').lower() == 'true'
            logger.info(f"USE_MODEL_SERVICE: {use_model_service}")
            if use_model_service:
                logger.info("使用模型服务处理图像")
                result = await process_with_model_service(file, content, model_name)
            else:
                logger.info("使用本地模型处理图像")
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
        
        # 使用DeepDanbooru提取标签
        if use_deepdanbooru:
            logger.info("使用DeepDanbooru提取标签")
            # 保存临时文件用于DeepDanbooru处理
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            try:
                # 初始化DeepDanbooru推理器
                deepdanbooru = DeepDanbooruInference()
                # 提取标签
                tags = deepdanbooru.predict(temp_path, top_k=20, threshold=0.5)
                # 添加到结果中
                if tags:
                    result['deepdanbooru_tags'] = tags
                    # 如果结果中没有tags字段，也添加到tags字段中
                    if 'tags' not in result:
                        result['tags'] = [tag['tag'] for tag in tags]
                    else:
                        # 合并标签，去重
                        existing_tags = set(result['tags'])
                        new_tags = [tag['tag'] for tag in tags if tag['tag'] not in existing_tags]
                        result['tags'].extend(new_tags)
                logger.info(f"DeepDanbooru提取到 {len(tags)} 个标签")
            except Exception as e:
                logger.error(f"DeepDanbooru标签提取失败: {e}")
            finally:
                # 清理临时文件
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        logger.error(f"清理临时文件失败: {e}")
        
        # 缓存结果
        cache_manager.set(cache_key, result, ttl=3600)
        
        return result
        
    except Exception as e:
        logger.error(f"处理图像失败: {e}")
        return {"error": str(e)}


async def process_batch_images(files, model_name, cache_bypass=False, use_coreml=False, use_model=False, use_attributes=False, use_deepdanbooru=True, max_concurrency=4):
    """
    批量处理图像
    
    Args:
        files: 上传的文件列表
        model_name: 模型名称
        cache_bypass: 是否绕过缓存
        use_coreml: 是否使用CoreML
        use_model: 是否使用模型
        use_attributes: 是否使用属性
        use_deepdanbooru: 是否使用DeepDanbooru标签提取
        max_concurrency: 最大并发数
    
    Returns:
        list: 处理结果列表
    """
    import asyncio
    import time
    
    start_time = time.time()
    results = []
    
    # 限制并发数
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_file(file):
        async with semaphore:
            try:
                result = await process_single_image(
                    file, 
                    model_name, 
                    cache_bypass=cache_bypass, 
                    use_coreml=use_coreml, 
                    use_model=use_model, 
                    use_attributes=use_attributes,
                    use_deepdanbooru=use_deepdanbooru
                )
                return {
                    "filename": file.filename,
                    "result": result,
                    "success": True
                }
            except Exception as e:
                logger.error(f"处理文件 {file.filename} 失败: {e}")
                return {
                    "filename": file.filename,
                    "result": {"error": str(e)},
                    "success": False
                }
    
    # 并行处理所有文件
    tasks = [process_file(file) for file in files]
    results = await asyncio.gather(*tasks)
    
    # 计算处理时间
    total_time = time.time() - start_time
    avg_time_per_file = total_time / len(files) if files else 0
    
    logger.info(f"批量处理完成: {len(files)}个文件，总耗时: {total_time:.2f}秒，平均每个文件: {avg_time_per_file:.2f}秒")
    
    # 统计成功率
    success_count = sum(1 for r in results if r.get('success', False))
    success_rate = (success_count / len(files)) * 100 if files else 0
    
    logger.info(f"批量处理成功率: {success_rate:.2f}% ({success_count}/{len(files)})")
    
    return results


async def process_multi_role_image(file, model_name, cache_bypass=False, use_coreml=False, use_model=False, use_attributes=False, use_deepdanbooru=True):
    """
    处理多角色图像
    
    Args:
        file: 上传的文件
        model_name: 模型名称
        cache_bypass: 是否绕过缓存
        use_coreml: 是否使用CoreML
        use_model: 是否使用模型
        use_attributes: 是否使用属性
        use_deepdanbooru: 是否使用DeepDanbooru标签提取
    
    Returns:
        dict: 多角色处理结果
    """
    try:
        # 读取文件内容
        content = await file.read()
        
        # 检查文件大小
        if len(content) == 0:
            return {"error": "文件为空"}
        
        # 生成缓存键
        cache_key = _generate_cache_key(content, f"multi_role_{model_name}")
        
        # 尝试从缓存获取结果
        cache_manager = get_cache_manager()
        if not cache_bypass:
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"从缓存获取多角色检测结果: {cache_key}")
                return cached_result
        
        # 处理图像
        logger.info(f"process_multi_role_image 参数: use_model={use_model}, use_coreml={use_coreml}, model_name={model_name}, use_deepdanbooru={use_deepdanbooru}")
        if use_model and not use_coreml:
            use_model_service = os.environ.get('USE_MODEL_SERVICE', 'True').lower() == 'true'
            logger.info(f"USE_MODEL_SERVICE: {use_model_service}")
            if use_model_service:
                logger.info("使用模型服务处理多角色图像")
                result = await process_with_model_service(file, content, model_name, multi_role=True)
            else:
                logger.info("使用本地多角色检测")
                # 保存临时文件
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False) as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name
                
                try:
                    # 导入多角色检测器
                    from src.core.detection.multi_role_detection import MultiRoleDetector
                    
                    # 初始化检测器
                    detector = MultiRoleDetector(model_name=model_name)
                    
                    # 检测角色
                    results = detector.detect_roles(temp_path)
                    
                    # 处理结果
                    processed_results = []
                    for result in results:
                        processed_results.append({
                            "role": result.get("role", "unknown"),
                            "similarity": float(result.get("similarity", 0.0)),
                            "tags": result.get("attributes", []),
                            "bbox": result.get("bbox", {}),
                            "confidence": float(result.get("confidence", 0.0))
                        })
                    
                    # 构建结果
                    result = {
                        "roles": processed_results,
                        "count": len(processed_results),
                        "nsfw": {"is_nsfw": False, "details": {}}
                    }
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception as e:
                            logger.error(f"清理临时文件失败: {e}")
        else:
            # 保存临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            try:
                # 导入多角色检测器
                from src.core.detection.multi_role_detection import MultiRoleDetector
                
                # 初始化检测器
                detector = MultiRoleDetector(model_name=model_name)
                
                # 检测角色
                results = detector.detect_roles(temp_path)
                
                # 处理结果
                processed_results = []
                for result in results:
                    processed_results.append({
                        "role": result.get("role", "unknown"),
                        "similarity": float(result.get("similarity", 0.0)),
                        "tags": result.get("attributes", []),
                        "bbox": result.get("bbox", {}),
                        "confidence": float(result.get("confidence", 0.0))
                    })
                
                # 构建结果
                result = {
                    "roles": processed_results,
                    "count": len(processed_results),
                    "nsfw": {"is_nsfw": False, "details": {}}
                }
            finally:
                # 清理临时文件
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        logger.error(f"清理临时文件失败: {e}")
        
        # 使用DeepDanbooru提取标签
        if use_deepdanbooru:
            logger.info("使用DeepDanbooru提取标签")
            # 保存临时文件用于DeepDanbooru处理
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            try:
                # 初始化DeepDanbooru推理器
                deepdanbooru = DeepDanbooruInference()
                # 提取标签
                tags = deepdanbooru.predict(temp_path, top_k=20, threshold=0.5)
                # 添加到结果中
                if tags:
                    result['deepdanbooru_tags'] = tags
                    # 添加到tags字段中
                    if 'tags' not in result:
                        result['tags'] = [tag['tag'] for tag in tags]
                    else:
                        # 合并标签，去重
                        existing_tags = set(result['tags'])
                        new_tags = [tag['tag'] for tag in tags if tag['tag'] not in existing_tags]
                        result['tags'].extend(new_tags)
                logger.info(f"DeepDanbooru提取到 {len(tags)} 个标签")
            except Exception as e:
                logger.error(f"DeepDanbooru标签提取失败: {e}")
            finally:
                # 清理临时文件
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        logger.error(f"清理临时文件失败: {e}")
        
        # 缓存结果
        cache_manager.set(cache_key, result, ttl=3600)
        
        return result
        
    except Exception as e:
        logger.error(f"处理多角色图像失败: {e}")
        return {"error": str(e)}

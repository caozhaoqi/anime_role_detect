#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的图像加载器

使用libjpeg-turbo加速图像解码，支持图像缓存和WebP格式
"""

import os
import hashlib
import threading
from io import BytesIO
from PIL import Image
from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("image_loader")

# 图像缓存（内存缓存）
image_cache = {}
cache_lock = threading.Lock()

# 缓存配置
CACHE_MAX_SIZE = 100  # 最大缓存图像数量
CACHE_TTL = 3600  # 缓存过期时间（秒）


def compute_image_hash(file_path):
    """
    计算图像文件的哈希值作为缓存键
    
    Args:
        file_path: 图像文件路径
        
    Returns:
        哈希字符串
    """
    file_stat = os.stat(file_path)
    # 使用文件路径、大小和修改时间生成哈希
    hash_content = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
    return hashlib.md5(hash_content.encode()).hexdigest()


def load_image_with_cache(file_path, use_cache=True):
    """
    加载图像并使用缓存
    
    Args:
        file_path: 图像文件路径
        use_cache: 是否使用缓存
        
    Returns:
        PIL图像对象
    """
    if use_cache:
        cache_key = compute_image_hash(file_path)
        
        with cache_lock:
            if cache_key in image_cache:
                cached_data = image_cache[cache_key]
                # 检查缓存是否过期
                import time
                if time.time() - cached_data['timestamp'] < CACHE_TTL:
                    logger.debug(f"命中图像缓存: {file_path}")
                    return cached_data['image'].copy()
        
    # 加载图像
    img = load_image(file_path)
    
    if use_cache:
        with cache_lock:
            # 检查缓存大小并清理旧缓存
            if len(image_cache) >= CACHE_MAX_SIZE:
                # 删除最旧的缓存
                oldest_key = min(image_cache.keys(), key=lambda k: image_cache[k]['timestamp'])
                del image_cache[oldest_key]
            
            import time
            image_cache[cache_key] = {
                'image': img.copy(),
                'timestamp': time.time()
            }
            logger.debug(f"图像已缓存: {file_path}")
    
    return img


def load_image(file_path):
    """
    使用优化的方式加载图像
    
    优先使用libjpeg-turbo加速JPEG解码，如果不可用则回退到标准PIL
    
    Args:
        file_path: 图像文件路径
        
    Returns:
        PIL图像对象（RGB格式）
    """
    # 确保PIL Image已导入
    from PIL import Image
    
    try:
        # 尝试使用libjpeg-turbo加速
        try:
            from turbojpeg import TurboJPEG
            turbo_jpeg = TurboJPEG()
            
            # 只对JPEG文件使用TurboJPEG
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ('.jpg', '.jpeg'):
                with open(file_path, 'rb') as f:
                    jpeg_data = f.read()
                
                # 使用TurboJPEG解码
                img = turbo_jpeg.decode(jpeg_data)
                # 转换为PIL图像
                return Image.fromarray(img)
        except ImportError:
            logger.debug("turbojpeg不可用，使用标准PIL加载")
        except Exception as e:
            logger.debug(f"TurboJPEG解码失败: {e}，回退到标准PIL")
        
        # 标准PIL加载（支持所有格式）
        return Image.open(file_path).convert("RGB")
    
    except Exception as e:
        logger.error(f"加载图像失败: {file_path}, 错误: {e}")
        raise


def convert_to_webp(image, quality=80):
    """
    将图像转换为WebP格式
    
    Args:
        image: PIL图像对象
        quality: WebP质量（0-100）
        
    Returns:
        WebP格式的字节数据
    """
    buffer = BytesIO()
    image.save(buffer, format='WEBP', quality=quality)
    buffer.seek(0)
    return buffer.read()


def save_as_webp(image, output_path, quality=80):
    """
    将图像保存为WebP格式
    
    Args:
        image: PIL图像对象
        output_path: 输出文件路径
        quality: WebP质量（0-100）
    """
    # 确保输出路径有正确的扩展名
    if not output_path.lower().endswith('.webp'):
        output_path += '.webp'
    
    image.save(output_path, format='WEBP', quality=quality)
    logger.info(f"图像已保存为WebP格式: {output_path}")


def load_webp_or_convert(file_path, convert_on_load=False):
    """
    加载图像，如果是JPEG格式则可选转换为WebP
    
    Args:
        file_path: 图像文件路径
        convert_on_load: 是否在加载时转换为WebP并保存
        
    Returns:
        PIL图像对象
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # 如果是WebP格式，直接加载
    if ext == '.webp':
        return Image.open(file_path).convert("RGB")
    
    # 加载原始图像
    img = load_image(file_path)
    
    # 如果需要转换
    if convert_on_load and ext in ('.jpg', '.jpeg'):
        webp_path = os.path.splitext(file_path)[0] + '.webp'
        if not os.path.exists(webp_path):
            save_as_webp(img, webp_path)
            logger.info(f"已将JPEG转换为WebP: {webp_path}")
    
    return img


def get_cache_info():
    """
    获取图像缓存信息
    
    Returns:
        缓存统计信息字典
    """
    import time
    with cache_lock:
        info = {
            'cache_size': len(image_cache),
            'max_size': CACHE_MAX_SIZE,
            'ttl_seconds': CACHE_TTL,
            'entries': []
        }
        
        for key, data in image_cache.items():
            info['entries'].append({
                'key': key[:8] + '...',  # 只显示部分哈希
                'age_seconds': int(time.time() - data['timestamp']),
                'image_size': f"{data['image'].size[0]}x{data['image'].size[1]}"
            })
        
        return info


def clear_image_cache():
    """
    清空图像缓存
    """
    with cache_lock:
        image_cache.clear()
    logger.info("图像缓存已清空")
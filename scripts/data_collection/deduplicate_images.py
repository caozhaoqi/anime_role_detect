#!/usr/bin/env python3
"""
图片去重脚本
- 为每张图片生成哈希值
- 比较哈希值，识别重复图片
- 移除重复图片，保留一张
"""

import os
import hashlib
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入统一日志配置
from common.logging_config import get_logger

# 配置日志
logger = get_logger('data_collection.deduplicate_images', 'deduplicate_images.log')

# 全局配置
GLOBAL_CONFIG = {
    'image_dir': '../../data/role_images',
    'hash_algorithm': 'md5',  # 使用md5生成哈希值
    'max_workers': 4,  # 并行处理的线程数
    'remove_duplicates': True,  # 是否移除重复图片
    'min_image_size': 1024  # 最小图片大小（字节）
}

def calculate_image_hash(image_path):
    """计算图片的哈希值"""
    try:
        # 检查文件大小
        if os.path.getsize(image_path) < GLOBAL_CONFIG['min_image_size']:
            return None
        
        # 打开图片
        with Image.open(image_path) as img:
            # 调整图片大小，统一尺寸进行哈希计算
            img = img.resize((100, 100))
            # 转换为灰度图
            img = img.convert('L')
            # 计算哈希值
            hash_obj = hashlib.new(GLOBAL_CONFIG['hash_algorithm'])
            hash_obj.update(img.tobytes())
            return hash_obj.hexdigest()
    except Exception as e:
        logger.warning(f"计算图片哈希值失败: {image_path} - {str(e)}")
        return None

def process_role_images(role_name, role_dir):
    """处理单个角色的图片去重"""
    logger.info(f"开始处理角色 {role_name} 的图片去重")
    
    # 获取角色目录下的所有图片
    image_files = []
    for file_name in os.listdir(role_dir):
        file_path = os.path.join(role_dir, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            image_files.append(file_path)
    
    total_images = len(image_files)
    if total_images == 0:
        logger.info(f"角色 {role_name} 没有图片")
        return role_name, 0, 0
    
    logger.info(f"角色 {role_name} 共有 {total_images} 张图片")
    
    # 计算每张图片的哈希值
    image_hashes = {}
    duplicate_count = 0
    
    with ThreadPoolExecutor(max_workers=GLOBAL_CONFIG['max_workers']) as executor:
        futures = {executor.submit(calculate_image_hash, img_path): img_path for img_path in image_files}
        
        for future in as_completed(futures):
            img_path = futures[future]
            try:
                img_hash = future.result()
                if img_hash:
                    if img_hash in image_hashes:
                        # 发现重复图片
                        duplicate_count += 1
                        logger.info(f"发现重复图片: {os.path.basename(img_path)} 与 {os.path.basename(image_hashes[img_hash])} 重复")
                        
                        # 移除重复图片
                        if GLOBAL_CONFIG['remove_duplicates']:
                            try:
                                os.remove(img_path)
                                logger.info(f"已移除重复图片: {os.path.basename(img_path)}")
                            except Exception as e:
                                logger.error(f"移除重复图片失败: {img_path} - {str(e)}")
                    else:
                        # 新图片，记录哈希值
                        image_hashes[img_hash] = img_path
            except Exception as e:
                logger.error(f"处理图片失败: {img_path} - {str(e)}")
    
    unique_count = len(image_hashes)
    logger.info(f"角色 {role_name} 去重完成: 原始 {total_images} 张，去重后 {unique_count} 张，移除 {duplicate_count} 张重复图片")
    
    return role_name, total_images, duplicate_count

def main():
    """主函数"""
    logger.info("============================================================")
    logger.info("开始图片去重")
    logger.info("============================================================")
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, GLOBAL_CONFIG['image_dir'])
    
    # 检查目录是否存在
    if not os.path.exists(image_dir):
        logger.error(f"图片目录不存在: {image_dir}")
        return
    
    # 获取所有角色目录
    role_dirs = []
    for item in os.listdir(image_dir):
        item_path = os.path.join(image_dir, item)
        if os.path.isdir(item_path):
            role_dirs.append((item, item_path))
    
    logger.info(f"发现 {len(role_dirs)} 个角色目录")
    
    # 处理每个角色的图片
    total_images = 0
    total_duplicates = 0
    
    for role_name, role_dir in role_dirs:
        _, role_images, role_duplicates = process_role_images(role_name, role_dir)
        total_images += role_images
        total_duplicates += role_duplicates
    
    logger.info("\n============================================================")
    logger.info("图片去重完成")
    logger.info(f"总处理图片数: {total_images}")
    logger.info(f"总移除重复图片数: {total_duplicates}")
    logger.info(f"重复率: {total_duplicates / total_images * 100:.2f}%" if total_images > 0 else "无图片")
    logger.info("============================================================")

if __name__ == "__main__":
    main()

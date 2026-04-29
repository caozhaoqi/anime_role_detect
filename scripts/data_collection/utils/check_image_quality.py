#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查已下载图片的质量和分辨率
"""

import os
from PIL import Image
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('check_image_quality.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局配置
GLOBAL_CONFIG = {
    'download_dir': '../../data/role_images',
    'min_resolution': [800, 800]
}

def check_image_quality(image_path, min_resolution):
    """检查单张图片的质量和分辨率"""
    try:
        # 打开图片
        img = Image.open(image_path)
        
        # 检查分辨率
        width, height = img.size
        if width < min_resolution[0] or height < min_resolution[1]:
            return False, f"分辨率不足 ({width}x{height})"
        
        # 检查图片是否损坏
        img.verify()
        
        return True, f"分辨率符合要求 ({width}x{height})"
    except Exception as e:
        return False, str(e)

def process_role(role_name):
    """处理单个角色的图片"""
    role_dir = os.path.join(GLOBAL_CONFIG['download_dir'], role_name)
    if not os.path.exists(role_dir):
        logger.warning(f"角色 {role_name} 的目录不存在: {role_dir}")
        return role_name, 0, 0, 0
    
    # 统计图片数量
    image_files = []
    for file in os.listdir(role_dir):
        if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            image_files.append(os.path.join(role_dir, file))
    
    total_images = len(image_files)
    if total_images == 0:
        logger.info(f"角色 {role_name} 没有图片")
        return role_name, 0, 0, 0
    
    # 检查图片质量
    valid_count = 0
    invalid_count = 0
    
    for image_path in image_files:
        is_valid, message = check_image_quality(image_path, GLOBAL_CONFIG['min_resolution'])
        if is_valid:
            valid_count += 1
            logger.debug(f"{role_name}/{os.path.basename(image_path)}: {message}")
        else:
            invalid_count += 1
            logger.warning(f"{role_name}/{os.path.basename(image_path)}: {message}")
    
    logger.info(f"角色 {role_name}: 共 {total_images} 张图片，有效 {valid_count} 张，无效 {invalid_count} 张")
    return role_name, total_images, valid_count, invalid_count

def main():
    """主函数"""
    print("=" * 60)
    print("检查图片质量和分辨率")
    print("=" * 60)
    
    # 确保下载目录存在
    if not os.path.exists(GLOBAL_CONFIG['download_dir']):
        logger.error(f"下载目录不存在: {GLOBAL_CONFIG['download_dir']}")
        return
    
    # 遍历所有角色目录
    roles = []
    for dir_name in os.listdir(GLOBAL_CONFIG['download_dir']):
        role_dir = os.path.join(GLOBAL_CONFIG['download_dir'], dir_name)
        if os.path.isdir(role_dir):
            roles.append(dir_name)
    
    if not roles:
        logger.error("没有找到角色目录")
        return
    
    logger.info(f"找到 {len(roles)} 个角色目录")
    
    # 处理每个角色
    total_stats = {
        'total_roles': len(roles),
        'total_images': 0,
        'valid_images': 0,
        'invalid_images': 0
    }
    
    for role_name in roles:
        role_name, total, valid, invalid = process_role(role_name)
        total_stats['total_images'] += total
        total_stats['valid_images'] += valid
        total_stats['invalid_images'] += invalid
    
    # 输出总结果
    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)
    print(f"总角色数: {total_stats['total_roles']}")
    print(f"总图片数: {total_stats['total_images']}")
    print(f"有效图片: {total_stats['valid_images']}")
    print(f"无效图片: {total_stats['invalid_images']}")
    print(f"有效率: {total_stats['valid_images'] / total_stats['total_images'] * 100:.2f}%" if total_stats['total_images'] > 0 else "有效率: 0%")
    print("=" * 60)

if __name__ == "__main__":
    main()

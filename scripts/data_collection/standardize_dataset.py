#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准化数据集：
1. 只保留英文角色名目录
2. 将所有图片转换为 jpg 或 png 格式
3. 根据 loli-role.txt 合并数据
"""

import os
import sys
import shutil
import logging
from PIL import Image

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("standardize_dataset")

def parse_role_list(role_list_path):
    """
    解析角色列表，提取英文角色名
    :param role_list_path: 角色列表文件路径
    :return: 英文角色名集合
    """
    english_names = set()
    
    with open(role_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 每行格式：中文 游戏 英文 日文
            parts = line.split()
            if len(parts) >= 3:
                english_name = parts[2]
                english_names.add(english_name)
    
    logger.info(f"从角色列表中解析出 {len(english_names)} 个英文角色名")
    return english_names

def convert_to_jpg(image_path, output_path):
    """
    将图片转换为 JPG 格式
    :param image_path: 原始图片路径
    :param output_path: 输出路径
    """
    try:
        with Image.open(image_path) as img:
            # 处理透明度
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode == 'P':
                img = img.convert('RGB')
            
            # 保存为 JPG
            img.save(output_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        logger.warning(f"转换图片失败 {image_path}: {e}")
        return False

def standardize_dataset(dataset_path, role_list_path, dry_run=False):
    """
    标准化数据集
    """
    # 解析角色列表
    valid_english_names = parse_role_list(role_list_path)
    
    logger.info("=" * 60)
    logger.info(f"开始标准化数据集")
    logger.info(f"数据集路径: {dataset_path}")
    logger.info(f"模式: {'预览' if dry_run else '实际执行'}")
    logger.info("=" * 60)
    
    # 获取当前目录列表
    all_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    # 统计变量
    removed_dirs = 0
    converted_images = 0
    removed_images = 0
    
    for dir_name in all_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        
        # 检查是否为有效英文角色名
        if dir_name not in valid_english_names:
            logger.info(f"删除无效目录: {dir_name}")
            if not dry_run:
                shutil.rmtree(dir_path)
            removed_dirs += 1
            continue
        
        # 处理有效目录中的图片
        files = os.listdir(dir_path)
        for filename in files:
            file_path = os.path.join(dir_path, filename)
            
            # 跳过目录
            if os.path.isdir(file_path):
                continue
            
            # 获取文件扩展名
            name, ext = os.path.splitext(filename)
            ext = ext.lower()
            
            # 检查是否需要转换格式
            if ext not in ('.jpg', '.jpeg', '.png'):
                # 尝试转换
                new_filename = name + '.jpg'
                new_file_path = os.path.join(dir_path, new_filename)
                
                logger.info(f"转换格式: {filename} -> {new_filename}")
                if not dry_run:
                    if convert_to_jpg(file_path, new_file_path):
                        os.remove(file_path)
                        converted_images += 1
                    else:
                        os.remove(file_path)
                        removed_images += 1
            elif ext == '.png':
                # 转换 PNG 到 JPG
                new_filename = name + '.jpg'
                new_file_path = os.path.join(dir_path, new_filename)
                
                if not os.path.exists(new_file_path):
                    logger.info(f"转换PNG到JPG: {filename} -> {new_filename}")
                    if not dry_run:
                        if convert_to_jpg(file_path, new_file_path):
                            os.remove(file_path)
                            converted_images += 1
    
    logger.info("=" * 60)
    logger.info(f"标准化完成！")
    logger.info(f"删除无效目录: {removed_dirs} 个")
    logger.info(f"转换图片格式: {converted_images} 张")
    logger.info(f"删除无法转换的图片: {removed_images} 张")
    logger.info("=" * 60)

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("用法:")
            print("  python standardize_dataset.py          # 执行标准化")
            print("  python standardize_dataset.py --dry-run # 预览模式")
            print("  python standardize_dataset.py --help   # 显示帮助")
            return
        elif sys.argv[1] == '--dry-run':
            standardize_dataset(
                dataset_path='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset',
                role_list_path='/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt',
                dry_run=True
            )
            return
    
    # 默认执行标准化
    standardize_dataset(
        dataset_path='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset',
        role_list_path='/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt',
        dry_run=False
    )

if __name__ == "__main__":
    main()

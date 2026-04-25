#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测下载数据中的损坏图片并将其移动到损坏目录
"""

import os
import argparse
import logging
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('check_corrupted_images.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CorruptedImageDetector:
    def __init__(self, max_workers=4):
        """
        初始化损坏图片检测器
        
        Args:
            max_workers: 最大线程数
        """
        self.max_workers = max_workers
    
    def is_image_corrupted(self, image_path):
        """
        检查图片是否损坏
        
        Args:
            image_path: 图片路径
        
        Returns:
            bool: 是否损坏
        """
        try:
            # 尝试打开图片
            with Image.open(image_path) as img:
                # 尝试加载图片数据
                img.verify()
                # 再次打开图片并转换为RGB以确保完全可读
                with Image.open(image_path) as img2:
                    img2.convert('RGB')
            return False
        except Exception as e:
            logger.debug(f"损坏图片: {image_path}, 错误: {e}")
            return True
    
    def detect_corrupted_images(self, data_dir):
        """
        检测并移动损坏的图片
        
        Args:
            data_dir: 数据目录
        """
        # 获取所有图片文件
        image_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    image_files.append(os.path.join(root, file))
        
        total_images = len(image_files)
        logger.info(f"找到 {total_images} 张图片")
        
        if not image_files:
            logger.info("没有找到图片")
            return
        
        # 使用线程池并行处理
        corrupted_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_image = {
                executor.submit(self.is_image_corrupted, image_path): image_path 
                for image_path in image_files
            }
            
            for future in tqdm(as_completed(future_to_image), total=total_images):
                image_path = future_to_image[future]
                try:
                    is_corrupted = future.result()
                    if is_corrupted:
                        # 创建损坏目录
                        corrupted_dir = os.path.join(os.path.dirname(image_path), "损坏")
                        os.makedirs(corrupted_dir, exist_ok=True)
                        
                        # 移动损坏的图片
                        dest_path = os.path.join(corrupted_dir, os.path.basename(image_path))
                        os.rename(image_path, dest_path)
                        corrupted_count += 1
                        logger.info(f"移动损坏图片: {image_path} -> {dest_path}")
                except Exception as e:
                    logger.error(f"处理图片失败 {image_path}: {e}")
        
        logger.info(f"检测完成: 共 {total_images} 张图片，{corrupted_count} 张损坏")
        logger.info(f"损坏率: {corrupted_count / total_images * 100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='检测损坏的图片')
    parser.add_argument('--data_dir', type=str, 
                        default='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/role_images',
                        help='数据目录')
    parser.add_argument('--max_workers', type=int, 
                        default=4,
                        help='最大线程数')
    
    args = parser.parse_args()
    
    logger.info(f"开始检测损坏图片")
    logger.info(f"数据目录: {args.data_dir}")
    
    detector = CorruptedImageDetector(max_workers=args.max_workers)
    detector.detect_corrupted_images(args.data_dir)
    
    logger.info("检测完成！")

if __name__ == '__main__':
    main()

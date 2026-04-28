#!/usr/bin/env python3
"""
数据清洗脚本
用于清洗采集的角色图片数据
"""
import os
import sys
import logging
import shutil
from PIL import Image
import hashlib
import concurrent.futures

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clean_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, collection_dir, output_dir=None, min_resolution=(200, 200), min_file_size=1024):
        """
        初始化数据清洗器
        
        Args:
            collection_dir: 采集数据目录
            output_dir: 清洗后输出目录
            min_resolution: 最小分辨率
            min_file_size: 最小文件大小（字节）
        """
        self.collection_dir = collection_dir
        self.output_dir = output_dir or os.path.join(collection_dir, 'cleaned')
        self.min_resolution = min_resolution
        self.min_file_size = min_file_size
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def calculate_hash(self, image_path):
        """
        计算图片哈希值
        
        Args:
            image_path: 图片路径
            
        Returns:
            str: 图片哈希值
        """
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"计算哈希值失败 {image_path}: {e}")
            return None
    
    def is_valid_image(self, image_path, role_hashes=None):
        """
        检查图片是否有效
        
        Args:
            image_path: 图片路径
            role_hashes: 当前角色的哈希集合（用于内部去重）
            
        Returns:
            bool: 是否有效
        """
        try:
            # 检查文件大小
            if os.path.getsize(image_path) < self.min_file_size:
                logger.warning(f"文件太小: {image_path}")
                return False
            
            # 检查图片分辨率
            with Image.open(image_path) as img:
                width, height = img.size
                if width < self.min_resolution[0] or height < self.min_resolution[1]:
                    logger.warning(f"分辨率过低: {image_path} ({width}x{height})")
                    return False
            
            # 检查重复（每个角色内部独立去重）
            img_hash = self.calculate_hash(image_path)
            if role_hashes is not None and img_hash in role_hashes:
                logger.warning(f"重复图片: {image_path}")
                return False
            
            if role_hashes is not None:
                role_hashes.add(img_hash)
            return True
        except Exception as e:
            logger.error(f"检查图片失败 {image_path}: {e}")
            return False
    
    def clean_role(self, role_name, role_dir):
        """
        清洗单个角色的数据
        
        Args:
            role_name: 角色名称
            role_dir: 角色目录
            
        Returns:
            dict: 清洗结果
        """
        logger.info(f"开始清洗角色: {role_name}")
        
        # 查找所有图片
        image_paths = []
        for root, dirs, files in os.walk(role_dir):
            for file in files:
                if file.endswith('.jpg'):
                    image_paths.append(os.path.join(root, file))
        
        total_images = len(image_paths)
        valid_images = 0
        invalid_images = 0
        
        # 创建角色输出目录
        role_output_dir = os.path.join(self.output_dir, role_name)
        os.makedirs(role_output_dir, exist_ok=True)
        
        # 每个角色内部独立去重
        role_hashes = set()
        
        # 处理图片
        for image_path in image_paths:
            if self.is_valid_image(image_path, role_hashes):
                # 复制到输出目录
                output_path = os.path.join(role_output_dir, os.path.basename(image_path))
                shutil.copy2(image_path, output_path)
                valid_images += 1
            else:
                invalid_images += 1
        
        logger.info(f"角色 {role_name} 清洗完成:")
        logger.info(f"  - 总图片数: {total_images}")
        logger.info(f"  - 有效图片: {valid_images}")
        logger.info(f"  - 无效图片: {invalid_images}")
        
        return {
            'role': role_name,
            'total': total_images,
            'valid': valid_images,
            'invalid': invalid_images
        }
    
    def clean(self, max_workers=4):
        """
        清洗所有角色数据
        
        Args:
            max_workers: 最大并发数
            
        Returns:
            dict: 总体清洗结果
        """
        logger.info("开始数据清洗")
        logger.info(f"采集目录: {self.collection_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"最小分辨率: {self.min_resolution}")
        logger.info(f"最小文件大小: {self.min_file_size} 字节")
        
        # 获取所有角色目录
        role_dirs = []
        for item in os.listdir(self.collection_dir):
            item_path = os.path.join(self.collection_dir, item)
            if os.path.isdir(item_path):
                role_dirs.append((item, item_path))
        
        total_roles = len(role_dirs)
        logger.info(f"找到 {total_roles} 个角色")
        
        # 并行处理
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_role = {executor.submit(self.clean_role, role, role_dir): role for role, role_dir in role_dirs}
            for future in concurrent.futures.as_completed(future_to_role):
                role = future_to_role[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"处理角色 {role} 失败: {e}")
        
        # 统计总体结果
        total_images = sum(r['total'] for r in results)
        total_valid = sum(r['valid'] for r in results)
        total_invalid = sum(r['invalid'] for r in results)
        
        logger.info("\n=== 清洗完成 ===")
        logger.info(f"总角色数: {total_roles}")
        logger.info(f"总图片数: {total_images}")
        logger.info(f"有效图片: {total_valid}")
        logger.info(f"无效图片: {total_invalid}")
        logger.info(f"有效率: {total_valid / total_images * 100:.2f}%")
        
        return {
            'total_roles': total_roles,
            'total_images': total_images,
            'valid_images': total_valid,
            'invalid_images': total_invalid,
            'valid_rate': total_valid / total_images * 100 if total_images > 0 else 0
        }

def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='数据清洗脚本')
    parser.add_argument('--collection_dir', type=str, required=True, help='采集数据目录')
    parser.add_argument('--output_dir', type=str, help='清洗后输出目录')
    parser.add_argument('--min_width', type=int, default=200, help='最小宽度')
    parser.add_argument('--min_height', type=int, default=200, help='最小高度')
    parser.add_argument('--min_size', type=int, default=1024, help='最小文件大小（字节）')
    parser.add_argument('--max_workers', type=int, default=4, help='最大并发数')
    
    args = parser.parse_args()
    
    cleaner = DataCleaner(
        collection_dir=args.collection_dir,
        output_dir=args.output_dir,
        min_resolution=(args.min_width, args.min_height),
        min_file_size=args.min_size
    )
    
    result = cleaner.clean(max_workers=args.max_workers)
    
    print("\n=== 清洗结果 ===")
    print(f"总角色数: {result['total_roles']}")
    print(f"总图片数: {result['total_images']}")
    print(f"有效图片: {result['valid_images']}")
    print(f"无效图片: {result['invalid_images']}")
    print(f"有效率: {result['valid_rate']:.2f}%")

if __name__ == "__main__":
    main()

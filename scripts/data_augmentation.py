#!/usr/bin/env python3
"""
数据增强脚本
对现有数据进行增强，生成更多训练样本
"""
import os
import sys
import argparse
import logging
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_augmentation')

class DataAugmentation:
    """数据增强类"""
    
    def __init__(self, input_dir, output_dir, augment_factor=3):
        """初始化数据增强模块
        
        Args:
            input_dir: 输入数据目录
            output_dir: 输出数据目录
            augment_factor: 增强因子，每个图像生成的增强样本数
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.augment_factor = augment_factor
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def augment_image(self, image_path, output_path):
        """增强单个图像
        
        Args:
            image_path: 原始图像路径
            output_path: 输出图像路径
        """
        try:
            # 打开图像
            img = Image.open(image_path)
            
            # 生成增强样本
            augmentations = [
                ('original', img),
                ('flip_horizontal', ImageOps.mirror(img)),
                ('flip_vertical', ImageOps.flip(img)),
                ('rotate_10', img.rotate(10, expand=True)),
                ('rotate_neg_10', img.rotate(-10, expand=True)),
                ('brightness_up', ImageEnhance.Brightness(img).enhance(1.2)),
                ('brightness_down', ImageEnhance.Brightness(img).enhance(0.8)),
                ('contrast_up', ImageEnhance.Contrast(img).enhance(1.2)),
                ('contrast_down', ImageEnhance.Contrast(img).enhance(0.8)),
                ('saturation_up', ImageEnhance.Color(img).enhance(1.2)),
                ('saturation_down', ImageEnhance.Color(img).enhance(0.8)),
            ]
            
            # 保存增强样本
            base_name = os.path.basename(output_path)
            base_name_without_ext = os.path.splitext(base_name)[0]
            ext = os.path.splitext(base_name)[1]
            
            for aug_name, aug_img in augmentations[:self.augment_factor+1]:
                aug_output_path = os.path.join(
                    os.path.dirname(output_path),
                    f"{base_name_without_ext}_{aug_name}{ext}"
                )
                aug_img.save(aug_output_path)
                logger.info(f"增强图像保存成功: {aug_output_path}")
                
        except Exception as e:
            logger.error(f"增强图像失败: {image_path}, 错误: {e}")
    
    def augment_directory(self):
        """增强目录中的所有图像"""
        logger.info(f"开始增强目录: {self.input_dir}")
        
        # 遍历输入目录
        for root, dirs, files in os.walk(self.input_dir):
            for dir_name in dirs:
                input_subdir = os.path.join(root, dir_name)
                # 计算相对路径
                relative_path = os.path.relpath(input_subdir, self.input_dir)
                output_subdir = os.path.join(self.output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                # 处理子目录中的图像
                subdir_files = os.listdir(input_subdir)
                image_files = [f for f in subdir_files 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                logger.info(f"处理目录: {input_subdir}, 发现 {len(image_files)} 个图像")
                
                for image_file in image_files:
                    input_image_path = os.path.join(input_subdir, image_file)
                    output_image_path = os.path.join(output_subdir, image_file)
                    self.augment_image(input_image_path, output_image_path)
    
    def run(self):
        """运行数据增强流程"""
        logger.info("开始数据增强流程")
        logger.info(f"输入目录: {self.input_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"增强因子: {self.augment_factor}")
        
        self.augment_directory()
        logger.info("数据增强流程完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据增强工具')
    parser.add_argument('--input_dir', type=str, required=True, help='输入数据目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出数据目录')
    parser.add_argument('--augment_factor', type=int, default=3, help='增强因子，每个图像生成的增强样本数')
    
    args = parser.parse_args()
    
    # 初始化数据增强模块
    augmentation = DataAugmentation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        augment_factor=args.augment_factor
    )
    
    # 运行数据增强
    augmentation.run()

if __name__ == "__main__":
    main()

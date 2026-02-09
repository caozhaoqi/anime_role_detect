#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强脚本

通过图像变换扩充数据集，增加数据多样性
"""

import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_augmentation')


class DataAugmentor:
    def __init__(self, input_dir='data/train', output_dir='data/train_augmented', augment_factor=2):
        """
        初始化数据增强器
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            augment_factor: 增强倍数
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.augment_factor = augment_factor
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 增强操作
        self.augmentations = [
            self._horizontal_flip,
            self._rotate,
            self._brightness,
            self._contrast,
            self._color_jitter,
            self._gaussian_blur,
            self._sharpness,
            self._crop,
            self._scale,
            self._perspective
        ]
    
    def _horizontal_flip(self, image):
        """水平翻转"""
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    
    def _rotate(self, image):
        """随机旋转"""
        angle = random.uniform(-15, 15)
        return image.rotate(angle, expand=False, fillcolor=(255, 255, 255))
    
    def _brightness(self, image):
        """调整亮度"""
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)
    
    def _contrast(self, image):
        """调整对比度"""
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)
    
    def _color_jitter(self, image):
        """颜色抖动"""
        enhancer = ImageEnhance.Color(image)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)
    
    def _gaussian_blur(self, image):
        """高斯模糊"""
        radius = random.uniform(0.1, 1.0)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _sharpness(self, image):
        """锐化"""
        enhancer = ImageEnhance.Sharpness(image)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)
    
    def _crop(self, image):
        """随机裁剪"""
        width, height = image.size
        crop_ratio = random.uniform(0.8, 0.95)
        
        new_width = int(width * crop_ratio)
        new_height = int(height * crop_ratio)
        
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        
        cropped = image.crop((left, top, left + new_width, top + new_height))
        return cropped.resize((width, height), Image.LANCZOS)
    
    def _scale(self, image):
        """缩放"""
        scale_factor = random.uniform(0.9, 1.1)
        width, height = image.size
        
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        scaled = image.resize((new_width, new_height), Image.LANCZOS)
        
        # 如果缩放后图像变小，填充白色背景
        if scale_factor < 1.0:
            background = Image.new('RGB', (width, height), (255, 255, 255))
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            background.paste(scaled, (left, top))
            return background
        # 如果缩放后图像变大，裁剪中心部分
        else:
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            return scaled.crop((left, top, left + width, top + height))
    
    def _perspective(self, image):
        """简单的透视变换模拟"""
        width, height = image.size
        
        # 随机选择变换类型
        transform_type = random.choice(['shear_x', 'shear_y', 'tilt'])
        
        if transform_type == 'shear_x':
            # 水平剪切
            shear_factor = random.uniform(-0.1, 0.1)
            return image.transform((width, height), Image.AFFINE, 
                                 (1, shear_factor, 0, 0, 1, 0), Image.BICUBIC)
        elif transform_type == 'shear_y':
            # 垂直剪切
            shear_factor = random.uniform(-0.1, 0.1)
            return image.transform((width, height), Image.AFFINE, 
                                 (1, 0, 0, shear_factor, 1, 0), Image.BICUBIC)
        else:
            # 简单的倾斜效果
            angle = random.uniform(-5, 5)
            return image.rotate(angle, expand=False, fillcolor=(255, 255, 255))
    
    def _augment_image(self, image_path, output_path, num_augmentations):
        """
        对单个图像进行增强
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径（不含扩展名）
            num_augmentations: 增强数量
        """
        try:
            image = Image.open(image_path).convert('RGB')
            
            # 保存原始图像
            base_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(base_name)[0]
            ext = '.jpg'
            
            # 生成增强图像
            for i in range(num_augmentations):
                augmented_image = image.copy()
                
                # 随机选择1-3个增强操作
                num_operations = random.randint(1, 3)
                selected_operations = random.sample(self.augmentations, num_operations)
                
                # 应用增强操作
                for operation in selected_operations:
                    try:
                        augmented_image = operation(augmented_image)
                    except Exception as e:
                        logger.warning(f"增强操作失败: {e}")
                        continue
                
                # 保存增强图像
                save_path = os.path.join(output_path, f"{name_without_ext}_aug{i:02d}{ext}")
                augmented_image.save(save_path, 'JPEG', quality=95)
            
            return num_augmentations
        except Exception as e:
            logger.error(f"增强图像失败 {image_path}: {e}")
            return 0
    
    def augment_dataset(self):
        """
        增强整个数据集
        """
        logger.info(f"开始数据增强，输入目录: {self.input_dir}, 输出目录: {self.output_dir}")
        
        total_augmented = 0
        total_processed = 0
        
        # 遍历所有角色目录
        for character_dir in os.listdir(self.input_dir):
            character_path = os.path.join(self.input_dir, character_dir)
            
            if not os.path.isdir(character_path):
                continue
            
            # 创建输出角色目录
            output_character_dir = os.path.join(self.output_dir, character_dir)
            os.makedirs(output_character_dir, exist_ok=True)
            
            # 获取所有图像文件
            image_files = [f for f in os.listdir(character_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not image_files:
                continue
            
            logger.info(f"处理角色: {character_dir}, 图像数量: {len(image_files)}")
            
            # 复制原始图像到输出目录
            for img_file in tqdm(image_files, desc=f"复制 {character_dir}"):
                src_path = os.path.join(character_path, img_file)
                dst_path = os.path.join(output_character_dir, img_file)
                
                # 转换为JPEG格式
                try:
                    image = Image.open(src_path).convert('RGB')
                    image.save(dst_path, 'JPEG', quality=95)
                except Exception as e:
                    logger.error(f"复制图像失败 {src_path}: {e}")
                    continue
            
            # 增强图像
            for img_file in tqdm(image_files, desc=f"增强 {character_dir}"):
                src_path = os.path.join(character_path, img_file)
                name_without_ext = os.path.splitext(img_file)[0]
                
                augmented_count = self._augment_image(
                    src_path, 
                    output_character_dir, 
                    self.augment_factor
                )
                
                total_augmented += augmented_count
                total_processed += 1
        
        logger.info(f"数据增强完成！处理了 {total_processed} 张图像，生成了 {total_augmented} 张增强图像")
        
        # 统计最终数据集
        self._print_statistics()
        
        return total_augmented
    
    def _print_statistics(self):
        """打印数据集统计信息"""
        total_characters = 0
        total_images = 0
        
        for character_dir in os.listdir(self.output_dir):
            character_path = os.path.join(self.output_dir, character_dir)
            
            if not os.path.isdir(character_path):
                continue
            
            image_count = len([f for f in os.listdir(character_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            
            if image_count > 0:
                total_characters += 1
                total_images += image_count
        
        print("\n" + "="*60)
        print("增强后数据集统计")
        print("="*60)
        print(f"总角色数: {total_characters}")
        print(f"总图像数: {total_images}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='数据增强脚本')
    
    parser.add_argument('--input-dir', type=str, 
                       default='data/train',
                       help='输入目录')
    parser.add_argument('--output-dir', type=str, 
                       default='data/train_augmented',
                       help='输出目录')
    parser.add_argument('--augment-factor', type=int, default=2,
                       help='增强倍数（每张图像生成的增强图像数量）')
    
    args = parser.parse_args()
    
    # 初始化增强器
    augmentor = DataAugmentor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        augment_factor=args.augment_factor
    )
    
    # 执行数据增强
    augmentor.augment_dataset()


if __name__ == '__main__':
    main()

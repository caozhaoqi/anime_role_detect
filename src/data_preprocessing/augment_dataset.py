#!/usr/bin/env python3
"""
数据增强脚本
为极小数据集生成更多合成样本
"""
import os
import sys
import argparse
import logging
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_augmentation')


def augment_image(img, num_augmentations=10):
    """
    对单个图像进行多种增强
    
    Args:
        img: PIL图像
        num_augmentations: 增强数量
        
    Returns:
        list: 增强后的图像列表
    """
    augmented_images = []
    
    # 基础变换
    transforms = [
        # 水平翻转
        lambda x: ImageOps.mirror(x),
        # 垂直翻转
        lambda x: ImageOps.flip(x),
        # 旋转
        lambda x: x.rotate(random.randint(-30, 30), expand=False),
        # 亮度调整
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.6, 1.4)),
        # 对比度调整
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.6, 1.4)),
        # 饱和度调整
        lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.6, 1.4)),
        # 锐度调整
        lambda x: ImageEnhance.Sharpness(x).enhance(random.uniform(0.2, 2.0)),
        # 随机裁剪然后缩放
        lambda x: random_crop_and_resize(x),
        # 高斯噪声
        lambda x: add_gaussian_noise(x),
        # 轻微模糊
        lambda x: x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    ]
    
    for i in range(num_augmentations):
        # 复制原始图像
        aug_img = img.copy()
        
        # 应用1-3种随机变换
        num_transforms = random.randint(1, 3)
        selected_transforms = random.sample(transforms, num_transforms)
        
        for transform in selected_transforms:
            try:
                aug_img = transform(aug_img)
            except Exception as e:
                logger.warning(f"变换失败: {e}")
                continue
        
        augmented_images.append(aug_img)
    
    return augmented_images


def random_crop_and_resize(img, crop_ratio=0.8):
    """
    随机裁剪然后缩放
    
    Args:
        img: PIL图像
        crop_ratio: 裁剪比例
        
    Returns:
        PIL图像
    """
    width, height = img.size
    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)
    
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    
    cropped = img.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height))


def add_gaussian_noise(img, mean=0, std=10):
    """
    添加高斯噪声
    
    Args:
        img: PIL图像
        mean: 噪声均值
        std: 噪声标准差
        
    Returns:
        PIL图像
    """
    img_array = np.array(img)
    noise = np.random.normal(mean, std, img_array.shape).astype(np.uint8)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)


def augment_dataset(input_dir, output_dir, num_augmentations_per_image=10):
    """
    增强整个数据集
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        num_augmentations_per_image: 每张图像的增强数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有类别
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    logger.info(f"找到 {len(classes)} 个类别")
    
    total_augmented = 0
    
    for cls in tqdm(classes, desc="增强类别"):
        cls_input_dir = os.path.join(input_dir, cls)
        cls_output_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_output_dir, exist_ok=True)
        
        # 遍历类别中的所有图像
        images = [f for f in os.listdir(cls_input_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        logger.info(f"类别 {cls} 包含 {len(images)} 张图像")
        
        # 复制原始图像
        for img_name in images:
            src_path = os.path.join(cls_input_dir, img_name)
            dst_path = os.path.join(cls_output_dir, img_name)
            try:
                img = Image.open(src_path)
                img.save(dst_path)
            except Exception as e:
                logger.warning(f"复制图像失败 {src_path}: {e}")
                continue
        
        # 增强每个图像
        for img_name in images:
            img_path = os.path.join(cls_input_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                augmented_images = augment_image(img, num_augmentations_per_image)
                
                # 保存增强后的图像
                base_name = os.path.splitext(img_name)[0]
                ext = os.path.splitext(img_name)[1]
                
                for i, aug_img in enumerate(augmented_images):
                    aug_img_name = f"{base_name}_aug_{i}{ext}"
                    aug_img_path = os.path.join(cls_output_dir, aug_img_name)
                    try:
                        aug_img.save(aug_img_path)
                        total_augmented += 1
                    except Exception as e:
                        logger.warning(f"保存增强图像失败 {aug_img_path}: {e}")
                        continue
            except Exception as e:
                logger.warning(f"增强图像失败 {img_path}: {e}")
                continue
    
    logger.info(f"数据增强完成！总共生成 {total_augmented} 张增强图像")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='数据增强脚本 - 为极小数据集生成更多合成样本')
    
    parser.add_argument('--input_dir', type=str, default='data/split_dataset/train', 
                       help='输入数据集目录')
    parser.add_argument('--output_dir', type=str, default='data/augmented_dataset', 
                       help='输出增强数据集目录')
    parser.add_argument('--num_augmentations', type=int, default=20, 
                       help='每张图像的增强数量')
    
    args = parser.parse_args()
    
    logger.info('开始数据增强...')
    logger.info(f'输入目录: {args.input_dir}')
    logger.info(f'输出目录: {args.output_dir}')
    logger.info(f'每张图像的增强数量: {args.num_augmentations}')
    
    # 执行数据增强
    augment_dataset(args.input_dir, args.output_dir, args.num_augmentations)


if __name__ == "__main__":
    # 添加缺失的导入
    from PIL import ImageFilter
    main()

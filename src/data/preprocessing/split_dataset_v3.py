#!/usr/bin/env python3
"""
数据集分割脚本 v3
将data/train目录中的数据分割为训练集和验证集
"""
import os
import shutil
import random
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('split_dataset_v3')


def split_dataset(input_dir, output_dir, train_ratio=0.8, min_samples=10):
    """
    分割数据集为训练集和验证集
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        min_samples: 最小样本数，低于此数量的角色将被过滤
    """
    # 确保输出目录存在
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    # 清空并重新创建目录
    for dir_path in [train_dir, val_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    
    # 获取所有子目录（角色）
    characters = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    logger.info(f"发现 {len(characters)} 个角色")
    
    total_images = 0
    total_train = 0
    total_val = 0
    filtered_characters = 0
    
    for character in characters:
        char_input_dir = os.path.join(input_dir, character)
        char_train_dir = os.path.join(train_dir, character)
        char_val_dir = os.path.join(val_dir, character)
        
        # 获取所有图像文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([f for f in os.listdir(char_input_dir) if f.lower().endswith(ext)])
        
        if not image_files:
            logger.warning(f"角色 {character} 没有图像文件")
            continue
        
        # 过滤样本数量过少的角色
        if len(image_files) < min_samples:
            filtered_characters += 1
            logger.warning(f"角色 {character} 样本数量不足 ({len(image_files)} < {min_samples})，已过滤")
            continue
        
        # 打乱图像顺序
        random.shuffle(image_files)
        
        # 分割图像
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # 确保验证集至少有1张图片
        if len(val_files) == 0:
            val_files = [train_files[-1]]
            train_files = train_files[:-1]
        
        # 确保角色目录存在
        os.makedirs(char_train_dir, exist_ok=True)
        os.makedirs(char_val_dir, exist_ok=True)
        
        # 复制到训练集
        for img_file in tqdm(train_files, desc=f"复制 {character} 训练图像"):
            src_path = os.path.join(char_input_dir, img_file)
            dst_path = os.path.join(char_train_dir, img_file)
            shutil.copy2(src_path, dst_path)
        
        # 复制到验证集
        for img_file in tqdm(val_files, desc=f"复制 {character} 验证图像"):
            src_path = os.path.join(char_input_dir, img_file)
            dst_path = os.path.join(char_val_dir, img_file)
            shutil.copy2(src_path, dst_path)
        
        total_images += len(image_files)
        total_train += len(train_files)
        total_val += len(val_files)
        
        logger.info(f"角色 {character}: 总图像 {len(image_files)}, 训练集 {len(train_files)}, 验证集 {len(val_files)}")
    
    logger.info(f"数据集分割完成！")
    logger.info(f"总角色数: {len(characters)}, 过滤角色数: {filtered_characters}, 有效角色数: {len(characters) - filtered_characters}")
    logger.info(f"总图像: {total_images}")
    logger.info(f"训练集: {total_train} ({total_train/total_images*100:.2f}%)")
    logger.info(f"验证集: {total_val} ({total_val/total_images*100:.2f}%)")


def main():
    """
    主函数
    """
    input_dir = 'data/train'
    output_dir = 'data/split_dataset_v3'
    
    logger.info('开始分割数据集...')
    logger.info(f'输入目录: {input_dir}')
    logger.info(f'输出目录: {output_dir}')
    
    split_dataset(input_dir, output_dir, train_ratio=0.8, min_samples=10)
    
    logger.info('数据集分割完成！')


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
数据集分割脚本
将all_characters目录下的数据集分割为训练集和验证集
"""
import os
import shutil
import random
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('split_dataset')

def split_dataset(source_dir, train_dir, val_dir, val_ratio=0.2):
    """分割数据集
    
    Args:
        source_dir: 源数据集目录
        train_dir: 训练集目录
        val_dir: 验证集目录
        val_ratio: 验证集比例
    """
    # 确保目标目录存在
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 获取所有角色目录
    character_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    logger.info(f"找到 {len(character_dirs)} 个角色目录")
    
    total_images = 0
    total_train = 0
    total_val = 0
    
    for character in character_dirs:
        character_path = os.path.join(source_dir, character)
        images = [f for f in os.listdir(character_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not images:
            logger.warning(f"角色 {character} 没有图像文件")
            continue
        
        # 打乱图像顺序
        random.shuffle(images)
        
        # 计算分割点
        val_size = int(len(images) * val_ratio)
        train_images = images[val_size:]
        val_images = images[:val_size]
        
        # 确保角色在训练集和验证集中的目录存在
        train_character_dir = os.path.join(train_dir, character)
        val_character_dir = os.path.join(val_dir, character)
        os.makedirs(train_character_dir, exist_ok=True)
        os.makedirs(val_character_dir, exist_ok=True)
        
        # 复制图像到训练集
        for img in train_images:
            src = os.path.join(character_path, img)
            dst = os.path.join(train_character_dir, img)
            shutil.copy2(src, dst)
        
        # 复制图像到验证集
        for img in val_images:
            src = os.path.join(character_path, img)
            dst = os.path.join(val_character_dir, img)
            shutil.copy2(src, dst)
        
        # 记录统计信息
        total_images += len(images)
        total_train += len(train_images)
        total_val += len(val_images)
        
        logger.info(f"角色 {character}: 总图像 {len(images)}, 训练集 {len(train_images)}, 验证集 {len(val_images)}")
    
    logger.info(f"数据集分割完成: 总图像 {total_images}, 训练集 {total_train}, 验证集 {total_val}")
    logger.info(f"训练集目录: {train_dir}")
    logger.info(f"验证集目录: {val_dir}")

def main():
    """主函数"""
    source_dir = 'data/all_characters'
    train_dir = 'data/split_dataset/train'
    val_dir = 'data/split_dataset/val'
    val_ratio = 0.2
    
    logger.info('开始分割数据集...')
    logger.info(f"源数据集目录: {source_dir}")
    logger.info(f"训练集目录: {train_dir}")
    logger.info(f"验证集目录: {val_dir}")
    logger.info(f"验证集比例: {val_ratio}")
    
    split_dataset(source_dir, train_dir, val_dir, val_ratio)
    
    logger.info('数据集分割完成！')

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集分割脚本

将数据集分割为训练集和验证集
"""

import os
import sys
import shutil
import random
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('split_dataset')


def split_dataset(source_dir, output_dir, train_ratio=0.8, seed=42):
    """
    分割数据集
    
    Args:
        source_dir: 源数据目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 遍历所有角色目录
    character_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    logger.info(f"开始分割数据集，共 {len(character_dirs)} 个角色")
    
    for character_dir in character_dirs:
        character_path = os.path.join(source_dir, character_dir)
        
        # 跳过非角色目录
        if not character_dir.startswith('sdv50_'):
            continue
        
        # 获取所有图片
        images = [f for f in os.listdir(character_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not images:
            logger.warning(f"角色 {character_dir} 没有图片，跳过")
            continue
        
        # 随机打乱
        random.shuffle(images)
        
        # 分割数据
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # 创建角色目录
        train_character_dir = os.path.join(train_dir, character_dir)
        val_character_dir = os.path.join(val_dir, character_dir)
        
        os.makedirs(train_character_dir, exist_ok=True)
        os.makedirs(val_character_dir, exist_ok=True)
        
        # 复制训练集图片
        for img_name in train_images:
            src_path = os.path.join(character_path, img_name)
            dst_path = os.path.join(train_character_dir, img_name)
            shutil.copy2(src_path, dst_path)
        
        # 复制验证集图片
        for img_name in val_images:
            src_path = os.path.join(character_path, img_name)
            dst_path = os.path.join(val_character_dir, img_name)
            shutil.copy2(src_path, dst_path)
        
        logger.info(f"角色 {character_dir}: 训练集 {len(train_images)} 张, 验证集 {len(val_images)} 张")
    
    # 统计信息
    train_count = sum(len([f for f in os.listdir(os.path.join(train_dir, d)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) 
                     for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    val_count = sum(len([f for f in os.listdir(os.path.join(val_dir, d)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) 
                   for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d)))
    
    logger.info(f"数据集分割完成")
    logger.info(f"训练集: {train_count} 张图片")
    logger.info(f"验证集: {val_count} 张图片")
    logger.info(f"总计: {train_count + val_count} 张图片")


if __name__ == '__main__':
    # 分割sdv50_train数据集
    split_dataset(
        source_dir='data/sdv50_train',
        output_dir='data/split_dataset',
        train_ratio=0.8,
        seed=42
    )

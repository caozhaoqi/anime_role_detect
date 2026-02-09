#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分脚本

将数据集划分为训练集和验证集
"""

import os
import argparse
import shutil
import random
import logging
from pathlib import Path
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('dataset_splitter')


class DatasetSplitter:
    def __init__(self, input_dir='data/train', output_dir='data/split_dataset', 
                 train_ratio=0.8, val_ratio=0.2):
        """
        初始化数据集划分器
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # 创建输出目录
        self.train_dir = os.path.join(output_dir, 'train')
        self.val_dir = os.path.join(output_dir, 'val')
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        
        # 设置随机种子
        random.seed(42)
    
    def split_dataset(self):
        """
        划分数据集
        """
        logger.info(f"开始划分数据集，输入目录: {self.input_dir}")
        logger.info(f"训练集比例: {self.train_ratio}, 验证集比例: {self.val_ratio}")
        
        total_train = 0
        total_val = 0
        total_characters = 0
        
        # 遍历所有角色目录
        for character_dir in os.listdir(self.input_dir):
            character_path = os.path.join(self.input_dir, character_dir)
            
            if not os.path.isdir(character_path):
                continue
            
            # 获取所有图像文件
            image_files = [f for f in os.listdir(character_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not image_files:
                continue
            
            # 随机打乱图像列表
            random.shuffle(image_files)
            
            # 计算训练集和验证集的数量
            num_images = len(image_files)
            num_train = int(num_images * self.train_ratio)
            num_val = num_images - num_train
            
            # 创建角色目录
            train_character_dir = os.path.join(self.train_dir, character_dir)
            val_character_dir = os.path.join(self.val_dir, character_dir)
            os.makedirs(train_character_dir, exist_ok=True)
            os.makedirs(val_character_dir, exist_ok=True)
            
            # 分割图像
            train_images = image_files[:num_train]
            val_images = image_files[num_train:]
            
            # 复制训练集图像
            for img_file in tqdm(train_images, desc=f"复制 {character_dir} 训练集"):
                src_path = os.path.join(character_path, img_file)
                dst_path = os.path.join(train_character_dir, img_file)
                shutil.copy2(src_path, dst_path)
            
            # 复制验证集图像
            for img_file in tqdm(val_images, desc=f"复制 {character_dir} 验证集"):
                src_path = os.path.join(character_path, img_file)
                dst_path = os.path.join(val_character_dir, img_file)
                shutil.copy2(src_path, dst_path)
            
            total_train += len(train_images)
            total_val += len(val_images)
            total_characters += 1
            
            logger.info(f"{character_dir}: 训练集 {len(train_images)} 张, 验证集 {len(val_images)} 张")
        
        logger.info(f"数据集划分完成！")
        logger.info(f"总角色数: {total_characters}")
        logger.info(f"训练集总图像数: {total_train}")
        logger.info(f"验证集总图像数: {total_val}")
        logger.info(f"总图像数: {total_train + total_val}")
        
        return {
            'total_characters': total_characters,
            'train_images': total_train,
            'val_images': total_val,
            'total_images': total_train + total_val
        }


def main():
    parser = argparse.ArgumentParser(description='数据集划分脚本')
    
    parser.add_argument('--input-dir', type=str, 
                       default='data/train',
                       help='输入目录')
    parser.add_argument('--output-dir', type=str, 
                       default='data/split_dataset',
                       help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='验证集比例')
    
    args = parser.parse_args()
    
    # 初始化划分器
    splitter = DatasetSplitter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    # 执行数据集划分
    stats = splitter.split_dataset()


if __name__ == '__main__':
    main()

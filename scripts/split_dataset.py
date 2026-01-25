#!/usr/bin/env python3
"""
数据集划分脚本
将增强后的数据划分为训练集和验证集
"""
import os
import shutil
import random
import argparse
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('split_dataset')


def split_dataset(input_dir, output_dir, train_ratio=0.8):
    """划分数据集为训练集和验证集
    
    Args:
        input_dir: 输入数据目录
        output_dir: 输出数据目录
        train_ratio: 训练集比例
    """
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    logger.info(f"开始划分数据集，输入目录: {input_dir}")
    logger.info(f"训练集比例: {train_ratio}, 验证集比例: {1-train_ratio}")
    
    # 遍历角色目录
    role_dirs = [d for d in os.listdir(input_dir) 
                if os.path.isdir(os.path.join(input_dir, d))]
    
    logger.info(f"找到 {len(role_dirs)} 个角色目录")
    
    total_train = 0
    total_val = 0
    
    for role in role_dirs:
        role_path = os.path.join(input_dir, role)
        
        # 获取角色的所有图像
        images = [f for f in os.listdir(role_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not images:
            logger.warning(f"角色 {role} 目录中没有图像")
            continue
        
        # 随机打乱图像
        random.shuffle(images)
        
        # 计算划分索引
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # 复制到训练目录
        train_role_dir = os.path.join(train_dir, role)
        os.makedirs(train_role_dir, exist_ok=True)
        
        for img in train_images:
            src = os.path.join(role_path, img)
            dst = os.path.join(train_role_dir, img)
            shutil.copy2(src, dst)
        
        # 复制到验证目录
        val_role_dir = os.path.join(val_dir, role)
        os.makedirs(val_role_dir, exist_ok=True)
        
        for img in val_images:
            src = os.path.join(role_path, img)
            dst = os.path.join(val_role_dir, img)
            shutil.copy2(src, dst)
        
        logger.info(f"角色 {role}: 训练集 {len(train_images)} 张, 验证集 {len(val_images)} 张")
        total_train += len(train_images)
        total_val += len(val_images)
    
    logger.info(f"数据集划分完成！")
    logger.info(f"总训练集: {total_train} 张图像")
    logger.info(f"总验证集: {total_val} 张图像")
    logger.info(f"输出目录: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据集划分工具')
    parser.add_argument('--input_dir', type=str, default='data/augmented_characters', help='输入数据目录')
    parser.add_argument('--output_dir', type=str, default='data/split_dataset', help='输出数据目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    
    args = parser.parse_args()
    
    split_dataset(args.input_dir, args.output_dir, args.train_ratio)


if __name__ == "__main__":
    main()

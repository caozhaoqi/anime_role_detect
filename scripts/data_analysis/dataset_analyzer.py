#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集分析脚本

分析当前数据集的分布情况，识别数据不足的角色
"""

import os
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dataset_analyzer')


class DatasetAnalyzer:
    def __init__(self, input_dir='data/train'):
        """
        初始化数据集分析器
        
        Args:
            input_dir: 输入目录
        """
        self.input_dir = input_dir
    
    def analyze_dataset(self):
        """
        分析数据集
        """
        logger.info(f"开始分析数据集，输入目录: {self.input_dir}")
        
        character_stats = {}
        total_images = 0
        total_characters = 0
        
        # 遍历所有角色目录
        for character_dir in os.listdir(self.input_dir):
            character_path = os.path.join(self.input_dir, character_dir)
            
            if not os.path.isdir(character_path):
                continue
            
            # 统计图像数量
            image_files = [f for f in os.listdir(character_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not image_files:
                continue
            
            image_count = len(image_files)
            character_stats[character_dir] = image_count
            total_images += image_count
            total_characters += 1
        
        # 排序角色统计
        sorted_stats = sorted(character_stats.items(), key=lambda x: x[1])
        
        logger.info(f"数据集分析完成！")
        logger.info(f"总角色数: {total_characters}")
        logger.info(f"总图像数: {total_images}")
        logger.info(f"平均每个角色图像数: {total_images / total_characters:.2f}")
        
        # 分析数据不足的角色
        insufficient_characters = [(char, count) for char, count in sorted_stats if count < 50]
        logger.info(f"数据不足的角色 (少于50张): {len(insufficient_characters)} 个")
        
        print("\n" + "="*80)
        print("数据集分析报告")
        print("="*80)
        print(f"总角色数: {total_characters}")
        print(f"总图像数: {total_images}")
        print(f"平均每个角色图像数: {total_images / total_characters:.2f}")
        print(f"\n数据不足的角色 (少于50张): {len(insufficient_characters)} 个")
        print("\n数据不足的角色列表:")
        print("-" * 50)
        for char, count in insufficient_characters[:20]:  # 只显示前20个
            print(f"{char}: {count} 张")
        
        if len(insufficient_characters) > 20:
            print(f"... 还有 {len(insufficient_characters) - 20} 个角色")
        
        print("\n" + "="*80)
        print("角色图像数量分布:")
        print("-" * 80)
        
        # 统计不同范围的角色数量
        ranges = [(0, 20), (20, 50), (50, 100), (100, float('inf'))]
        for i, (min_count, max_count) in enumerate(ranges):
            if max_count == float('inf'):
                count = len([c for c, cnt in character_stats.items() if cnt >= min_count])
                print(f"{min_count}+ 张: {count} 个角色")
            else:
                count = len([c for c, cnt in character_stats.items() if min_count <= cnt < max_count])
                print(f"{min_count}-{max_count} 张: {count} 个角色")
        
        print("="*80)
        
        return {
            'character_stats': character_stats,
            'total_characters': total_characters,
            'total_images': total_images,
            'average_images_per_character': total_images / total_characters,
            'insufficient_characters': insufficient_characters
        }
    
    def visualize_distribution(self):
        """
        可视化数据集分布
        """
        stats = self.analyze_dataset()
        character_stats = stats['character_stats']
        insufficient_characters = stats['insufficient_characters']
        
        # 绘制图像数量分布直方图
        image_counts = list(character_stats.values())
        
        plt.figure(figsize=(12, 6))
        plt.hist(image_counts, bins=20, edgecolor='black')
        plt.title('角色图像数量分布')
        plt.xlabel('图像数量')
        plt.ylabel('角色数量')
        plt.grid(axis='y', alpha=0.75)
        
        # 保存图表
        plt.savefig('dataset_distribution.png')
        logger.info('数据集分布图表已保存为 dataset_distribution.png')
        
        # 绘制数据不足的角色
        if insufficient_characters:
            chars, counts = zip(*insufficient_characters)
            
            plt.figure(figsize=(12, 8))
            plt.barh(chars[:20], counts[:20])  # 只显示前20个
            plt.title('数据不足的角色 (前20个)')
            plt.xlabel('图像数量')
            plt.ylabel('角色')
            plt.grid(axis='x', alpha=0.75)
            plt.tight_layout()
            
            plt.savefig('insufficient_characters.png')
            logger.info('数据不足角色图表已保存为 insufficient_characters.png')


def main():
    parser = argparse.ArgumentParser(description='数据集分析脚本')
    
    parser.add_argument('--input-dir', type=str, 
                       default='data/train',
                       help='输入目录')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化数据集分布')
    
    args = parser.parse_args()
    
    # 初始化分析器
    analyzer = DatasetAnalyzer(
        input_dir=args.input_dir
    )
    
    if args.visualize:
        analyzer.visualize_distribution()
    else:
        analyzer.analyze_dataset()


if __name__ == '__main__':
    main()

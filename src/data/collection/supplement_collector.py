#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据不足角色补充采集脚本

根据数据集分析结果，为数据不足的角色采集更多图像
"""

import os
import argparse
import logging
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.collection.keyword_based_collector import KeywordBasedDataCollector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_supplement_collector')


class DataSupplementCollector:
    def __init__(self, auto_spider_img_dir='auto_spider_img'):
        """
        初始化数据补充采集器
        
        Args:
            auto_spider_img_dir: 关键词文件目录
        """
        self.auto_spider_img_dir = auto_spider_img_dir
        self.collector = KeywordBasedDataCollector()
        
    def get_insufficient_characters(self, train_dir='data/train', threshold=50):
        """
        获取数据不足的角色
        
        Args:
            train_dir: 训练集目录
            threshold: 数据不足的阈值
            
        Returns:
            数据不足的角色列表
        """
        insufficient_characters = []
        
        for character_dir in os.listdir(train_dir):
            character_path = os.path.join(train_dir, character_dir)
            
            if not os.path.isdir(character_path):
                continue
            
            # 统计图像数量
            image_files = [f for f in os.listdir(character_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not image_files:
                continue
            
            image_count = len(image_files)
            
            if image_count < threshold:
                # 解析角色信息
                parts = character_dir.split('_')
                if len(parts) >= 3:
                    series_name = '_'.join(parts[:-1])
                    character_name = parts[-1]
                    insufficient_characters.append({
                        'full_name': character_dir,
                        'series_name': series_name,
                        'character_name': character_name,
                        'current_count': image_count
                    })
        
        # 按当前数量排序
        insufficient_characters.sort(key=lambda x: x['current_count'])
        return insufficient_characters
    
    def collect_supplement_data(self, threshold=50, max_images_per_character=100):
        """
        为数据不足的角色采集补充数据
        
        Args:
            threshold: 数据不足的阈值
            max_images_per_character: 每个角色的最大图像数
        """
        # 获取数据不足的角色
        insufficient_characters = self.get_insufficient_characters(threshold=threshold)
        
        if not insufficient_characters:
            logger.info("没有数据不足的角色需要补充")
            return
        
        logger.info(f"发现 {len(insufficient_characters)} 个数据不足的角色，开始补充采集")
        
        # 为每个数据不足的角色采集更多图像
        for char_info in insufficient_characters:
            series_name = char_info['series_name']
            character_name = char_info['character_name']
            current_count = char_info['current_count']
            
            # 计算需要采集的图像数
            needed_count = max(0, threshold - current_count)
            if needed_count <= 0:
                continue
            
            logger.info(f"为角色 {char_info['full_name']} 采集补充数据")
            logger.info(f"当前数量: {current_count}, 需要补充: {needed_count}")
            
            try:
                # 调用现有的采集方法
                # 注意：这里需要根据实际的series_name和character_name格式进行调整
                # 由于现有的collect_from_keywords方法是按系列采集的，我们需要修改它以支持单个角色采集
                
                # 这里我们简单地为每个系列重新采集，让它自动补充数据
                # 实际项目中可能需要更精确的控制
                result = self.collector.collect_from_keywords(series_name, max_images=max_images_per_character)
                
                if result:
                    logger.info(f"角色 {char_info['full_name']} 补充采集完成")
                else:
                    logger.warning(f"角色 {char_info['full_name']} 补充采集失败")
                    
            except Exception as e:
                logger.error(f"采集角色 {char_info['full_name']} 时出错: {str(e)}")
                continue
        
        logger.info("数据补充采集完成！")


def main():
    parser = argparse.ArgumentParser(description='数据不足角色补充采集脚本')
    
    parser.add_argument('--threshold', type=int, 
                       default=50,
                       help='数据不足的阈值')
    parser.add_argument('--max-images', type=int, 
                       default=100,
                       help='每个角色的最大图像数')
    parser.add_argument('--auto-spider-img-dir', type=str, 
                       default='auto_spider_img',
                       help='关键词文件目录')
    
    args = parser.parse_args()
    
    # 初始化采集器
    collector = DataSupplementCollector(
        auto_spider_img_dir=args.auto_spider_img_dir
    )
    
    # 开始采集
    collector.collect_supplement_data(
        threshold=args.threshold,
        max_images_per_character=args.max_images
    )


if __name__ == '__main__':
    main()

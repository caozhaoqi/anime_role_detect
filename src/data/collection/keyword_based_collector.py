#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于关键词文件的数据采集脚本

从 auto_spider_img 文件夹中的关键词文件读取角色信息，采集图像数据
"""

import os
import argparse
import logging
import time
import random
from tqdm import tqdm
from concurrent.futures import as_completed
from src.utils.concurrency_manager import ConcurrencyManager
from PIL import Image
from io import BytesIO
from pathlib import Path

# 导入工具类
from src.utils.http_utils import HTTPUtils
from src.utils.image_utils import ImageUtils
from src.utils.config_manager import config_manager
from src.utils.data_source_manager import DataSourceManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('keyword_data_collector')


class KeywordBasedDataCollector:
    def __init__(self, output_dir='data/train', max_workers=5):
        """
        初始化基于关键词的数据采集器
        
        Args:
            output_dir: 输出目录
            max_workers: 最大并发数
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建HTTP工具实例
        self.http_utils = HTTPUtils(
            max_retries=config_manager.get('network.max_retries'),
            backoff_factor=config_manager.get('network.backoff_factor'),
            timeout=config_manager.get('network.timeout')
        )
        
        # 创建图片工具实例
        self.image_utils = ImageUtils()
        
        # 创建并发管理器实例
        self.concurrency_manager = ConcurrencyManager(
            min_workers=config_manager.get('concurrency.min_workers'),
            max_workers=config_manager.get('concurrency.max_workers_limit'),
            check_interval=5
        )
        
        # 创建数据源管理器实例
        self.data_source_manager = DataSourceManager()
        
        # 关键词文件映射
        self.keyword_files = {
            'genshin_chinese': 'auto_spider_img/1_genshin_chinese_spider_img_keyword.txt',
            'genshin_english': 'auto_spider_img/2_genshin_english_spider_img_keyword.txt',
            'star_rail_chinese': 'auto_spider_img/3_star_rail_chinese_spider_img_keyword.txt',
            'star_rail_english': 'auto_spider_img/4_star_rail_english_spider_img_keyword.txt',
            'honkai3_english': 'auto_spider_img/5_honkai3_english_spider_img_keyword.txt',
            'honkai3_chinese': 'auto_spider_img/6_honkai3_chinese_spider_img_keyword.txt'
        }
    
    def load_keywords(self, keyword_file):
        """
        从关键词文件加载角色关键词
        
        Args:
            keyword_file: 关键词文件路径
            
        Returns:
            关键词列表
        """
        keywords = []
        
        try:
            with open(keyword_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        keywords.append(line)
            
            logger.info(f"从 {keyword_file} 加载了 {len(keywords)} 个关键词")
        except Exception as e:
            logger.error(f"加载关键词文件失败 {keyword_file}: {e}")
        
        return keywords
    
    def load_all_keywords(self):
        """
        加载所有关键词文件
        
        Returns:
            角色字典 {series: [keywords]}
        """
        all_keywords = {}
        
        for series, file_path in self.keyword_files.items():
            if os.path.exists(file_path):
                keywords = self.load_keywords(file_path)
                all_keywords[series] = keywords
            else:
                logger.warning(f"关键词文件不存在: {file_path}")
        
        return all_keywords
    

    
    def _download_image(self, url, save_path, character_name=None):
        """
        下载图像
        
        Args:
            url: 图像URL
            save_path: 保存路径
            character_name: 角色名称（用于内容相关性分析）
            
        Returns:
            是否下载成功
        """
        try:
            time.sleep(random.uniform(0.3, 0.8))
            
            # 下载图片内容
            content = self.http_utils.download_file(url)
            
            # 验证图片
            if not self.image_utils.validate_image(content):
                return False
            
            # 检查图片大小
            min_size = config_manager.get('collection.min_image_size')
            if not self.image_utils.check_image_size(content, min_size):
                return False
            
            # 计算图片质量分数
            quality_score = self.image_utils.calculate_image_quality(content)
            min_quality = config_manager.get('collection.min_quality_score', 60)
            if quality_score < min_quality:
                logger.warning(f"图片质量过低: {quality_score:.2f}, 跳过保存")
                return False
            
            # 分析图片内容
            content_analysis = self.image_utils.analyze_image_content(content)
            logger.debug(f"图片分析结果: {content_analysis}")
            
            # 检查内容类型
            if content_analysis['content_type'] in ['small_image', 'extreme_aspect_ratio']:
                logger.warning(f"图片内容类型不符合要求: {content_analysis['content_type']}, 跳过保存")
                return False
            
            # 计算内容相关性（如果提供了角色名称）
            if character_name:
                relevance_score = self.image_utils.calculate_content_relevance(content, character_name)
                min_relevance = config_manager.get('collection.min_relevance_score', 50)
                if relevance_score < min_relevance:
                    logger.warning(f"图片内容相关性过低: {relevance_score:.2f}, 跳过保存")
                    return False
            
            # 创建保存目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存图片
            filename = os.path.basename(save_path)
            output_dir = os.path.dirname(save_path)
            
            if self.image_utils.save_image(content, output_dir, filename, min_size):
                # 记录图片质量信息
                logger.info(f"图片保存成功: {save_path}, 质量分数: {quality_score:.2f}")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"下载图片失败: {url}, 错误: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            return False
    
    def _process_character(self, series, character_name, max_images=100):
        """
        处理单个角色
        
        Args:
            series: 系列名称
            character_name: 角色名称
            max_images: 最大图像数
            
        Returns:
            最终图像数量
        """
        # 创建角色目录
        character_dir = os.path.join(self.output_dir, f"{series}_{character_name}")
        os.makedirs(character_dir, exist_ok=True)
        
        # 获取已有图像数量
        existing_images = [f for f in os.listdir(character_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        existing_count = len(existing_images)
        
        if existing_count >= max_images:
            logger.info(f"{series}_{character_name} 已有 {existing_count} 张图像，达到上限")
            return existing_count
        
        needed_count = max_images - existing_count
        logger.info(f"开始为 {series}_{character_name} 收集 {needed_count} 张图像")
        
        # 生成搜索查询
        search_queries = [
            character_name,
            f"{character_name} anime",
            f"{character_name} character",
            f"{character_name} official"
        ]
        
        # 从多个数据源获取图像
        all_images = []
        for query in search_queries:
            if len(all_images) >= needed_count:
                break
            
            # 使用数据源管理器获取图像
            source_images = self.data_source_manager.fetch_images(
                query=query,
                limit=needed_count - len(all_images),
                max_sources=3  # 最多尝试3个数据源
            )
            all_images.extend(source_images)
        
        # 记录数据源性能
        source_stats = self.data_source_manager.get_data_source_stats()
        logger.info(f"数据源性能统计: {source_stats}")
        
        # 去重
        seen_urls = set()
        unique_images = []
        for img in all_images:
            if img['url'] not in seen_urls:
                seen_urls.add(img['url'])
                unique_images.append(img)
        all_images = unique_images
        
        # 下载图像
        downloaded_count = 0
        future_to_url = {}
        
        for i, img in enumerate(all_images[:needed_count]):
            save_path = os.path.join(character_dir, f"{series}_{character_name}_{existing_count + i:04d}.jpg")
            future = self.concurrency_manager.submit(self._download_image, img['url'], save_path, character_name)
            future_to_url[future] = img['url']
        
        for future in tqdm(as_completed(future_to_url), total=len(future_to_url), 
                          desc=f"下载 {series}_{character_name} 图像"):
            if future.result():
                downloaded_count += 1
        
        final_count = existing_count + downloaded_count
        logger.info(f"{series}_{character_name} 数据收集完成，当前共有 {final_count} 张图像")
        return final_count
    
    def collect_from_keywords(self, series_name, max_images=100):
        """
        从关键词文件采集数据
        
        Args:
            series_name: 系列名称
            max_images: 每个角色的最大图像数
            
        Returns:
            采集结果字典
        """
        if series_name not in self.keyword_files:
            logger.error(f"未知的系列名称: {series_name}")
            return {}
        
        keyword_file = self.keyword_files[series_name]
        keywords = self.load_keywords(keyword_file)
        
        if not keywords:
            logger.error(f"无法加载关键词文件: {keyword_file}")
            return {}
        
        logger.info(f"开始为系列 {series_name} 采集数据，共 {len(keywords)} 个角色")
        
        results = {}
        for i, character in enumerate(keywords):
            logger.info(f"处理角色 {i+1}/{len(keywords)}")
            count = self._process_character(series_name, character, max_images)
            results[f"{series_name}_{character}"] = count
            
            # 角色之间增加延迟
            if i < len(keywords) - 1:
                time.sleep(random.uniform(2.0, 4.0))
        
        logger.info(f"系列 {series_name} 数据采集完成")
        return results
    
    def collect_all_series(self, max_images=100):
        """
        采集所有系列的数据
        
        Args:
            max_images: 每个角色的最大图像数
            
        Returns:
            采集结果字典
        """
        all_results = {}
        
        for series_name in self.keyword_files.keys():
            logger.info(f"开始采集系列: {series_name}")
            results = self.collect_from_keywords(series_name, max_images)
            all_results.update(results)
            
            # 系列之间增加更长的延迟
            time.sleep(random.uniform(3.0, 5.0))
        
        return all_results
    
    def get_dataset_statistics(self):
        """
        获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        statistics = {
            'total_characters': 0,
            'total_images': 0,
            'characters': {}
        }
        
        for character_dir in os.listdir(self.output_dir):
            character_path = os.path.join(self.output_dir, character_dir)
            if os.path.isdir(character_path):
                images = [f for f in os.listdir(character_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                image_count = len(images)
                
                if image_count > 0:
                    statistics['total_characters'] += 1
                    statistics['total_images'] += image_count
                    statistics['characters'][character_dir] = image_count
        
        return statistics
    
    def validate_dataset(self):
        """
        验证数据集
        
        Returns:
            删除的无效图像数量
        """
        logger.info("开始验证数据集")
        
        invalid_count = 0
        total_count = 0
        
        for character_dir in os.listdir(self.output_dir):
            character_path = os.path.join(self.output_dir, character_dir)
            if os.path.isdir(character_path):
                for img_name in os.listdir(character_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        total_count += 1
                        img_path = os.path.join(character_path, img_name)
                        
                        try:
                            with Image.open(img_path) as img:
                                img.verify()
                        except Exception as e:
                            logger.warning(f"无效图像: {img_path}")
                            os.remove(img_path)
                            invalid_count += 1
        
        logger.info(f"数据集验证完成，检查了 {total_count} 张图像，删除了 {invalid_count} 张无效图像")
        return invalid_count


def main():
    parser = argparse.ArgumentParser(description='基于关键词文件的数据采集脚本')
    
    parser.add_argument('--output-dir', type=str, 
                       default='data/train',
                       help='输出目录')
    parser.add_argument('--max-workers', type=int, default=5,
                       help='最大并发数')
    parser.add_argument('--max-images', type=int, default=100,
                       help='每个角色的最大图像数')
    parser.add_argument('--series', type=str, 
                       help='指定要采集的系列 (genshin_chinese, genshin_english, star_rail_chinese, star_rail_english, honkai3_english, honkai3_chinese)')
    parser.add_argument('--all-series', action='store_true',
                       help='采集所有系列的数据')
    parser.add_argument('--validate', action='store_true',
                       help='验证数据集')
    parser.add_argument('--stats', action='store_true',
                       help='获取数据集统计信息')
    
    args = parser.parse_args()
    
    # 初始化采集器
    collector = KeywordBasedDataCollector(
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # 验证数据集
    if args.validate:
        collector.validate_dataset()
    
    # 获取统计信息
    if args.stats:
        stats = collector.get_dataset_statistics()
        print("\n" + "="*60)
        print("数据集统计信息")
        print("="*60)
        print(f"总角色数: {stats['total_characters']}")
        print(f"总图像数: {stats['total_images']}")
        print("\n各角色图像数 (前20个):")
        for character, count in sorted(stats['characters'].items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {character}: {count}")
        print("="*60)
    
    # 采集数据
    if args.all_series:
        results = collector.collect_all_series(args.max_images)
        
        print("\n" + "="*60)
        print("数据采集结果")
        print("="*60)
        print(f"总共采集了 {len(results)} 个角色")
        total_images = sum(results.values())
        print(f"总共采集了 {total_images} 张图像")
        print("="*60)
        
        # 验证数据集
        collector.validate_dataset()
        
    elif args.series:
        results = collector.collect_from_keywords(args.series, args.max_images)
        
        print("\n" + "="*60)
        print(f"系列 {args.series} 数据采集结果")
        print("="*60)
        for character, count in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {character}: {count}")
        print("="*60)
        
        # 验证数据集
        collector.validate_dataset()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于关键词文件的数据采集脚本

从 auto_spider_img 文件夹中的关键词文件读取角色信息，采集图像数据
"""

import os
import argparse
import requests
import logging
import time
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO
from pathlib import Path

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
        
        # 请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
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
    
    def _fetch_from_safebooru(self, query, limit=50):
        """
        从Safebooru获取图像
        
        Args:
            query: 搜索查询
            limit: 获取数量
            
        Returns:
            图像URL列表
        """
        base_url = 'https://safebooru.org/index.php'
        images = []
        
        try:
            params = {
                'page': 'dapi',
                's': 'post',
                'q': 'index',
                'tags': query,
                'limit': limit,
                'json': '1'
            }
            
            time.sleep(random.uniform(1.0, 2.0))
            
            response = requests.get(base_url, params=params, headers=self.headers, timeout=20)
            response.raise_for_status()
            
            if not response.text:
                return images
            
            data = response.json()
            
            if isinstance(data, dict) and 'posts' in data:
                data = data['posts']
            
            for post in data:
                if post.get('file_url'):
                    images.append({
                        'url': post['file_url'],
                        'width': post.get('width', 0),
                        'height': post.get('height', 0)
                    })
            
            logger.info(f"从Safebooru获取了 {len(images)} 张图像，查询: {query}")
        except Exception as e:
            logger.error(f"从Safebooru获取图像失败，查询: {query}, 错误: {e}")
        
        return images
    
    def _fetch_from_waifu_pics(self, limit=50):
        """
        从Waifu.pics获取随机图像
        
        Args:
            limit: 获取数量
            
        Returns:
            图像URL列表
        """
        base_url = 'https://api.waifu.pics/sfw/waifu'
        images = []
        
        try:
            for _ in range(limit):
                time.sleep(random.uniform(0.5, 1.0))
                response = requests.get(base_url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if data.get('url'):
                    images.append({
                        'url': data['url'],
                        'width': 0,
                        'height': 0
                    })
            
            logger.info(f"从Waifu.pics获取了 {len(images)} 张图像")
        except Exception as e:
            logger.error(f"从Waifu.pics获取图像失败: {e}")
        
        return images
    
    def _download_image(self, url, save_path):
        """
        下载图像
        
        Args:
            url: 图像URL
            save_path: 保存路径
            
        Returns:
            是否下载成功
        """
        try:
            time.sleep(random.uniform(0.3, 0.8))
            response = requests.get(url, headers=self.headers, timeout=15, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type:
                return False
            
            image = Image.open(BytesIO(response.content))
            
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode == 'P':
                image = image.convert('RGB')
            
            max_size = 1024
            if max(image.width, image.height) > max_size:
                ratio = max_size / max(image.width, image.height)
                new_width = int(image.width * ratio)
                new_height = int(image.height * ratio)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            image.save(save_path, 'JPEG', quality=95)
            
            with Image.open(save_path) as img:
                img.verify()
            
            return True
        except Exception as e:
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
        
        # 从Safebooru获取图像
        all_images = []
        for query in search_queries:
            if len(all_images) >= needed_count:
                break
            
            source_images = self._fetch_from_safebooru(query, limit=needed_count - len(all_images))
            all_images.extend(source_images)
        
        # 如果还不够，尝试其他数据源
        if len(all_images) < needed_count:
            additional_images = self._fetch_from_waifu_pics(limit=needed_count - len(all_images))
            all_images.extend(additional_images)
        
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
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {}
            
            for i, img in enumerate(all_images[:needed_count]):
                save_path = os.path.join(character_dir, f"{series}_{character_name}_{existing_count + i:04d}.jpg")
                future = executor.submit(self._download_image, img['url'], save_path)
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

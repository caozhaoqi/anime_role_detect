#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于系列的角色数据采集脚本

从 auto_spider_img/characters/ 目录中读取角色列表，按系列分类进行数据采集
根据数据采集分配计划，确保每个角色获得足够的数据
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

# 添加当前目录到系统路径
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置工具
from src.utils.config_utils import (
    get_train_dir,
    get_characters_dir,
    get_anime_set_file,
    get_max_images_per_character,
    get_min_images_per_character,
    get_min_image_size,
    get_max_image_size,
    get_min_aspect_ratio,
    get_max_aspect_ratio
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('series_data_collector')


class SeriesBasedDataCollector:
    def __init__(self, output_dir=None, max_workers=5):
        """
        初始化基于系列的数据采集器
        
        Args:
            output_dir: 输出目录，默认为配置中的训练目录
            max_workers: 最大并发数
        """
        self.output_dir = output_dir or get_train_dir()
        self.max_workers = max_workers
        self.min_images = get_min_images_per_character()
        self.ideal_images = get_max_images_per_character()
        
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
        
        # 获取角色目录
        self.characters_dir = get_characters_dir()
        
        # 角色文件映射（基于anime_set.txt）
        self.character_files = {
            'genshin_impact': os.path.join(self.characters_dir, '原神.txt'),
            'honkai_star_rail': os.path.join(self.characters_dir, '崩坏 星穹铁道.txt'),
            'honkai_impact_3': os.path.join(self.characters_dir, '崩坏三.txt'),
            'wuthering_waves': os.path.join(self.characters_dir, '鸣潮.txt'),
            'arknights_endedge': os.path.join(self.characters_dir, '明日方舟 终末地.txt'),
            'tower_of_fantasy': os.path.join(self.characters_dir, '幻塔.txt'),
            'zenless_zone_zero': os.path.join(self.characters_dir, '绝区零.txt'),
            'honkai_academy': os.path.join(self.characters_dir, '崩坏学园.txt'),
            'oshi_no_ko': os.path.join(self.characters_dir, '我推的孩子.txt'),
            'spy_x_family': os.path.join(self.characters_dir, '间谍过家家.txt'),
            'k_on': os.path.join(self.characters_dir, '轻音少女.txt')
        }
        
        # 系列优先级映射（基于anime_set.txt顺序）
        self.priority_mapping = {
            'genshin_impact': 1,
            'honkai_star_rail': 2,
            'honkai_impact_3': 3,
            'wuthering_waves': 4,
            'arknights_endedge': 5,
            'tower_of_fantasy': 6,
            'zenless_zone_zero': 7,
            'honkai_academy': 8,
            'oshi_no_ko': 9,
            'spy_x_family': 10,
            'k_on': 11
        }
    
    def load_characters(self, character_file):
        """
        从角色文件加载角色列表
        
        Args:
            character_file: 角色文件路径
            
        Returns:
            角色列表
        """
        characters = []
        
        try:
            with open(character_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # 过滤版本信息和活动名称
                        if any(keyword in line for keyword in ['版本', '活动', '更新', '时间', '联动', '合作', '×', 'x', '×', '主题', '特别篇', '系列', '年', '月', '日', '参考资料']):
                            continue
                        # 过滤系统/游戏相关词汇
                        if any(keyword in line for keyword in ['游戏类型', '货币', '消耗品', '强化材料', '遗器', '光锥', '任务道具', '其他材料']):
                            continue
                        # 过滤过长或过短的行
                        if len(line) < 2 or len(line) > 30:
                            continue
                        # 过滤纯数字或纯符号行
                        if line.replace('.', '').replace('·', '').replace('•', '').replace('&', '').isalnum():
                            # 过滤纯数字
                            if line.isdigit():
                                continue
                            characters.append(line)
            
            logger.info(f"从 {character_file} 加载了 {len(characters)} 个角色")
        except Exception as e:
            logger.error(f"加载角色文件失败 {character_file}: {e}")
        
        return characters
    
    def load_all_characters(self):
        """
        加载所有角色文件
        
        Returns:
            角色字典 {series: [characters]}
        """
        all_characters = {}
        
        for series, file_path in self.character_files.items():
            if os.path.exists(file_path):
                characters = self.load_characters(file_path)
                if characters:
                    all_characters[series] = characters
            else:
                logger.warning(f"角色文件不存在: {file_path}")
        
        logger.info(f"总共加载了 {sum(len(chars) for chars in all_characters.values())} 个角色")
        return all_characters
    
    def count_existing_images(self, series, character):
        """
        统计角色已有的图像数量
        
        Args:
            series: 系列名称
            character: 角色名称
            
        Returns:
            现有图像数量
        """
        character_dir = os.path.join(self.output_dir, f"{series}_{character}")
        if not os.path.exists(character_dir):
            return 0
        
        image_files = [f for f in os.listdir(character_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        return len(image_files)
    
    def fetch_image_from_safebooru(self, keyword, retries=3):
        """
        从Safebooru获取图像
        
        Args:
            keyword: 搜索关键词
            retries: 重试次数
            
        Returns:
            图像内容或None
        """
        for i in range(retries):
            try:
                search_url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&tags={keyword}&limit=1"
                response = requests.get(search_url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    from xml.etree import ElementTree as ET
                    root = ET.fromstring(response.content)
                    
                    for post in root.findall('post'):
                        image_url = post.get('file_url')
                        if image_url:
                            image_response = requests.get(image_url, headers=self.headers, timeout=10)
                            if image_response.status_code == 200:
                                return image_response.content
            except Exception as e:
                logger.debug(f"Safebooru采集失败 {keyword}: {e}")
            
            time.sleep(random.uniform(0.5, 1.5))
        
        return None
    
    def fetch_image_from_waifu_pics(self, keyword, retries=3):
        """
        从Waifu.pics获取图像
        
        Args:
            keyword: 搜索关键词
            retries: 重试次数
            
        Returns:
            图像内容或None
        """
        for i in range(retries):
            try:
                # Waifu.pics API
                api_url = f"https://api.waifu.pics/sfw/waifu"
                response = requests.get(api_url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    image_url = data.get('url')
                    if image_url:
                        image_response = requests.get(image_url, headers=self.headers, timeout=10)
                        if image_response.status_code == 200:
                            return image_response.content
            except Exception as e:
                logger.debug(f"Waifu.pics采集失败 {keyword}: {e}")
            
            time.sleep(random.uniform(0.5, 1.5))
        
        return None
    
    def fetch_image(self, keyword):
        """
        从多个源获取图像
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            图像内容或None
        """
        # 尝试从Safebooru获取
        image_content = self.fetch_image_from_safebooru(keyword)
        if image_content:
            return image_content
        
        # 尝试从Waifu.pics获取
        image_content = self.fetch_image_from_waifu_pics(keyword)
        if image_content:
            return image_content
        
        return None
    
    def save_image(self, series, character, image_content):
        """
        保存图像到指定目录
        
        Args:
            series: 系列名称
            character: 角色名称
            image_content: 图像内容
            
        Returns:
            保存成功返回True，失败返回False
        """
        character_dir = os.path.join(self.output_dir, f"{series}_{character}")
        os.makedirs(character_dir, exist_ok=True)
        
        # 生成文件名
        existing_count = self.count_existing_images(series, character)
        filename = f"{series}_{character}_{existing_count:04d}.jpg"
        file_path = os.path.join(character_dir, filename)
        
        try:
            # 验证图像
            image = Image.open(BytesIO(image_content))
            
            # 检查图像大小
            min_size = get_min_image_size()
            if image.width < min_size or image.height < min_size:
                logger.debug(f"图像分辨率过低: {filename} ({image.width}x{image.height})")
                return False
            
            # 检查宽高比
            aspect_ratio = image.width / image.height
            min_ratio = get_min_aspect_ratio()
            max_ratio = get_max_aspect_ratio()
            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                logger.debug(f"图像宽高比不符合要求: {filename} ({aspect_ratio:.2f})")
                return False
            
            # 转换RGBA为RGB
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])  # 3是alpha通道
                image = background
            
            # 保存图像
            image.save(file_path, 'JPEG', quality=90)
            logger.debug(f"保存图像成功: {filename}")
            return True
        except Exception as e:
            logger.error(f"保存图像失败 {filename}: {e}")
            return False
    
    def collect_character_data(self, series, character, target_count):
        """
        为单个角色采集数据
        
        Args:
            series: 系列名称
            character: 角色名称
            target_count: 目标图像数量
            
        Returns:
            成功采集的图像数量
        """
        existing_count = self.count_existing_images(series, character)
        needed_count = max(0, target_count - existing_count)
        
        if needed_count <= 0:
            logger.info(f"{series}_{character} 已有足够数据: {existing_count}张")
            return 0
        
        logger.info(f"开始采集 {series}_{character}, 需要 {needed_count} 张图像")
        
        collected_count = 0
        attempts = 0
        max_attempts = needed_count * 5  # 最大尝试次数
        
        while collected_count < needed_count and attempts < max_attempts:
            # 尝试不同的关键词组合
            keywords = [
                character,
                f"{series} {character}",
                character.replace('·', ' '),
                character.replace('•', ' ')
            ]
            
            for keyword in keywords:
                if collected_count >= needed_count:
                    break
                
                image_content = self.fetch_image(keyword)
                if image_content:
                    if self.save_image(series, character, image_content):
                        collected_count += 1
                        # 随机延迟避免被封禁
                        time.sleep(random.uniform(0.5, 1.5))
            
            attempts += 1
            
            # 如果连续失败，增加延迟
            if attempts % 10 == 0:
                logger.info(f"{series}_{character} 采集进度: {collected_count}/{needed_count}")
                time.sleep(random.uniform(2, 3))
        
        logger.info(f"{series}_{character} 采集完成，成功 {collected_count} 张，尝试 {attempts} 次")
        return collected_count
    
    def collect_all_data(self):
        """
        采集所有角色的数据
        """
        all_characters = self.load_all_characters()
        
        # 按优先级排序系列
        sorted_series = sorted(all_characters.keys(), key=lambda x: self.priority_mapping.get(x, 999))
        
        total_collected = 0
        total_needed = 0
        
        for series in sorted_series:
            characters = all_characters[series]
            logger.info(f"开始采集系列: {series} (共 {len(characters)} 个角色)")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for character in characters:
                    existing_count = self.count_existing_images(series, character)
                    needed_count = max(0, self.min_images - existing_count)
                    total_needed += needed_count
                    
                    if needed_count > 0:
                        future = executor.submit(self.collect_character_data, series, character, self.min_images)
                        futures.append(future)
                
                # 等待所有任务完成
                for future in as_completed(futures):
                    try:
                        collected = future.result()
                        total_collected += collected
                    except Exception as e:
                        logger.error(f"采集任务失败: {e}")
            
            logger.info(f"系列 {series} 采集完成")
        
        logger.info(f"所有系列采集完成！")
        logger.info(f"总共需要: {total_needed} 张图像")
        logger.info(f"成功采集: {total_collected} 张图像")
    
    def collect_priority_data(self):
        """
        优先采集数据不足的角色
        """
        all_characters = self.load_all_characters()
        priority_tasks = []
        
        # 收集需要采集的任务
        for series, characters in all_characters.items():
            for character in characters:
                existing_count = self.count_existing_images(series, character)
                if existing_count < self.min_images:
                    priority_tasks.append((series, character, existing_count))
        
        # 按现有数据量排序，数据最少的优先
        priority_tasks.sort(key=lambda x: x[2])
        
        logger.info(f"发现 {len(priority_tasks)} 个需要优先采集的角色")
        
        total_collected = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for series, character, existing_count in priority_tasks:
                future = executor.submit(self.collect_character_data, series, character, self.min_images)
                futures.append(future)
            
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    collected = future.result()
                    total_collected += collected
                except Exception as e:
                    logger.error(f"优先采集任务失败: {e}")
        
        logger.info(f"优先采集完成！成功采集: {total_collected} 张图像")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='基于系列的角色数据采集脚本')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    parser.add_argument('--max-workers', type=int, default=5, help='最大并发数')
    parser.add_argument('--mode', type=str, default='priority', choices=['all', 'priority'], help='采集模式')
    
    args = parser.parse_args()
    
    collector = SeriesBasedDataCollector(
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    if args.mode == 'priority':
        collector.collect_priority_data()
    else:
        collector.collect_all_data()


if __name__ == '__main__':
    main()

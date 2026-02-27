#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色文本描述采集器

从多个来源采集角色文本描述，处理和存储描述数据
"""

import os
import json
import requests
import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from pathlib import Path

# 添加当前目录到系统路径
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置工具
from src.utils.config_utils import (
    get_train_dir,
    get_characters_dir,
    get_anime_set_file
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('text_description_collector')


class TextDescriptionCollector:
    def __init__(self, output_dir=None, max_workers=3):
        """
        初始化文本描述采集器
        
        Args:
            output_dir: 输出目录，默认为配置中的训练目录
            max_workers: 最大并发数
        """
        self.output_dir = output_dir or get_train_dir()
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
        
        # 获取角色目录
        self.characters_dir = get_characters_dir()
        
        # 角色文件映射（基于anime_set.txt）
        self.character_files = {
            'genshin_impact': os.path.join(self.characters_dir, '原神.txt'),
            'honkai_star_rail': os.path.join(self.characters_dir, '崩坏 星穹铁道.txt'),
            'honkai_impact_3': os.path.join(self.characters_dir, '崩坏三.txt'),
            'blue_archive': os.path.join(self.characters_dir, '蔚蓝档案.txt'),
            'dust_white_zone': os.path.join(self.characters_dir, '尘白禁区.txt'),
            'wuthering_waves': os.path.join(self.characters_dir, '鸣潮.txt'),
            'arknights_endedge': os.path.join(self.characters_dir, '明日方舟 终末地.txt'),
            'tower_of_fantasy': os.path.join(self.characters_dir, '幻塔.txt'),
            'zenless_zone_zero': os.path.join(self.characters_dir, '绝区零.txt'),
            'honkai_academy': os.path.join(self.characters_dir, '崩坏学园.txt'),
            'oshi_no_ko': os.path.join(self.characters_dir, '我推的孩子.txt'),
            'spy_x_family': os.path.join(self.characters_dir, '间谍过家家.txt'),
            'k_on': os.path.join(self.characters_dir, '轻音少女.txt'),
            'frieren': os.path.join(self.characters_dir, '葬送的芙莉莲.txt'),
            'bang_dream': os.path.join(self.characters_dir, 'bang dream.txt'),
            'its_my_go': os.path.join(self.characters_dir, 'it\'s my go!!!!!.txt'),
            'ave_mujica': os.path.join(self.characters_dir, 'ave mujica.txt'),
            'kaguya_sama': os.path.join(self.characters_dir, '辉夜大小姐想让我告白.txt'),
            'madoka_magica': os.path.join(self.characters_dir, '魔法少女小圆.txt'),
            'song_doll': os.path.join(self.characters_dir, '颂乐人偶.txt'),
            'lost_children': os.path.join(self.characters_dir, '迷途之子.txt')
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
    
    def fetch_description_from_web(self, series, character, retries=3):
        """
        从网络搜索获取角色描述
        
        Args:
            series: 系列名称
            character: 角色名称
            retries: 重试次数
            
        Returns:
            描述文本列表
        """
        descriptions = []
        
        for i in range(retries):
            try:
                # 构建搜索关键词
                search_query = f"{character} {series} 角色介绍"
                search_url = f"https://www.google.com/search?q={requests.utils.quote(search_query)}"
                
                response = requests.get(search_url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # 提取搜索结果中的文本
                    search_results = soup.find_all('div', class_='BNeawe')
                    for result in search_results:
                        text = result.get_text(strip=True)
                        if text and len(text) > 50 and len(text) < 500:
                            descriptions.append(text)
                            if len(descriptions) >= 3:
                                break
                    
                    if descriptions:
                        break
            except Exception as e:
                logger.debug(f"网络搜索失败 {character}: {e}")
            
            time.sleep(random.uniform(1, 2))
        
        return descriptions
    
    def generate_description(self, series, character):
        """
        基于模板生成角色描述
        
        Args:
            series: 系列名称
            character: 角色名称
            
        Returns:
            生成的描述文本
        """
        # 简单的模板生成
        templates = [
            f"{character}是{series}中的角色，拥有独特的外观和个性。",
            f"作为{series}的重要角色，{character}具有鲜明的性格特点和背景故事。",
            f"{character}在{series}中扮演着重要角色，深受粉丝喜爱。"
        ]
        
        return templates
    
    def collect_descriptions(self, series, character):
        """
        为单个角色采集描述
        
        Args:
            series: 系列名称
            character: 角色名称
            
        Returns:
            描述列表
        """
        descriptions = []
        
        # 从网络搜索获取描述
        web_descriptions = self.fetch_description_from_web(series, character)
        descriptions.extend(web_descriptions)
        
        # 如果网络搜索失败，使用模板生成
        if not descriptions:
            template_descriptions = self.generate_description(series, character)
            descriptions.extend(template_descriptions)
        
        # 处理和过滤描述
        processed_descriptions = []
        for i, desc in enumerate(descriptions):
            # 过滤重复和低质量描述
            if desc and len(desc) > 50:
                processed_descriptions.append({
                    "id": f"desc_{i+1:03d}",
                    "text": desc,
                    "source": "web_search" if i < len(web_descriptions) else "template",
                    "quality": 0.8 if i < len(web_descriptions) else 0.5
                })
        
        return processed_descriptions
    
    def save_descriptions(self, series, character, descriptions):
        """
        保存角色描述到JSON文件
        
        Args:
            series: 系列名称
            character: 角色名称
            descriptions: 描述列表
            
        Returns:
            保存成功返回True，失败返回False
        """
        character_dir = os.path.join(self.output_dir, f"{series}_{character}")
        os.makedirs(character_dir, exist_ok=True)
        
        desc_file_path = os.path.join(character_dir, "descriptions.json")
        
        # 构建描述数据
        desc_data = {
            "character": character,
            "series": series,
            "descriptions": descriptions,
            "image_descriptions": {}
        }
        
        # 获取角色目录中的图像文件
        image_files = [f for f in os.listdir(character_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        
        # 为每个图像关联描述
        for img_file in image_files:
            # 简单的关联策略：为每个图像关联前2个描述
            if descriptions:
                related_desc_ids = [desc["id"] for desc in descriptions[:2]]
                desc_data["image_descriptions"][img_file] = related_desc_ids
        
        try:
            with open(desc_file_path, 'w', encoding='utf-8') as f:
                json.dump(desc_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存描述成功: {series}_{character} ({len(descriptions)}条)")
            return True
        except Exception as e:
            logger.error(f"保存描述失败 {series}_{character}: {e}")
            return False
    
    def collect_character_descriptions(self, series, character):
        """
        为单个角色采集并保存描述
        
        Args:
            series: 系列名称
            character: 角色名称
            
        Returns:
            成功采集的描述数量
        """
        # 检查是否已有描述文件
        character_dir = os.path.join(self.output_dir, f"{series}_{character}")
        desc_file_path = os.path.join(character_dir, "descriptions.json")
        
        if os.path.exists(desc_file_path):
            try:
                with open(desc_file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                existing_count = len(existing_data.get("descriptions", []))
                if existing_count >= 3:
                    logger.info(f"{series}_{character} 已有足够描述: {existing_count}条")
                    return 0
            except Exception as e:
                logger.error(f"读取现有描述失败: {e}")
        
        # 采集描述
        logger.info(f"开始采集 {series}_{character} 的描述")
        descriptions = self.collect_descriptions(series, character)
        
        if descriptions:
            success = self.save_descriptions(series, character, descriptions)
            if success:
                return len(descriptions)
        
        return 0
    
    def collect_all_descriptions(self):
        """
        采集所有角色的描述
        """
        all_characters = self.load_all_characters()
        
        total_collected = 0
        
        for series, characters in all_characters.items():
            logger.info(f"开始采集系列: {series} (共 {len(characters)} 个角色)")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for character in characters:
                    future = executor.submit(self.collect_character_descriptions, series, character)
                    futures.append(future)
                
                # 等待所有任务完成
                for future in as_completed(futures):
                    try:
                        collected = future.result()
                        total_collected += collected
                    except Exception as e:
                        logger.error(f"采集任务失败: {e}")
            
            logger.info(f"系列 {series} 描述采集完成")
        
        logger.info(f"所有系列描述采集完成！")
        logger.info(f"成功采集: {total_collected} 条描述")


def main():
    """
    主函数
    """
    collector = TextDescriptionCollector()
    collector.collect_all_descriptions()


if __name__ == '__main__':
    main()

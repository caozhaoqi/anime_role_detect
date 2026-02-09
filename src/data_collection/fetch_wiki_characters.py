#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从官方wiki网站获取角色列表脚本

从fandom wiki等官方wiki网站获取真实的角色名称
"""

import os
import requests
import logging
from bs4 import BeautifulSoup
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('wiki_characters_fetcher')


class WikiCharactersFetcher:
    def __init__(self):
        """
        初始化wiki角色获取器
        """
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive'
        }
        
        # 官方wiki配置
        self.wiki_config = {
            'genshin_impact': {
                'name': '原神',
                'wiki_url': 'https://genshin-impact.fandom.com/zh/wiki/角色'
            },
            'honkai_star_rail': {
                'name': '崩坏 星穹铁道',
                'wiki_url': 'https://honkai-star-rail.fandom.com/zh/wiki/角色'
            },
            'honkai_impact_3': {
                'name': '崩坏三',
                'wiki_url': 'https://honkaiimpact3.fandom.com/zh/wiki/角色'
            },
            'wuthering_waves': {
                'name': '鸣潮',
                'wiki_url': 'https://wuthering-waves.fandom.com/zh/wiki/角色'
            },
            'arknights_endedge': {
                'name': '明日方舟 终末地',
                'wiki_url': 'https://arknights-endoftheearth.fandom.com/zh/wiki/角色'
            },
            'tower_of_fantasy': {
                'name': '幻塔',
                'wiki_url': 'https://toweroffantasy.fandom.com/zh/wiki/角色'
            },
            'zenless_zone_zero': {
                'name': '绝区零',
                'wiki_url': 'https://zenless-zone-zero.fandom.com/zh/wiki/角色'
            },
            'honkai_academy': {
                'name': '崩坏学园',
                'wiki_url': 'https://honkaiacademia.fandom.com/zh/wiki/角色'
            },
            'oshi_no_ko': {
                'name': '我推的孩子',
                'wiki_url': 'https://oshi-no-ko.fandom.com/zh/wiki/角色'
            },
            'spy_x_family': {
                'name': '间谍过家家',
                'wiki_url': 'https://spy-x-family.fandom.com/zh/wiki/角色'
            },
            'k_on': {
                'name': '轻音少女',
                'wiki_url': 'https://k-on.fandom.com/zh/wiki/角色'
            }
        }
        
        # 输出目录
        self.output_dir = 'auto_spider_img/characters'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fetch_url(self, url, retries=3):
        """
        带重试的请求函数
        
        Args:
            url: URL地址
            retries: 重试次数
            
        Returns:
            响应对象或None
        """
        for i in range(retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=15)
                if response.status_code == 200:
                    return response
                logger.warning(f"请求失败 {url}: {response.status_code}")
            except Exception as e:
                logger.error(f"请求异常 {url}: {str(e)}")
            time.sleep(2)
        return None
    
    def fetch_from_fandom_wiki(self, url):
        """
        从fandom wiki获取角色列表
        
        Args:
            url: fandom wiki角色页面URL
            
        Returns:
            角色列表
        """
        characters = []
        
        response = self.fetch_url(url)
        if not response:
            return characters
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找所有可能的角色名称位置
            
            # 1. 查找表格中的角色名称
            tables = soup.find_all('table', class_='wikitable')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # 跳过表头
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        # 通常第一个单元格是角色名称
                        name_cell = cells[0]
                        # 查找链接中的文本
                        links = name_cell.find_all('a')
                        if links:
                            for link in links:
                                name = link.get_text(strip=True)
                                name = self.clean_character_name(name)
                                if name:
                                    characters.append(name)
                        else:
                            # 如果没有链接，直接获取文本
                            name = name_cell.get_text(strip=True)
                            name = self.clean_character_name(name)
                            if name:
                                characters.append(name)
            
            # 2. 查找分类链接中的角色名称
            category_links = soup.find_all('a', class_='category-page__member-link')
            for link in category_links:
                name = link.get_text(strip=True)
                name = self.clean_character_name(name)
                if name:
                    characters.append(name)
            
            # 3. 查找列表中的角色名称
            lists = soup.find_all(['ul', 'ol'])
            for lst in lists:
                items = lst.find_all('li')
                for item in items:
                    # 查找链接
                    links = item.find_all('a')
                    if links:
                        for link in links:
                            name = link.get_text(strip=True)
                            name = self.clean_character_name(name)
                            if name:
                                characters.append(name)
                    else:
                        # 直接获取文本
                        name = item.get_text(strip=True)
                        name = self.clean_character_name(name)
                        if name:
                            characters.append(name)
            
            # 去重
            characters = list(set(characters))
            logger.info(f"从fandom wiki获取到 {len(characters)} 个角色")
            
        except Exception as e:
            logger.error(f"解析fandom wiki失败: {str(e)}")
        
        return characters
    
    def clean_character_name(self, name):
        """
        清理角色名称
        
        Args:
            name: 原始角色名称
            
        Returns:
            清理后的角色名称或None
        """
        import re
        
        if not name:
            return None
        
        # 移除括号及其内容
        name = re.sub(r'\(.*?\)', '', name)
        name = re.sub(r'（.*?）', '', name)
        name = re.sub(r'\[.*?\]', '', name)
        
        # 移除特殊字符
        name = name.strip()
        
        # 过滤无效名称
        if len(name) < 2 or len(name) > 20:
            return None
        
        # 过滤数字和符号
        if re.match(r'^[\d\W]+$', name):
            return None
        
        # 过滤常见非角色词汇
        exclude_words = [
            '列表', '角色', '人物', '配音', 'CV', '声优', '演员', '介绍', '概览',
            '相关', '制作', '音乐', '首页', '模板', '分类', '编辑', '更多', '查看',
            '收起', '图鉴', '攻略', '角色列表', '人物列表', '登场角色', '主要角色',
            '次要角色', '游戏角色', '动漫角色', '虚拟角色', '角色介绍', '角色设定',
            'Category', '分类', '讨论', '编辑', '历史', '查看源代码', '刷新',
            '分享', '添加到', '更多选项', '贡献', '最近更改', '随机页面', '帮助'
        ]
        if any(word in name for word in exclude_words):
            return None
        
        return name
    
    def fetch_series_characters(self, series_key):
        """
        获取指定系列的角色列表
        
        Args:
            series_key: 系列键
            
        Returns:
            角色列表
        """
        config = self.wiki_config.get(series_key)
        if not config:
            logger.error(f"系列配置不存在: {series_key}")
            return []
        
        logger.info(f"开始从官方wiki获取 {config['name']} 的角色列表")
        
        # 从fandom wiki获取
        characters = self.fetch_from_fandom_wiki(config['wiki_url'])
        
        logger.info(f"成功获取 {config['name']} 的 {len(characters)} 个角色")
        return characters
    
    def save_characters(self, series_key, characters):
        """
        保存角色列表到文件
        
        Args:
            series_key: 系列键
            characters: 角色列表
        """
        config = self.wiki_config.get(series_key)
        if not config:
            return
        
        filename = f"{self.output_dir}/{config['name']}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for character in characters:
                f.write(f"{character}\n")
        
        logger.info(f"保存角色列表成功: {filename} ({len(characters)} 个角色)")
    
    def process_all_series(self):
        """
        处理所有系列
        """
        for series_key, config in self.wiki_config.items():
            logger.info(f"=== 开始处理系列: {config['name']} ===")
            characters = self.fetch_series_characters(series_key)
            
            if characters:
                self.save_characters(series_key, characters)
            else:
                logger.warning(f"未从官方wiki获取到 {config['name']} 的角色列表")
            
            # 避免请求过快
            time.sleep(3)
        
        logger.info("所有系列处理完成！")


def main():
    """
    主函数
    """
    fetcher = WikiCharactersFetcher()
    fetcher.process_all_series()


if __name__ == '__main__':
    main()

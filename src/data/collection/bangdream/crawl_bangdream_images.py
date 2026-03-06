#!/usr/bin/env python3
"""
BangDream角色图片采集脚本
采集mygo、avemujica等系列的角色图片
"""
import os
import time
import random
import argparse
import logging
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('crawl_bangdream_images')

class BangDreamImageCrawler:
    """BangDream角色图片采集器"""
    
    def __init__(self, output_dir, max_images=100):
        """初始化采集器
        
        Args:
            output_dir: 输出目录
            max_images: 每个角色的最大图片数量
        """
        self.output_dir = output_dir
        self.max_images = max_images
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        }
    
    def crawl_images(self, character_name, series):
        """采集指定角色的图片
        
        Args:
            character_name: 角色名称
            series: 系列名称
        """
        # 创建角色目录
        character_dir = os.path.join(self.output_dir, f'bangdream_{character_name}')
        os.makedirs(character_dir, exist_ok=True)
        
        # 构建搜索查询
        search_query = f"bangdream {series} {character_name} 角色图片"
        logger.info(f"开始采集角色: {character_name} (系列: {series})")
        
        # 采集图片
        image_count = 0
        page = 1
        
        while image_count < self.max_images and page <= 5:
            logger.info(f"搜索第 {page} 页")
            
            # 使用Bing图片搜索
            search_url = f"https://www.bing.com/images/search?q={search_query.replace(' ', '+')}&first={((page-1)*35)+1}"
            
            try:
                response = self.session.get(search_url, headers=self.headers, timeout=10)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"搜索失败: {e}")
                page += 1
                continue
            
            # 解析HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 查找图片元素
            image_elements = soup.find_all('img', {'class': 'mimg'})
            
            if not image_elements:
                logger.warning(f"第 {page} 页未找到图片")
                page += 1
                continue
            
            # 下载图片
            for img_element in image_elements:
                if image_count >= self.max_images:
                    break
                
                img_url = img_element.get('src') or img_element.get('data-src')
                if not img_url:
                    continue
                
                # 确保URL完整
                if not img_url.startswith('http'):
                    img_url = f"https://www.bing.com{img_url}"
                
                try:
                    # 下载图片
                    img_response = self.session.get(img_url, headers=self.headers, timeout=10)
                    img_response.raise_for_status()
                    
                    # 验证图片
                    image = Image.open(BytesIO(img_response.content))
                    image.verify()
                    
                    # 保存图片
                    img_path = os.path.join(character_dir, f'bangdream_{character_name}_{image_count}_{int(time.time())}.jpg')
                    with open(img_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    image_count += 1
                    logger.info(f"已下载 {image_count}/{self.max_images} 张图片")
                    
                    # 随机延迟
                    time.sleep(random.uniform(0.5, 2.0))
                    
                except Exception as e:
                    logger.error(f"下载图片失败: {e}")
                    continue
            
            page += 1
        
        logger.info(f"角色 {character_name} 图片采集完成，共下载 {image_count} 张图片")
        return image_count
    
    def crawl_all_characters(self):
        """采集所有指定的BangDream角色图片"""
        # 定义要采集的角色
        characters = [
            # MyGO!!!!!
            {'name': '千早爱音', 'series': 'mygo'},  # 千早爱音 (MyGO!!!!!)
            {'name': '高松灯', 'series': 'mygo'},    # 高松灯 (MyGO!!!!!)
            {'name': '椎名立希', 'series': 'mygo'},  # 椎名立希 (MyGO!!!!!)
            {'name': '长崎素世', 'series': 'mygo'},  # 长崎素世 (MyGO!!!!!) - soyo
            {'name': '玉井祥子', 'series': 'mygo'},  # 玉井祥子 (MyGO!!!!!) - 祥子
            
            # Ave Mujica
            {'name': '瑠夏', 'series': 'avemujica'},  # 瑠夏 (Ave Mujica)
            {'name': '理芽', 'series': 'avemujica'},  # 理芽 (Ave Mujica)
            {'name': '夏芽', 'series': 'avemujica'},  # 夏芽 (Ave Mujica)
            {'name': '桃花', 'series': 'avemujica'},  # 桃花 (Ave Mujica)
            {'name': '梨梨花', 'series': 'avemujica'},# 梨梨花 (Ave Mujica)
            
            # 其他系列
            {'name': '户山香澄', 'series': 'afterglow'},  # 户山香澄 (Poppin'Party)
            {'name': '市谷有咲', 'series': 'afterglow'},  # 市谷有咲 (Poppin'Party)
            {'name': '弦卷心', 'series': 'afterglow'},    # 弦卷心 (Hello, Happy World!)
            {'name': '冰川纱夜', 'series': 'afterglow'},  # 冰川纱夜 (Roselia)
            {'name': '丸山彩', 'series': 'afterglow'},    # 丸山彩 (Pastel*Palettes)
        ]
        
        total_images = 0
        
        for character in characters:
            image_count = self.crawl_images(character['name'], character['series'])
            total_images += image_count
            
            # 每个角色之间的延迟
            time.sleep(random.uniform(2.0, 5.0))
        
        logger.info(f"所有角色图片采集完成，共下载 {total_images} 张图片")
        return total_images

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BangDream角色图片采集脚本')
    parser.add_argument('--output_dir', type=str, default='data/all_characters', help='输出目录')
    parser.add_argument('--max_images', type=int, default=50, help='每个角色的最大图片数量')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info('开始采集BangDream角色图片...')
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"每个角色最大图片数量: {args.max_images}")
    
    # 创建采集器
    crawler = BangDreamImageCrawler(args.output_dir, args.max_images)
    
    # 开始采集
    total_images = crawler.crawl_all_characters()
    
    logger.info(f"采集完成！共下载 {total_images} 张图片")

if __name__ == "__main__":
    main()

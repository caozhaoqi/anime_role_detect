#!/usr/bin/env python3
"""
BangDream角色图片采集脚本
使用Bing图片搜索API采集BangDream角色图片
"""
import os
import requests
import argparse
import logging
from PIL import Image
from io import BytesIO
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('crawl_bangdream_characters')

class BangDreamCrawler:
    """BangDream角色图片采集器"""
    
    def __init__(self, output_dir, num_images=50):
        """初始化采集器
        
        Args:
            output_dir: 输出目录
            num_images: 每角色采集图片数量
        """
        self.output_dir = output_dir
        self.num_images = num_images
        self.api_key = "your_bing_api_key"  # 需要替换为实际的Bing API密钥
        self.search_url = "https://api.bing.microsoft.com/v7.0/images/search"
    
    def crawl_character(self, character_name):
        """采集单个角色的图片
        
        Args:
            character_name: 角色名称
        """
        logger.info(f"开始采集角色: {character_name}")
        
        # 创建角色目录
        character_dir = os.path.join(self.output_dir, f"bangdream_{character_name}")
        os.makedirs(character_dir, exist_ok=True)
        
        # 构建搜索查询
        query = f"BangDream {character_name} 角色 图片"
        logger.info(f"搜索查询: {query}")
        
        # 发送搜索请求
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key
        }
        params = {
            "q": query,
            "count": self.num_images,
            "imageType": "photo",
            "safeSearch": "Strict"
        }
        
        try:
            response = requests.get(self.search_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            search_results = response.json()
            
            # 处理搜索结果
            image_urls = [img['contentUrl'] for img in search_results.get('value', [])]
            logger.info(f"找到 {len(image_urls)} 张图片")
            
            # 下载图片
            for i, url in enumerate(image_urls):
                try:
                    logger.info(f"下载图片 {i+1}/{len(image_urls)}: {url}")
                    img_response = requests.get(url, timeout=10)
                    img_response.raise_for_status()
                    
                    # 打开并保存图片
                    image = Image.open(BytesIO(img_response.content))
                    image_path = os.path.join(character_dir, f"bangdream_{character_name}_{i}_{int(time.time())}.jpg")
                    image.save(image_path, quality=90)
                    logger.info(f"保存图片: {image_path}")
                    
                    # 避免请求过快
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"下载图片失败: {e}")
                    continue
        except Exception as e:
            logger.error(f"搜索失败: {e}")
    
    def crawl_all_characters(self):
        """采集所有BangDream角色的图片"""
        # BangDream主要角色列表
        characters = [
            "户山香澄", "花园多惠", "牛込里美", "山吹沙绫", "市谷有咲",  # Poppin'Party
            "凑友希那", "冰川纱夜", "今井莉莎", "宇田川亚子", "白金燐子",  # Roselia
            "弦卷心", "濑田薰", "北泽育美", "丸山彩", "白鹭千圣",  # Afterglow
            "奥泽美咲", "二叶筑紫", "美竹兰", "青叶摩卡", "上原绯玛丽",  # Pastel*Palettes
            "米歇尔", "松原花音", "樱川惠", "大和麻弥", "若宫伊芙"  # Hello, Happy World!
        ]
        
        logger.info(f"开始采集 {len(characters)} 个BangDream角色的图片")
        
        for character in characters:
            self.crawl_character(character)
            # 避免请求过快
            time.sleep(2)
        
        logger.info("BangDream角色图片采集完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BangDream角色图片采集脚本')
    parser.add_argument('--output_dir', type=str, default='data/all_characters', help='输出目录')
    parser.add_argument('--num_images', type=int, default=50, help='每角色采集图片数量')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建采集器
    crawler = BangDreamCrawler(args.output_dir, args.num_images)
    
    # 采集所有角色
    crawler.crawl_all_characters()

if __name__ == "__main__":
    main()

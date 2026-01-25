#!/usr/bin/env python3
"""
数据采集脚本
从网络上获取更多与现有角色相关的图像数据
"""
import os
import requests
import time
import random
import logging
from urllib.parse import quote
from bs4 import BeautifulSoup
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_collection')


class ImageCollector:
    """图像采集器"""
    
    def __init__(self, output_dir, max_images_per_character=100):
        """初始化
        
        Args:
            output_dir: 输出目录
            max_images_per_character: 每个角色最大采集图像数
        """
        self.output_dir = output_dir
        self.max_images_per_character = max_images_per_character
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        }
    
    def search_images(self, query, num_images=50):
        """
        搜索图像
        
        Args:
            query: 搜索关键词
            num_images: 搜索图像数量
            
        Returns:
            图像URL列表
        """
        image_urls = []
        
        # 使用Bing图像搜索
        try:
            encoded_query = quote(query)
            url = f"https://www.bing.com/images/search?q={encoded_query}&count=50"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取图像URL
            for img in soup.find_all('img', {'class': 'mimg'}):
                if 'src' in img.attrs:
                    img_url = img['src']
                    if img_url.startswith('http'):
                        image_urls.append(img_url)
                elif 'data-src' in img.attrs:
                    img_url = img['data-src']
                    if img_url.startswith('http'):
                        image_urls.append(img_url)
            
            # 如果Bing搜索结果不够，尝试使用Google
            if len(image_urls) < num_images:
                logger.info(f"Bing搜索结果不足，尝试Google搜索: {query}")
                encoded_query = quote(query)
                url = f"https://www.google.com/search?q={encoded_query}&tbm=isch"
                
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for img in soup.find_all('img'):
                    if 'src' in img.attrs:
                        img_url = img['src']
                        if img_url.startswith('http') and not img_url.startswith('data:'):
                            image_urls.append(img_url)
        
        except Exception as e:
            logger.error(f"搜索图像出错: {str(e)}")
        
        # 去重
        image_urls = list(set(image_urls))
        return image_urls[:num_images]
    
    def download_image(self, url, save_path):
        """
        下载图像
        
        Args:
            url: 图像URL
            save_path: 保存路径
            
        Returns:
            是否下载成功
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            # 验证图像是否有效
            from PIL import Image
            img = Image.open(save_path)
            img.verify()
            
            return True
        
        except Exception as e:
            logger.error(f"下载图像出错 {url}: {str(e)}")
            # 删除无效文件
            if os.path.exists(save_path):
                os.remove(save_path)
            return False
    
    def collect_images(self, characters):
        """
        为多个角色采集图像
        
        Args:
            characters: 角色列表
        """
        for character in tqdm(characters, desc="采集角色图像"):
            # 创建角色目录
            char_dir = os.path.join(self.output_dir, character)
            os.makedirs(char_dir, exist_ok=True)
            
            # 计算现有图像数量
            existing_images = len([f for f in os.listdir(char_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            # 需要采集的图像数量
            need_images = max(0, self.max_images_per_character - existing_images)
            
            if need_images <= 0:
                logger.info(f"角色 {character} 已有足够图像 ({existing_images} 张)")
                continue
            
            logger.info(f"开始采集角色 {character} 的图像，需要 {need_images} 张")
            
            # 生成搜索关键词
            search_queries = [
                character,
                f"{character} 动漫",
                f"{character} 二次元",
                f"{character} 同人",
                f"{character} 插画"
            ]
            
            collected = 0
            for query in search_queries:
                if collected >= need_images:
                    break
                
                logger.info(f"搜索关键词: {query}")
                
                # 搜索图像
                image_urls = self.search_images(query, num_images=need_images - collected)
                
                # 下载图像
                for img_url in image_urls:
                    if collected >= need_images:
                        break
                    
                    # 生成文件名
                    img_name = f"{character}_{collected + existing_images}_{int(time.time())}.jpg"
                    save_path = os.path.join(char_dir, img_name)
                    
                    # 下载图像
                    if self.download_image(img_url, save_path):
                        collected += 1
                        logger.info(f"成功下载 {character} 的图像 {collected}/{need_images}")
                        
                        # 随机延迟，避免被封禁
                        time.sleep(random.uniform(0.5, 2.0))
            
            logger.info(f"角色 {character} 采集完成，新增 {collected} 张图像，总计 {existing_images + collected} 张")


def main():
    """主函数"""
    # 输出目录
    output_dir = 'data/all_characters'
    
    # 从现有目录获取角色列表
    characters = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    logger.info(f"发现 {len(characters)} 个角色")
    logger.info(f"角色列表: {characters}")
    
    # 创建采集器
    collector = ImageCollector(
        output_dir=output_dir,
        max_images_per_character=100  # 每个角色采集100张图像
    )
    
    # 开始采集
    logger.info('开始采集图像数据...')
    collector.collect_images(characters)
    logger.info('图像数据采集完成！')


if __name__ == "__main__":
    main()

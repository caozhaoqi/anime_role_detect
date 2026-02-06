#!/usr/bin/env python3
"""
扩充训练数据，添加新角色样本
"""
import os
import sys
import json
import argparse
import logging
import time
import random
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('augment_training_data')

class DataAugmenter:
    """数据扩充器"""
    
    def __init__(self, output_dir='data/train', max_images_per_character=50):
        """初始化扩充器
        
        Args:
            output_dir: 训练数据输出目录
            max_images_per_character: 每个角色的最大图片数量
        """
        self.output_dir = output_dir
        self.max_images_per_character = max_images_per_character
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        }
    
    def _init_session(self):
        """初始化会话"""
        try:
            import requests
            self.session = requests.Session()
            logger.info("HTTP会话初始化成功")
        except ImportError:
            logger.error("requests库未安装，请运行: pip install requests")
            raise
    
    def augment_character(self, character_name, series_name):
        """扩充指定角色的数据
        
        Args:
            character_name: 角色名称
            series_name: 系列名称
            
        Returns:
            int: 扩充的图片数量
        """
        if not self.session:
            self._init_session()
        
        # 创建角色目录
        character_dir = os.path.join(self.output_dir, f"{series_name}_{character_name}")
        os.makedirs(character_dir, exist_ok=True)
        
        # 检查已有图片数量
        existing_images = [f for f in os.listdir(character_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        existing_count = len(existing_images)
        
        if existing_count >= self.max_images_per_character:
            logger.info(f"角色 {character_name} 已有足够的训练样本 ({existing_count} 张)")
            return 0
        
        # 需要下载的图片数量
        need_download = self.max_images_per_character - existing_count
        logger.info(f"开始扩充角色: {character_name} (系列: {series_name})，需要下载 {need_download} 张图片")
        
        # 构建搜索查询
        search_query = f"{series_name} {character_name} 角色图片 高清"
        
        image_count = 0
        page = 1
        
        while image_count < need_download and page <= 10:
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
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                image_elements = soup.find_all('img', {'class': 'mimg'})
            except ImportError:
                logger.error("bs4库未安装，请运行: pip install beautifulsoup4")
                raise
            
            if not image_elements:
                logger.warning(f"第 {page} 页未找到图片")
                page += 1
                continue
            
            # 下载图片
            for img_element in image_elements:
                if image_count >= need_download:
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
                    from PIL import Image
                    from io import BytesIO
                    image = Image.open(BytesIO(img_response.content))
                    image.verify()
                    
                    # 保存图片
                    img_filename = f"{series_name}_{character_name}_{existing_count + image_count:04d}.jpg"
                    img_path = os.path.join(character_dir, img_filename)
                    with open(img_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    image_count += 1
                    logger.info(f"已下载 {existing_count + image_count:04d}/{self.max_images_per_character:04d} 张图片")
                    
                    # 随机延迟，避免被封禁
                    time.sleep(random.uniform(1.0, 2.0))
                    
                except Exception as e:
                    logger.error(f"下载图片失败: {e}")
                    continue
            
            page += 1
            time.sleep(random.uniform(2.0, 3.0))
        
        logger.info(f"角色 {character_name} 数据扩充完成，新增 {image_count} 张图片")
        return image_count

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='扩充训练数据，添加新角色样本')
    
    parser.add_argument('--characters', type=str, required=True,
                       help='角色列表JSON字符串，格式: [{"name": "角色名", "series": "系列名"}]')
    parser.add_argument('--output_dir', type=str, default='data/train',
                       help='训练数据输出目录')
    parser.add_argument('--max_images', type=int, default=50,
                       help='每个角色的最大图片数量')
    
    args = parser.parse_args()
    
    # 解析角色列表
    try:
        characters = json.loads(args.characters)
    except json.JSONDecodeError as e:
        logger.error(f"角色列表JSON格式错误: {e}")
        sys.exit(1)
    
    # 验证角色格式
    for i, char in enumerate(characters):
        if not isinstance(char, dict) or 'name' not in char or 'series' not in char:
            logger.error(f"角色 {i+1} 格式错误，需要包含 name 和 series 字段")
            sys.exit(1)
    
    logger.info(f"准备扩充 {len(characters)} 个角色的数据")
    
    # 创建扩充器
    augmenter = DataAugmenter(
        output_dir=args.output_dir,
        max_images_per_character=args.max_images
    )
    
    total_images = 0
    for character in characters:
        image_count = augmenter.augment_character(
            character['name'],
            character['series']
        )
        total_images += image_count
        time.sleep(random.uniform(3.0, 5.0))
    
    logger.info(f"数据扩充完成，共新增 {total_images} 张图片")
    
    # 打印摘要
    print("\n" + "="*60)
    print("训练数据扩充完成")
    print("="*60)
    print(f"扩充角色数: {len(characters)}")
    print(f"新增图片数: {total_images}")
    print("="*60)

if __name__ == '__main__':
    main()

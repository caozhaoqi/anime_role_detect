#!/usr/bin/env python3
"""
二次元插画采集脚本
使用专门的二次元图片数据源进行采集
"""
import os
import sys
import time
import json
import requests
import random
import hashlib
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('collect_anime_images.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnimeImageSource:
    """二次元图片数据源基类"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def search_images(self, keyword, count=50):
        """搜索图片，返回URL列表"""
        raise NotImplementedError
    
    def download_image(self, url):
        """下载图片"""
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                return response.content
        except Exception as e:
            logger.error(f"下载图片失败 {url}: {e}")
        return None

class DanbooruSource(AnimeImageSource):
    """Danbooru数据源"""
    
    def search_images(self, keyword, count=50):
        urls = []
        try:
            page = 1
            while len(urls) < count:
                url = f"https://danbooru.donmai.us/posts.json?tags={keyword}&page={page}&limit=100"
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                    for post in data:
                        if 'file_url' in post:
                            urls.append(post['file_url'])
                    page += 1
                else:
                    break
        except Exception as e:
            logger.error(f"Danbooru搜索失败 {keyword}: {e}")
        return urls[:count]

class ZerochanSource(AnimeImageSource):
    """Zerochan数据源"""
    
    def search_images(self, keyword, count=50):
        urls = []
        try:
            page = 1
            while len(urls) < count:
                url = f"https://www.zerochan.net/{keyword}?p={page}"
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    content = response.text
                    # 提取图片URL
                    import re
                    pattern = r'<a href="(/full/\d+)".*?<img src="(//.+?)"'
                    matches = re.findall(pattern, content)
                    for full_link, thumb_link in matches:
                        full_url = f"https://www.zerochan.net{full_link}"
                        urls.append(full_url)
                    page += 1
                    if len(matches) == 0:
                        break
                else:
                    break
        except Exception as e:
            logger.error(f"Zerochan搜索失败 {keyword}: {e}")
        return urls[:count]

class KonachanSource(AnimeImageSource):
    """Konachan数据源"""
    
    def search_images(self, keyword, count=50):
        urls = []
        try:
            page = 1
            while len(urls) < count:
                url = f"https://konachan.net/post.json?tags={keyword}&page={page}&limit=100"
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                    for post in data:
                        if 'file_url' in post:
                            urls.append(post['file_url'])
                    page += 1
                else:
                    break
        except Exception as e:
            logger.error(f"Konachan搜索失败 {keyword}: {e}")
        return urls[:count]

class YandereSource(AnimeImageSource):
    """Yandere数据源"""
    
    def search_images(self, keyword, count=50):
        urls = []
        try:
            page = 1
            while len(urls) < count:
                url = f"https://yande.re/post.json?tags={keyword}&page={page}&limit=100"
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                    for post in data:
                        if 'file_url' in post:
                            urls.append(post['file_url'])
                    page += 1
                else:
                    break
        except Exception as e:
            logger.error(f"Yandere搜索失败 {keyword}: {e}")
        return urls[:count]

class AnimeImageCollector:
    """二次元图片采集器"""
    
    def __init__(self, output_dir='./anime_images', min_resolution=(512, 512)):
        self.output_dir = output_dir
        self.min_resolution = min_resolution
        self.sources = [
            DanbooruSource(),
            ZerochanSource(),
            KonachanSource(),
            YandereSource()
        ]
        os.makedirs(output_dir, exist_ok=True)
    
    def is_valid_image(self, image_content):
        """检查图片是否有效"""
        try:
            img = Image.open(BytesIO(image_content))
            width, height = img.size
            if width >= self.min_resolution[0] and height >= self.min_resolution[1]:
                return True, width, height
            return False, width, height
        except Exception as e:
            logger.error(f"检查图片失败: {e}")
            return False, 0, 0
    
    def download_and_save(self, url, save_path):
        """下载并保存图片"""
        content = None
        for source in self.sources:
            content = source.download_image(url)
            if content:
                break
        
        if not content:
            return False, "下载失败"
        
        is_valid, width, height = self.is_valid_image(content)
        if not is_valid:
            return False, f"分辨率过低 {width}x{height}"
        
        try:
            with open(save_path, 'wb') as f:
                f.write(content)
            return True, f"成功 {width}x{height}"
        except Exception as e:
            return False, f"保存失败: {e}"
    
    def collect_for_role(self, role_name, target_count=80):
        """为单个角色采集图片"""
        role_dir = os.path.join(self.output_dir, role_name)
        os.makedirs(role_dir, exist_ok=True)
        
        # 检查已有图片数量
        existing = [f for f in os.listdir(role_dir) if f.endswith('.jpg')]
        if len(existing) >= target_count:
            logger.info(f"角色 {role_name} 已有 {len(existing)} 张图片，跳过")
            return {'role': role_name, 'success': 0, 'failed': 0, 'existing': len(existing)}
        
        need_count = target_count - len(existing)
        logger.info(f"角色 {role_name} 需要采集 {need_count} 张图片")
        
        # 从所有数据源搜索
        all_urls = []
        for source in self.sources:
            urls = source.search_images(role_name, count=need_count * 2)
            all_urls.extend(urls)
            time.sleep(1)  # 避免请求过快
        
        # 去重并打乱顺序
        unique_urls = list(set(all_urls))
        random.shuffle(unique_urls)
        
        success_count = 0
        failed_count = 0
        
        for url in unique_urls[:need_count * 2]:
            if success_count >= need_count:
                break
            
            # 生成文件名
            file_hash = hashlib.md5(url.encode()).hexdigest()
            save_path = os.path.join(role_dir, f"{file_hash}.jpg")
            
            if os.path.exists(save_path):
                continue
            
            success, msg = self.download_and_save(url, save_path)
            if success:
                success_count += 1
                logger.debug(f"下载成功: {role_name} - {msg}")
            else:
                failed_count += 1
                logger.debug(f"下载失败: {role_name} - {msg}")
            
            time.sleep(0.5)  # 控制下载速度
        
        logger.info(f"角色 {role_name} 采集完成: 成功 {success_count}, 失败 {failed_count}")
        return {
            'role': role_name,
            'success': success_count,
            'failed': failed_count,
            'existing': len(existing)
        }
    
    def collect_all(self, role_list, target_count=80, max_workers=3):
        """采集所有角色"""
        logger.info(f"开始采集二次元图片，共 {len(role_list)} 个角色")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.collect_for_role, role, target_count) 
                      for role in role_list]
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"处理失败: {e}")
        
        # 统计结果
        total_success = sum(r['success'] for r in results)
        total_failed = sum(r['failed'] for r in results)
        
        logger.info("\n=== 采集完成 ===")
        logger.info(f"总角色数: {len(results)}")
        logger.info(f"成功下载: {total_success}")
        logger.info(f"下载失败: {total_failed}")
        
        return {
            'total_roles': len(results),
            'total_success': total_success,
            'total_failed': total_failed,
            'details': results
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='二次元图片采集脚本')
    parser.add_argument('--role_file', type=str, required=True, help='角色列表文件')
    parser.add_argument('--output_dir', type=str, default='./anime_images', help='输出目录')
    parser.add_argument('--target_count', type=int, default=80, help='每个角色目标图片数')
    parser.add_argument('--max_workers', type=int, default=3, help='最大并发数')
    
    args = parser.parse_args()
    
    # 读取角色列表
    with open(args.role_file, 'r', encoding='utf-8') as f:
        role_list = [line.strip() for line in f if line.strip()]
    
    logger.info(f"读取到 {len(role_list)} 个角色")
    
    # 创建采集器
    collector = AnimeImageCollector(output_dir=args.output_dir)
    
    # 开始采集
    result = collector.collect_all(role_list, args.target_count, args.max_workers)
    
    # 保存结果
    with open('anime_collection_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("\n采集完成！结果已保存到 anime_collection_result.json")

if __name__ == "__main__":
    main()

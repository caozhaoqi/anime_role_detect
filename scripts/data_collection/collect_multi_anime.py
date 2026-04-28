#!/usr/bin/env python3
"""
多源二次元图片采集脚本
尝试多个不同的数据源
"""
import os
import sys
import time
import json
import requests
import hashlib
import random
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('collect_multi_anime.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnimeImageCollector:
    """多源二次元图片采集器"""
    
    def __init__(self, output_dir='./anime_images'):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        os.makedirs(output_dir, exist_ok=True)
    
    def try_waifu_pics(self, categories=None):
        """尝试waifu.pics API"""
        if categories is None:
            categories = ['waifu', 'neko', 'shinobu', 'megumin', 'bully', 'cuddle', 'cry', 'hug', 'awoo', 'handhold']
        
        for category in categories:
            try:
                url = f"https://api.waifu.pics/sfw/{category}"
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if 'url' in data:
                            img_url = data['url']
                            img_response = self.session.get(img_url, timeout=10)
                            if img_response.status_code == 200 and len(img_response.content) > 1000:
                                return img_response.content
                    except:
                        pass
            except:
                pass
            time.sleep(0.3)
        return None
    
    def try_lolibooru(self, tags):
        """尝试Lolibooru"""
        try:
            url = f"https://lolibooru.moe/post.json?tags={requests.utils.quote(tags)}&limit=10"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data:
                    post = random.choice(data)
                    if 'file_url' in post:
                        img_response = self.session.get(post['file_url'], timeout=15)
                        if img_response.status_code == 200 and len(img_response.content) > 1000:
                            return img_response.content
        except Exception as e:
            logger.debug(f"Lolibooru失败: {e}")
        return None
    
    def try_gelbooru(self, tags):
        """尝试Gelbooru"""
        try:
            url = f"https://gelbooru.com/index.php?page=dapi&s=post&q=index&json=1&tags={requests.utils.quote(tags)}&limit=10"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data:
                    post = random.choice(data)
                    if 'file_url' in post:
                        img_response = self.session.get(post['file_url'], timeout=15)
                        if img_response.status_code == 200 and len(img_response.content) > 1000:
                            return img_response.content
        except Exception as e:
            logger.debug(f"Gelbooru失败: {e}")
        return None
    
    def try_rule34(self, tags):
        """尝试Rule34"""
        try:
            url = f"https://rule34.xxx/index.php?page=dapi&s=post&q=index&json=1&tags={requests.utils.quote(tags)}&limit=10"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data:
                    post = random.choice(data)
                    if 'file_url' in post:
                        img_response = self.session.get(post['file_url'], timeout=15)
                        if img_response.status_code == 200 and len(img_response.content) > 1000:
                            return img_response.content
        except Exception as e:
            logger.debug(f"Rule34失败: {e}")
        return None
    
    def try_hypnohub(self, tags):
        """尝试HypnoHub"""
        try:
            url = f"https://hypnohub.net/post/index.json?tags={requests.utils.quote(tags)}&limit=10"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data:
                    post = random.choice(data)
                    if 'file_url' in post:
                        img_response = self.session.get(post['file_url'], timeout=15)
                        if img_response.status_code == 200 and len(img_response.content) > 1000:
                            return img_response.content
        except Exception as e:
            logger.debug(f"Hypnohub失败: {e}")
        return None
    
    def try_tbib(self, tags):
        """尝试Tbib"""
        try:
            url = f"https://tbib.org/index.php?page=dapi&s=post&q=index&json=1&tags={requests.utils.quote(tags)}&limit=10"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data:
                    post = random.choice(data)
                    if 'file_url' in post:
                        img_response = self.session.get(post['file_url'], timeout=15)
                        if img_response.status_code == 200 and len(img_response.content) > 1000:
                            return img_response.content
        except Exception as e:
            logger.debug(f"Tbib失败: {e}")
        return None
    
    def try_xbooru(self, tags):
        """尝试Xbooru"""
        try:
            url = f"https://xbooru.com/index.php?page=dapi&s=post&q=index&json=1&tags={requests.utils.quote(tags)}&limit=10"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data:
                    post = random.choice(data)
                    if 'file_url' in post:
                        img_response = self.session.get(post['file_url'], timeout=15)
                        if img_response.status_code == 200 and len(img_response.content) > 1000:
                            return img_response.content
        except Exception as e:
            logger.debug(f"Xbooru失败: {e}")
        return None
    
    def try_all_sources(self, role_name):
        """尝试所有数据源"""
        # 提取角色关键词
        keywords = role_name.split()[0]  # 取第一个词作为关键词
        
        sources = [
            lambda: self.try_waifu_pics(),
            lambda: self.try_lolibooru(keywords),
            lambda: self.try_gelbooru(keywords),
            lambda: self.try_hypnohub(keywords),
            lambda: self.try_tbib(keywords),
            lambda: self.try_xbooru(keywords),
        ]
        
        for source in sources:
            content = source()
            if content and len(content) > 1000:
                return content
            time.sleep(0.5)
        
        return None
    
    def is_valid_image(self, content):
        """检查图片是否有效"""
        try:
            img = Image.open(BytesIO(content))
            width, height = img.size
            if width >= 512 and height >= 512:
                return True, width, height
            return False, width, height
        except:
            return False, 0, 0
    
    def collect_for_role(self, role_name, target_count=30):
        """为角色采集图片"""
        role_dir = os.path.join(self.output_dir, role_name)
        os.makedirs(role_dir, exist_ok=True)
        
        # 检查已有图片
        existing = [f for f in os.listdir(role_dir) if f.endswith('.jpg')]
        existing_count = len(existing)
        
        if existing_count >= target_count:
            logger.info(f"角色 {role_name} 已有 {existing_count} 张图片，跳过")
            return {'role': role_name, 'success': 0, 'failed': 0, 'existing': existing_count}
        
        need_count = target_count - existing_count
        logger.info(f"角色 {role_name} 需要采集 {need_count} 张图片")
        
        success_count = 0
        failed_count = 0
        
        for i in range(need_count * 3):
            if success_count >= need_count:
                break
            
            content = self.try_all_sources(role_name)
            
            if not content:
                failed_count += 1
                continue
            
            is_valid, width, height = self.is_valid_image(content)
            if not is_valid:
                failed_count += 1
                continue
            
            # 保存图片
            file_hash = hashlib.md5(content).hexdigest()
            save_path = os.path.join(role_dir, f"{file_hash}.jpg")
            
            if os.path.exists(save_path):
                continue
            
            try:
                with open(save_path, 'wb') as f:
                    f.write(content)
                success_count += 1
                logger.debug(f"成功: {role_name} - {width}x{height}")
            except Exception as e:
                failed_count += 1
                logger.debug(f"保存失败: {e}")
            
            time.sleep(0.5)
        
        logger.info(f"角色 {role_name} 采集完成: 成功 {success_count}, 失败 {failed_count}")
        return {
            'role': role_name,
            'success': success_count,
            'failed': failed_count,
            'existing': existing_count
        }
    
    def collect_all(self, role_list, target_count=30):
        """采集所有角色"""
        logger.info(f"开始采集，共 {len(role_list)} 个角色")
        
        results = []
        for role in role_list:
            result = self.collect_for_role(role, target_count)
            results.append(result)
        
        total_success = sum(r['success'] for r in results)
        total_failed = sum(r['failed'] for r in results)
        
        logger.info(f"\n=== 采集完成 ===")
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
    
    parser = argparse.ArgumentParser(description='多源二次元图片采集')
    parser.add_argument('--role_file', type=str, required=True, help='角色列表文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--target_count', type=int, default=30, help='目标数量')
    
    args = parser.parse_args()
    
    with open(args.role_file, 'r', encoding='utf-8') as f:
        role_list = [line.strip() for line in f if line.strip()]
    
    logger.info(f"读取到 {len(role_list)} 个角色")
    
    collector = AnimeImageCollector(output_dir=args.output_dir)
    result = collector.collect_all(role_list, args.target_count)
    
    with open('multi_anime_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("\n采集完成！")

if __name__ == "__main__":
    main()

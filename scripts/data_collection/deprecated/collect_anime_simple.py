#!/usr/bin/env python3
"""
简化版二次元图片采集脚本
使用可靠的公开API进行采集
"""
import os
import sys
import time
import json
import requests
import hashlib
from PIL import Image
from io import BytesIO
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('collect_anime_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnimeImageCollector:
    """简化版二次元图片采集器"""
    
    def __init__(self, output_dir='./anime_images'):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        os.makedirs(output_dir, exist_ok=True)
    
    def download_waifu_pics(self, category='waifu'):
        """从waifu.pics下载图片"""
        try:
            url = f"https://api.waifu.pics/sfw/{category}"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if 'url' in data:
                    img_url = data['url']
                    # 下载图片
                    img_response = self.session.get(img_url, timeout=15)
                    if img_response.status_code == 200:
                        return img_response.content, img_url
        except Exception as e:
            logger.debug(f"waifu.pics失败: {e}")
        return None, None
    
    def download_anime_pictures(self):
        """从anime-pictures.net下载图片"""
        try:
            # 获取随机图片
            url = "https://anime-pictures.net/api/v3/images/random"
            params = {
                'lang': 'en',
                'type': 'jpg'
            }
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if 'images' in data and len(data['images']) > 0:
                    img_info = data['images'][0]
                    if 'url' in img_info:
                        img_url = img_info['url']
                        img_response = self.session.get(img_url, timeout=15)
                        if img_response.status_code == 200:
                            return img_response.content, img_url
        except Exception as e:
            logger.debug(f"anime-pictures.net失败: {e}")
        return None, None
    
    def download_random_anime(self):
        """尝试多个数据源下载图片"""
        sources = [
            lambda: self.download_waifu_pics('waifu'),
            lambda: self.download_waifu_pics('neko'),
            lambda: self.download_waifu_pics('shinobu'),
            lambda: self.download_waifu_pics('megumin'),
            lambda: self.download_anime_pictures()
        ]
        
        for source in sources:
            content, url = source()
            if content:
                return content, url
            time.sleep(0.5)
        
        return None, None
    
    def is_valid_image(self, content):
        """检查图片是否有效"""
        try:
            img = Image.open(BytesIO(content))
            width, height = img.size
            # 检查分辨率
            if width >= 512 and height >= 512:
                return True, width, height
            return False, width, height
        except Exception as e:
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
        downloaded_urls = set()
        
        # 尝试下载
        for i in range(need_count * 3):  # 尝试3倍数量
            if success_count >= need_count:
                break
            
            content, url = self.download_random_anime()
            
            if not content or url in downloaded_urls:
                failed_count += 1
                continue
            
            downloaded_urls.add(url)
            
            is_valid, width, height = self.is_valid_image(content)
            if not is_valid:
                failed_count += 1
                continue
            
            # 保存图片
            file_hash = hashlib.md5(url.encode()).hexdigest()
            save_path = os.path.join(role_dir, f"{file_hash}.jpg")
            
            try:
                with open(save_path, 'wb') as f:
                    f.write(content)
                success_count += 1
                logger.debug(f"成功: {role_name} - {width}x{height}")
            except Exception as e:
                failed_count += 1
                logger.debug(f"保存失败: {e}")
            
            time.sleep(1)  # 控制速度
        
        logger.info(f"角色 {role_name} 采集完成: 成功 {success_count}, 失败 {failed_count}")
        return {
            'role': role_name,
            'success': success_count,
            'failed': failed_count,
            'existing': existing_count
        }
    
    def collect_all(self, role_list, target_count=30):
        """采集所有角色"""
        logger.info(f"开始采集二次元图片，共 {len(role_list)} 个角色")
        
        results = []
        for role in role_list:
            result = self.collect_for_role(role, target_count)
            results.append(result)
        
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
    
    parser = argparse.ArgumentParser(description='简化版二次元图片采集脚本')
    parser.add_argument('--role_file', type=str, required=True, help='角色列表文件')
    parser.add_argument('--output_dir', type=str, default='./anime_images', help='输出目录')
    parser.add_argument('--target_count', type=int, default=30, help='每个角色目标图片数')
    
    args = parser.parse_args()
    
    # 读取角色列表
    with open(args.role_file, 'r', encoding='utf-8') as f:
        role_list = [line.strip() for line in f if line.strip()]
    
    logger.info(f"读取到 {len(role_list)} 个角色")
    
    # 创建采集器
    collector = AnimeImageCollector(output_dir=args.output_dir)
    
    # 开始采集
    result = collector.collect_all(role_list, args.target_count)
    
    # 保存结果
    with open('anime_collection_simple_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("\n采集完成！结果已保存到 anime_collection_simple_result.json")

if __name__ == "__main__":
    main()

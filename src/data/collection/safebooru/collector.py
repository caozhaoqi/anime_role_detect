#!/usr/bin/env python3
"""
Safebooru数据采集器
从Safebooru采集二次元角色图片
"""
import os
import requests
import argparse
import time
import random
from tqdm import tqdm

def download_image(url, save_path):
    """下载图片"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://safebooru.org/',
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"下载失败 {url}: {e}")
        return False

def collect_from_safebooru(tag, output_dir, limit=20):
    """从Safebooru采集单个标签的图片"""
    print(f"\n采集标签: {tag}")
    
    import urllib.parse
    encoded_tag = urllib.parse.quote(tag)
    api_url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1&limit={limit*2}&tags={encoded_tag}"
    
    try:
        response = requests.get(api_url, timeout=15)
        response.raise_for_status()
        posts = response.json()
        
        print(f"找到 {len(posts)} 个结果")
        
        downloaded = 0
        for i, post in enumerate(posts[:limit]):
            if 'file_url' in post:
                url = post['file_url']
                if not url.startswith('http'):
                    url = "https://safebooru.org/images/" + post['directory'] + "/" + post['image']
                
                file_ext = os.path.splitext(url)[1]
                if not file_ext:
                    file_ext = '.jpg'
                
                save_path = os.path.join(output_dir, f"{tag}_{i+1}{file_ext}")
                
                print(f"下载 {i+1}/{limit}: {url}")
                if download_image(url, save_path):
                    downloaded += 1
                    time.sleep(random.uniform(0.5, 1.5))
        
        print(f"成功下载 {downloaded} 张图片")
        return downloaded
    except Exception as e:
        print(f"采集失败: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Safebooru数据采集器')
    parser.add_argument('--tag', required=True, help='要采集的标签')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--limit', type=int, default=20, help='采集图片数量')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    collect_from_safebooru(args.tag, args.output_dir, args.limit)

if __name__ == '__main__':
    main()

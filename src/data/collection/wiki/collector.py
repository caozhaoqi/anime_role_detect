#!/usr/bin/env python3
"""
Wiki数据采集器
从Wiki采集角色信息和图片
"""
import os
import requests
import argparse
import json
from bs4 import BeautifulSoup

def download_image(url, save_path):
    """下载图片"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"下载失败 {url}: {e}")
        return False

def collect_from_wiki(wiki_url, output_dir, limit=20):
    """从Wiki采集角色图片"""
    print(f"从Wiki采集: {wiki_url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        
        response = requests.get(wiki_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        image_urls = []
        for img in soup.find_all('img'):
            src = img.get('src')
            if src and not src.startswith('data:'):
                if src.startswith('//'):
                    src = 'https:' + src
                elif not src.startswith('http'):
                    continue
                
                if any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                    image_urls.append(src)
        
        print(f"找到 {len(image_urls)} 张图片")
        
        downloaded = 0
        for i, image_url in enumerate(image_urls[:limit]):
            file_ext = os.path.splitext(image_url)[1]
            if not file_ext:
                file_ext = '.jpg'
            
            save_path = os.path.join(output_dir, f"wiki_{i+1}{file_ext}")
            
            print(f"下载 {i+1}/{limit}: {image_url}")
            if download_image(image_url, save_path):
                downloaded += 1
        
        print(f"成功下载 {downloaded} 张图片")
        return downloaded
    except Exception as e:
        print(f"Wiki采集失败: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Wiki数据采集器')
    parser.add_argument('--wiki_url', required=True, help='Wiki页面URL')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--limit', type=int, default=20, help='采集图片数量')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    collect_from_wiki(args.wiki_url, args.output_dir, args.limit)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
B站数据采集器
从B站采集动漫视频和图片
"""
import os
import requests
import argparse
import json
from tqdm import tqdm

def download_image(url, save_path):
    """下载图片"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.bilibili.com/',
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"下载失败 {url}: {e}")
        return False

def collect_from_bilibili(keyword, output_dir, limit=20):
    """从B站搜索并下载封面图"""
    print(f"从B站搜索: {keyword}")
    
    try:
        import urllib.parse
        encoded_keyword = urllib.parse.quote(keyword)
        api_url = f"https://api.bilibili.com/x/web-interface/search/all?keyword={encoded_keyword}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        
        response = requests.get(api_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('code') != 0:
            print(f"B站API返回错误: {data.get('message')}")
            return 0
        
        image_urls = []
        results = data.get('data', {}).get('result', [])
        
        for result_type in results:
            if result_type == 'video':
                videos = results[result_type]
                for video in videos[:limit]:
                    pic_url = video.get('pic')
                    if pic_url:
                        image_urls.append(pic_url)
        
        print(f"找到 {len(image_urls)} 个封面图")
        
        downloaded = 0
        for i, image_url in enumerate(image_urls[:limit]):
            file_ext = os.path.splitext(image_url)[1]
            if not file_ext:
                file_ext = ".jpg"
            
            clean_keyword = keyword.replace(" ", "_")
            save_path = os.path.join(output_dir, f"bilibili_{clean_keyword}_{i+1}{file_ext}")
            
            print(f"下载 {i+1}/{limit}: {image_url}")
            if download_image(image_url, save_path):
                downloaded += 1
        
        print(f"成功下载 {downloaded} 张图片")
        return downloaded
    except Exception as e:
        print(f"B站采集失败: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='B站数据采集器')
    parser.add_argument('--keyword', required=True, help='搜索关键词')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--limit', type=int, default=20, help='采集图片数量')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    collect_from_bilibili(args.keyword, args.output_dir, args.limit)

if __name__ == '__main__':
    main()

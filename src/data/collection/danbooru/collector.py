#!/usr/bin/env python3
"""
Danbooru数据采集器
从Danbooru采集二次元角色图片
"""
import os
import requests
import argparse
import base64
import time
from tqdm import tqdm

def download_image(url, save_path):
    """下载图片"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"下载失败 {url}: {e}")
        return False

def collect_from_danbooru(tags, limit, output_dir, api_key=None, user=None):
    """从Danbooru搜索并下载二次元图片"""
    print(f"从Danbooru搜索标签: {tags}")
    
    try:
        import urllib.parse
        encoded_tags = urllib.parse.quote(tags)
        api_url = f"https://danbooru.donmai.us/posts.json?limit={limit*2}&tags={encoded_tags}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json"
        }
        if api_key and user:
            headers["Authorization"] = f"Basic {base64.b64encode(f'{user}:{api_key}'.encode()).decode()}"
        
        response = requests.get(api_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        posts = response.json()
        
        image_urls = []
        for post in posts:
            if 'file_url' in post and 'rating' in post and post['rating'] == 's':
                image_urls.append(post["file_url"])
        
        print(f"找到 {len(image_urls)} 个安全结果")
        
        downloaded = 0
        for i, image_url in enumerate(image_urls[:limit]):
            file_ext = os.path.splitext(image_url)[1]
            if not file_ext:
                file_ext = ".jpg"
            
            clean_tags = tags.replace(" ", "_").replace(":", "")
            save_path = os.path.join(output_dir, f"danbooru_{clean_tags}_{i+1}{file_ext}")
            
            print(f"下载 {image_url} 到 {save_path}")
            if download_image(image_url, save_path):
                downloaded += 1
                time.sleep(1)
        
        return downloaded
    except Exception as e:
        print(f"Danbooru采集失败: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Danbooru数据采集器')
    parser.add_argument('--tag', required=True, help='要采集的标签')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--limit', type=int, default=20, help='采集图片数量')
    parser.add_argument('--api_key', help='Danbooru API密钥')
    parser.add_argument('--user', help='Danbooru用户名')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    collect_from_danbooru(args.tag, args.limit, args.output_dir, args.api_key, args.user)

if __name__ == '__main__':
    main()

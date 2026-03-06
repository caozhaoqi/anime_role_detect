#!/usr/bin/env python3
"""
批量采集蔚蓝档案角色数据
使用Safebooru API采集
"""
import os
import requests
import argparse
import time
import random
from tqdm import tqdm

# 蔚蓝档案角色列表（使用英文标签）
BLUE_ARCHIVE_CHARACTERS = [
    'arona_(blue_archive)',
    'plana_(blue_archive)',
    'hina_(blue_archive)',
    'ako_(blue_archive)',
    'iori_(blue_archive)',
    'chinatsu_(blue_archive)',
    'iroha_(blue_archive)',
    'aru_(blue_archive)',
    'mutsuki_(blue_archive)',
    'kayoko_(blue_archive)',
    'haruka_(blue_archive)',
    'haruna_(blue_archive)',
    'junko_(blue_archive)',
    'izumi_(blue_archive)',
    'kaede_(blue_archive)',
    'juri_(blue_archive)',
    'sena_(blue_archive)',
    'megu_(blue_archive)',
    'asuna_(blue_archive)',
    'karin_(blue_archive)',
    'shizuko_(blue_archive)',
    'swimsuit_arona',
    'swimsuit_hina',
]

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

def collect_character(character_tag, output_dir, limit=20):
    """采集单个角色的图片"""
    print(f"\n采集角色: {character_tag}")
    
    # 创建角色目录
    char_dir = os.path.join(output_dir, character_tag.replace('_(blue_archive)', '').replace('swimsuit_', 'swimsuit_'))
    os.makedirs(char_dir, exist_ok=True)
    
    # 构建API URL
    import urllib.parse
    encoded_tag = urllib.parse.quote(character_tag)
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
                
                save_path = os.path.join(char_dir, f"{character_tag}_{i+1}{file_ext}")
                
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
    parser = argparse.ArgumentParser(description='批量采集蔚蓝档案角色数据')
    parser.add_argument('--output_dir', default='data/train', help='输出目录')
    parser.add_argument('--limit', type=int, default=20, help='每个角色采集的图片数量')
    parser.add_argument('--characters', nargs='+', help='指定要采集的角色标签')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定要采集的角色列表
    characters = args.characters if args.characters else BLUE_ARCHIVE_CHARACTERS
    
    print(f"准备采集 {len(characters)} 个角色的数据")
    print(f"每个角色采集 {args.limit} 张图片")
    
    total_downloaded = 0
    for character in characters:
        downloaded = collect_character(character, args.output_dir, args.limit)
        total_downloaded += downloaded
        time.sleep(1)
    
    print(f"\n采集完成！共下载 {total_downloaded} 张图片")

if __name__ == '__main__':
    main()

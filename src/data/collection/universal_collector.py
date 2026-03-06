#!/usr/bin/env python3
"""
通用数据采集器
整合多个数据源进行数据采集
"""
import os
import sys
import argparse
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safebooru.collector import collect_from_safebooru
from pixiv.collector import collect_from_pixiv_api
from danbooru.collector import collect_from_danbooru

def collect_character(character_name, output_dir, limit=20, refresh_token=None, sources=['safebooru', 'danbooru', 'pixiv']):
    """
    采集单个角色的数据
    
    Args:
        character_name: 角色名称/标签
        output_dir: 输出目录
        limit: 采集数量
        refresh_token: Pixiv API的refresh_token
        sources: 数据源列表
    """
    print(f"\n采集角色: {character_name}")
    print(f"数据源: {', '.join(sources)}")
    
    char_dir = os.path.join(output_dir, character_name)
    os.makedirs(char_dir, exist_ok=True)
    
    total_downloaded = 0
    
    for source in sources:
        print(f"\n尝试从 {source} 采集...")
        
        if source == 'safebooru':
            downloaded = collect_from_safebooru(character_name, char_dir, limit)
            total_downloaded += downloaded
            if downloaded > 0:
                break
        
        elif source == 'danbooru':
            downloaded = collect_from_danbooru(character_name, limit, char_dir)
            total_downloaded += downloaded
            if downloaded > 0:
                break
        
        elif source == 'pixiv':
            downloaded = collect_from_pixiv_api(character_name, limit, char_dir, refresh_token)
            total_downloaded += downloaded
            if downloaded > 0:
                break
        
        time.sleep(1)
    
    print(f"\n角色 {character_name} 采集完成，共下载 {total_downloaded} 张图片")
    return total_downloaded

def main():
    parser = argparse.ArgumentParser(description='通用数据采集器')
    parser.add_argument('--character', required=True, help='角色名称/标签')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--limit', type=int, default=20, help='采集图片数量')
    parser.add_argument('--refresh_token', help='Pixiv API的refresh_token')
    parser.add_argument('--sources', nargs='+', default=['safebooru', 'danbooru', 'pixiv'],
                       help='数据源列表 (safebooru, danbooru, pixiv)')
    
    args = parser.parse_args()
    
    collect_character(args.character, args.output_dir, args.limit, args.refresh_token, args.sources)

if __name__ == '__main__':
    main()

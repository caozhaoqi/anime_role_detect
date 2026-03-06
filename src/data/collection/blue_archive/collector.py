#!/usr/bin/env python3
"""
蔚蓝档案数据采集器
批量采集蔚蓝档案角色数据
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safebooru.collector import collect_from_safebooru

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
]

def collect_blue_archive(output_dir, limit=20):
    """批量采集蔚蓝档案角色数据"""
    print(f"准备采集 {len(BLUE_ARCHIVE_CHARACTERS)} 个角色的数据")
    print(f"每个角色采集 {limit} 张图片")
    
    total_downloaded = 0
    for character in BLUE_ARCHIVE_CHARACTERS:
        char_dir = os.path.join(output_dir, character.replace('_(blue_archive)', ''))
        os.makedirs(char_dir, exist_ok=True)
        
        downloaded = collect_from_safebooru(character, char_dir, limit)
        total_downloaded += downloaded
    
    print(f"\n采集完成！共下载 {total_downloaded} 张图片")
    return total_downloaded

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='蔚蓝档案数据采集器')
    parser.add_argument('--output_dir', default='data/train', help='输出目录')
    parser.add_argument('--limit', type=int, default=20, help='每个角色采集的图片数量')
    
    args = parser.parse_args()
    
    collect_blue_archive(args.output_dir, args.limit)

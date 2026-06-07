#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理空目录并重新采集缺少图片的角色
"""

import os
import json
import shutil
from pathlib import Path

def remove_empty_dirs(image_dir: str) -> int:
    """删除空目录"""
    removed = 0
    for name in os.listdir(image_dir):
        dir_path = os.path.join(image_dir, name)
        if os.path.isdir(dir_path):
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            if not files:
                shutil.rmtree(dir_path)
                removed += 1
    return removed

def get_characters_with_low_images(image_dir: str, min_count: int = 10) -> list:
    """获取图片数量不足的角色"""
    extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    low_chars = []
    
    for name in os.listdir(image_dir):
        dir_path = os.path.join(image_dir, name)
        if os.path.isdir(dir_path):
            count = sum(1 for f in os.listdir(dir_path) 
                       if os.path.isfile(os.path.join(dir_path, f)) 
                       and os.path.splitext(f)[1].lower() in extensions)
            if count < min_count:
                low_chars.append((name, count))
    
    return low_chars

def main():
    image_dir = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/merged_images'
    
    # 删除空目录
    print(f"正在清理空目录...")
    removed = remove_empty_dirs(image_dir)
    print(f"已删除 {removed} 个空目录")
    
    # 检查剩余目录
    remaining = sum(1 for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name)))
    print(f"剩余目录数: {remaining}")
    
    # 获取图片不足的角色
    low_chars = get_characters_with_low_images(image_dir)
    print(f"\n图片不足10张的角色 ({len(low_chars)} 个):")
    for name, count in low_chars:
        print(f"  {name}: {count} 张")
    
    # 生成重新采集列表
    low_names = {name for name, _ in low_chars}
    
    # 加载角色列表
    with open('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/roles.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 找出需要重新采集的角色
    to_collect = []
    for char in data['characters']:
        chinese_name = char['chinese_name']
        danbooru_tag = char.get('danbooru_tag', '')
        
        # 如果中文名或标签名的目录图片不足
        if chinese_name in low_names or danbooru_tag in low_names:
            to_collect.append(char)
    
    print(f"\n需要重新采集的角色: {len(to_collect)} 个")
    
    # 保存需要采集的角色列表
    output_file = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/to_collect.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(to_collect, f, indent=2, ensure_ascii=False)
    print(f"已保存到: {output_file}")

if __name__ == '__main__':
    main()

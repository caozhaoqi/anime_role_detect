#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.append('spider_image_system/src')
from run.constants import get_pinyin, PINYIN_MAPPING

def main():
    role_file = Path('auto_spider_img/loli-role.txt')
    url_dir = Path('spider_image_system/data/img_url')
    img_dir = Path('data/organized_images')
    
    # 读取角色名单
    with open(role_file, 'r') as f:
        roles = [line.split()[0] for line in f if line.strip()]
    
    print('=' * 80)
    print('角色名单与URL文件/图片文件夹匹配检查')
    print('=' * 80)
    print(f'名单中角色总数: {len(roles)}')
    print()
    
    # 检查URL文件
    print('📋 检查URL文件:')
    url_count = 0
    for role in roles:
        pinyin = get_pinyin(role)
        url_file = url_dir / f'{pinyin}_img.txt'
        if url_file.exists():
            url_count += 1
    print(f'  ✅ 存在URL文件: {url_count}/{len(roles)}')
    print(f'  ❌ 缺失URL文件: {len(roles) - url_count}/{len(roles)}')
    
    # 检查图片文件夹
    print()
    print('📁 检查图片文件夹:')
    img_count = 0
    for role in roles:
        pinyin = get_pinyin(role)
        img_folder = img_dir / pinyin
        if img_folder.exists():
            img_count += 1
    print(f'  ✅ 存在图片文件夹: {img_count}/{len(roles)}')
    print(f'  ❌ 缺失图片文件夹: {len(roles) - img_count}/{len(roles)}')
    
    # 详细列出缺失情况
    print()
    print('🔍 详细缺失列表:')
    
    # URL文件缺失
    print()
    print('URL文件缺失的角色:')
    missing_url = []
    for role in roles:
        pinyin = get_pinyin(role)
        url_file = url_dir / f'{pinyin}_img.txt'
        if not url_file.exists():
            missing_url.append((role, pinyin))
    
    if missing_url:
        for role, pinyin in missing_url:
            print(f'  - {role} ({pinyin})')
    else:
        print('  无')
    
    # 图片文件夹缺失
    print()
    print('图片文件夹缺失的角色:')
    missing_img = []
    for role in roles:
        pinyin = get_pinyin(role)
        img_folder = img_dir / pinyin
        if not img_folder.exists():
            missing_img.append((role, pinyin))
    
    if missing_img:
        for role, pinyin in missing_img:
            print(f'  - {role} ({pinyin})')
    else:
        print('  无')
    
    # 图片数量低于50的角色
    print()
    print('📊 图片数量低于50张的角色:')
    low_count = []
    for role in roles:
        pinyin = get_pinyin(role)
        img_folder = img_dir / pinyin
        if img_folder.exists():
            count = len(list(img_folder.glob('*')))
            if count < 50:
                low_count.append((role, pinyin, count))
    
    if low_count:
        for role, pinyin, count in low_count:
            print(f'  - {role} ({pinyin}): {count} 张')
    else:
        print('  无')
    
    print()
    print('=' * 80)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并所有名单角色的图片到正确的拼音文件夹"""
import os
import shutil
from pathlib import Path
from pypinyin import lazy_pinyin, Style

IMG_DIR = Path('data/organized_images')

# 角色映射 - 把不同格式的文件夹名称映射到正确的拼音
ROLE_MAPPINGS = {
    # 阿尼亚
    'a1ni4ya4': ['a1ni2ya4', 'Anya', 'アーニャ', 'anya'],
    # 布洛妮娅
    'bu4luo4ni2ya4': ['bu4luo4ni1ya4', 'Bronya', 'ブロニア'],
    # 爱丽儿
    'ai4li4er2': ['ai4li4er3', 'Ariel', 'アリエル'],
    # 希儿
    'xi1er2': ['Seele', 'シール', 'xi1er'],
    # 祢豆子
    'mi2dou4zi': ['mi3dou4zi5', 'ni2dou4zi5', 'Nezuko', '竈門祢豆子', 'nezuko'],
}

# 遍历每个角色映射
for target_pinyin, source_names in ROLE_MAPPINGS.items():
    target_dir = IMG_DIR / target_pinyin
    target_dir.mkdir(exist_ok=True)
    moved = 0
    for name in source_names:
        src_dir = IMG_DIR / name
        if src_dir.exists():
            # 移动所有文件
            for f in src_dir.glob('*'):
                if f.is_file():
                    # 避免重名
                    dest = target_dir / f.name
                    counter = 1
                    while dest.exists():
                        name_no_ext, ext = f.stem, f.suffix
                        dest = target_dir / f'{name_no_ext}_{counter}{ext}'
                        counter += 1
                    shutil.move(str(f), str(dest))
                    moved += 1
            # 删除源目录
            print(f'删除空目录: {src_dir.name}')
            try:
                src_dir.rmdir()
            except:
                pass
    if moved > 0:
        print(f'✅ 合并 {target_pinyin}: 移动了 {moved} 张图片')
        final_count = len(list(target_dir.glob('*')))
        print(f'   现在共有 {final_count} 张图片')

# 最终统计
print('\n' + '='*60)
print('📊 最终统计')
print('='*60)
for target_pinyin in ROLE_MAPPINGS.keys():
    target_dir = IMG_DIR / target_pinyin
    if target_dir.exists():
        count = len(list(target_dir.glob('*')))
        print(f'{target_pinyin}: {count} 张')

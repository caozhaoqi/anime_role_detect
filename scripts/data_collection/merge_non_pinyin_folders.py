#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并不同语言的角色名文件夹到拼音名"""
import os
import shutil
from pathlib import Path

IMG_DIR = Path('data/organized_images')

# 合并映射表：目标文件夹 -> 源文件夹列表
MERGE_MAP = {
    'shen2le4': ['Kagura', 'shen2le4 yin1yang2shi1', 'カグラ'],
    'mi2dou4zi': ['Nezuko', 'ni2dou4zi5'],
    'yao2yao2': ['Yaoyao yuan2shen2', 'yao2yao2 yuan2shen2'],
    'tian1tong2ai4li4si1': ['Aris wei4lan2dang4an4', 'ありす'],
    'a1luo4na4': ['a1luo2na4'],
    'you4hu2': ['ユウホ'],
    'na4xi1da2': ['na4xi1da4'],
    'ke3li4': ['ke3li2'],
    'ke4la1la1': ['ke4la1la1'],  # 克拉拉
}

def merge_folders():
    total_moved = 0
    for target, sources in MERGE_MAP.items():
        target_dir = IMG_DIR / target
        target_dir.mkdir(exist_ok=True)

        for src_name in sources:
            src_dir = IMG_DIR / src_name
            if src_dir.exists() and src_dir != target_dir:
                # 移动所有文件
                files = list(src_dir.glob('*'))
                if files:
                    moved = 0
                    for f in files:
                        if f.is_file():
                            dest = target_dir / f.name
                            # 避免重名
                            if dest.exists():
                                name_no_ext, ext = f.stem, f.suffix
                                counter = 1
                                while dest.exists():
                                    dest = target_dir / f'{name_no_ext}_{counter}{ext}'
                                    counter += 1
                            shutil.move(str(f), str(dest))
                            moved += 1
                    print(f'✅ {src_name} -> {target}: 移动 {moved} 个文件')

                    # 如果目录空了，删除
                    try:
                        if not any(src_dir.iterdir()):
                            src_dir.rmdir()
                            print(f'   删除空目录: {src_name}')
                    except:
                        pass
                total_moved += moved

    print(f'\n总计移动: {total_moved} 个文件')

    # 显示还需要处理的文件夹
    print('\n=== 还需要检查的文件夹 ===')
    for d in sorted(IMG_DIR.iterdir()):
        if d.is_dir():
            # 检查是否有包含空格的文件夹
            if ' ' in d.name or any(c.isalpha() and ord(c) > 127 for c in d.name):
                count = len(list(d.glob('*')))
                print(f'  {d.name}: {count} 个文件')

if __name__ == '__main__':
    print('=' * 60)
    print('🚀 开始合并不同语言的角色文件夹')
    print('=' * 60)
    merge_folders()

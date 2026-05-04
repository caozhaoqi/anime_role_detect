#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并声调不同的角色文件夹"""
import os
import shutil
from pathlib import Path

IMG_DIR = Path('data/organized_images')

# 声调差异映射
ROLE_FIXES = {
    # 阿尼亚
    'a1ni2ya4': ['a1ni4ya4'],
    # 布洛妮娅
    'bu4luo4ni1ya4': ['bu4luo4ni2ya4'],
    # 爱丽儿
    'ai4li4er2': ['ai4li4er3'],
    # 希儿
    'xi1er2': ['xi1er3'],
}

for target_pinyin, src_names in ROLE_FIXES.items():
    target_dir = IMG_DIR / target_pinyin
    target_dir.mkdir(exist_ok=True)
    moved = 0
    for src_name in src_names:
        src_dir = IMG_DIR / src_name
        if src_dir.exists():
            for f in src_dir.glob('*'):
                if f.is_file():
                    dest = target_dir / f.name
                    counter = 1
                    while dest.exists():
                        name_no_ext, ext = f.stem, f.suffix
                        dest = target_dir / f'{name_no_ext}_{counter}{ext}'
                        counter += 1
                    shutil.move(str(f), str(dest))
                    moved += 1
            try:
                src_dir.rmdir()
                print(f'删除空目录: {src_name}')
            except:
                pass
    if moved > 0:
        print(f'✅ 合并 {target_pinyin}: 移动了 {moved} 张图片')
        count = len(list(target_dir.glob('*')))
        print(f'   现在共有 {count} 张图片')

print('\n' + '='*60)
print('📊 最终统计')
print('='*60)
for target_pinyin in ROLE_FIXES.keys():
    target_dir = IMG_DIR / target_pinyin
    count = len(list(target_dir.glob('*'))) if target_dir.exists() else 0
    print(f'{target_pinyin}: {count} 张')

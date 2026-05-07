#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并变体文件夹到主目录"""
import shutil
from pathlib import Path

IMG_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images')
OTHER_DIR = IMG_DIR / '其他'

MERGE_MAPPING = {
    'a1lu4': 'a1luo4na4',
    'zhi4': 'zhi4nai3',
    'ren3': 'ren3ye3ren3',
    'カグラ': 'shen2le4',
}

def main():
    print("=" * 60)
    print("📁 合并变体文件夹到主目录")
    print("=" * 60)

    for other_folder, main_folder in MERGE_MAPPING.items():
        other_path = OTHER_DIR / other_folder
        main_path = IMG_DIR / main_folder

        if not other_path.exists():
            print(f"\n⚠️ {other_folder} 不存在，跳过")
            continue

        if not main_path.exists():
            print(f"\n❌ {main_folder} 不存在，无法合并")
            continue

        other_files = list(other_path.glob('*'))
        file_count = len(other_files)

        print(f"\n📁 {other_folder} -> {main_folder} ({file_count} files)")

        moved_count = 0
        for file in other_files:
            if file.is_file():
                dest = main_path / file.name
                if dest.exists():
                    print(f"   ⚠️ 跳过重复文件: {file.name}")
                    continue
                shutil.move(str(file), str(dest))
                moved_count += 1

        print(f"   ✅ 移动了 {moved_count} 个文件")

        try:
            other_path.rmdir()
            print(f"   ✅ 删除空文件夹: {other_folder}")
        except:
            print(f"   ⚠️ 文件夹不为空，保留: {other_folder}")

    print("\n" + "=" * 60)
    print("✅ 合并完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()
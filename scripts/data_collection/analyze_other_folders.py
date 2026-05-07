#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析其他目录中的文件夹，识别应该保留的"""
import hashlib
from pathlib import Path

IMG_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images')
OTHER_DIR = IMG_DIR / '其他'

FOLDER_MAPPING = {
    'Nezuko': 'mi2dou4zi',
    'a1lu4': 'a1luo4na4',
    'zhi4': 'zhi4nai3',
    'ren3': 'ren3ye3ren3',
    'カグラ': 'shen2le4',
    'shen2le4 yin1yang2shi1': 'shen2le4',
    'yao2yao2 yuan2shen2': 'yao2yao2',
    'Yaoyao yuan2shen2': 'yao2yao2',
}

def get_file_hash(filepath):
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def main():
    print("=" * 60)
    print("📊 分析其他目录中的文件夹")
    print("=" * 60)

    for other_folder in sorted(OTHER_DIR.iterdir()):
        if not other_folder.is_dir():
            continue

        folder_name = other_folder.name
        file_count = len(list(other_folder.glob('*')))

        print(f"\n📁 {folder_name} ({file_count} files)")

        if folder_name in FOLDER_MAPPING:
            main_folder = IMG_DIR / FOLDER_MAPPING[folder_name]
            if main_folder.exists():
                main_files = set()
                for f in main_folder.glob('*'):
                    h = get_file_hash(f)
                    if h:
                        main_files.add(h)

                other_unique = 0
                for f in other_folder.glob('*'):
                    h = get_file_hash(f)
                    if h and h not in main_files:
                        other_unique += 1

                if other_unique > 0:
                    print(f"   ⚠️ 主目录 {FOLDER_MAPPING[folder_name]} 存在")
                    print(f"   📊 其他目录有 {other_unique} 张唯一图片")
                    print(f"   💡 建议：合并到主目录")
                else:
                    print(f"   ✅ 主目录 {FOLDER_MAPPING[folder_name]} 存在")
                    print(f"   📊 所有图片都是重复的")
                    print(f"   💡 建议：删除此文件夹")
            else:
                print(f"   ❌ 主目录 {FOLDER_MAPPING[folder_name]} 不存在")
                print(f"   💡 建议：重命名并移到主目录")
        else:
            print(f"   ❓ 未找到对应的映射")
            print(f"   💡 建议：手动检查")

if __name__ == '__main__':
    main()
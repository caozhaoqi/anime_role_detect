#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查30个额外文件夹的图片是否与65个角色重复"""
import hashlib
from pathlib import Path

IMG_DIR = Path('data/organized_images')

# 65个角色的文件夹
ALLOWED_FOLDERS = {
    'a1luo4na4', 'pu3la1na4', 'na4xi1da2', 'ti2bao3', 'ke3li4', 'di2ao4na4',
    'yao2yao2', 'xi1ge2wen2', 'lei3bei4', 'hei1ta3', 'fu2xuan2', 'qi1qi1',
    'zao3you4', 'duo1li4', 'ka3qi2na4', 'san1yue4qi1', 'hua1huo3', 'yin2lang2',
    'tian1tong2ai4li4si1', 'zao3wu4', 'wei2li3nai4', 'an1ke3', 'you4hu2', 'luo4ke3ke3',
    'luo4qian4', 'xiao3mei3yan4', 'xue4xiao3ban3', 'lei2mu3', 'la1mu3', 'kang1na4',
    'si4mi4nai3', 'kai3lu4', 'ke4luo2luo2', 'xiao3shan3', 'yi1li4ya3', 'ren3ye3ren3',
    'zhi4nai3', 'xiao3mai2', 'sha1wu4', 'mao1gong1you4nai4', 'de2li4sha1', 'bu4luo4ni2ya4',
    'ke3lin2', 'ai4li4er3', 'shen2le4', 'bai2shang4chui1xue3', 'yue4qian1ye4', 'fu2li4xi1ya4',
    'li4ta3la1', 'wei2pu3lei3', 'xia4ke4li3', 'na4gan1', 'ke1xie4ni2ya4', 'qi2ta3',
    'kou4er3fu2', 'ke4luo2li4ke1', 'pei4li3ti2ya4', 'a1ni4ya4', 'luo4xi1', 'mi2dou4zi',
    'xi1er3', 'xing4', 'yi1se4lin2', 'fu2lan2', 'fei1mi3li4si1'
}

def get_file_hash(filepath):
    """获取文件MD5哈希"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

print("=" * 60)
print("🔍 检查额外文件夹图片是否重复")
print("=" * 60)

# 第一步：收集65个角色的所有图片哈希
print("\n📊 第一步：收集65个角色的图片哈希...")
role_hashes = set()
role_count = 0
for folder in ALLOWED_FOLDERS:
    folder_path = IMG_DIR / folder
    if folder_path.exists():
        for img in folder_path.glob('*'):
            if img.is_file():
                h = get_file_hash(img)
                if h:
                    role_hashes.add(h)
                role_count += 1

print(f"   65个角色共 {role_count} 张图片, {len(role_hashes)} 个唯一哈希")

# 第二步：检查额外文件夹
print("\n📊 第二步：检查额外文件夹...")
extra_folders = []
for d in IMG_DIR.iterdir():
    if d.is_dir() and d.name not in ALLOWED_FOLDERS:
        extra_folders.append(d.name)

print(f"   发现 {len(extra_folders)} 个额外文件夹")

# 第三步：逐个检查额外文件夹
total_duplicates = 0
total_unique = 0

for folder in sorted(extra_folders):
    folder_path = IMG_DIR / folder
    imgs = list(folder_path.glob('*'))
    img_count = len([f for f in imgs if f.is_file()])

    dup_count = 0
    unique_count = 0

    for img in imgs:
        if img.is_file():
            h = get_file_hash(img)
            if h in role_hashes:
                dup_count += 1
            else:
                unique_count += 1

    dup_pct = dup_count / img_count * 100 if img_count > 0 else 0
    status = "⚠️" if unique_count > 0 else "❌"

    print(f"\n{status} {folder}: {img_count} 张图片")
    print(f"   重复: {dup_count} ({dup_pct:.1f}%)")
    print(f"   唯一: {unique_count}")

    total_duplicates += dup_count
    total_unique += unique_count

print("\n" + "=" * 60)
print("📊 汇总")
print("=" * 60)
print(f"额外文件夹总数: {len(extra_folders)}")
print(f"总重复图片: {total_duplicates} 张")
print(f"总唯一图片: {total_unique} 张")
print(f"总图片数: {total_duplicates + total_unique} 张")

if total_unique > 0:
    print(f"\n⚠️ 有 {total_unique} 张图片是唯一的，不能删除！")
    print("   这些文件夹需要保留")
else:
    print(f"\n✅ 所有 {total_duplicates} 张图片都是重复的，可以删除这些文件夹")

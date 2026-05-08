#!/usr/bin/env python3
"""仔细核对角色名"""
from pathlib import Path
import sys
sys.path.insert(0, 'spider_image_system/src/run')
from constants import PINYIN_MAPPING

def main():
    print("=" * 70)
    print("📋 角色名核对报告")
    print("=" * 70)
    
    # 读取角色名单
    with open('auto_spider_img/loli-role.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    role_list = []
    for i, line in enumerate(lines, 1):
        if line.strip():
            parts = line.strip().split()
            if parts:
                role_list.append({
                    'line': i,
                    'name': parts[0],
                    'source': parts[1] if len(parts) > 1 else '',
                    'english': parts[2] if len(parts) > 2 else '',
                    'japanese': parts[3] if len(parts) > 3 else ''
                })
    
    print(f"\n📌 名单中的角色总数: {len(role_list)}")
    
    # 获取实际文件夹
    img_dir = Path('data/organized_images')
    actual_folders = set()
    for item in img_dir.iterdir():
        if item.is_dir() and item.name not in ['trash', 'trash_nsfw', 'trash_multi_face', '其他']:
            actual_folders.add(item.name)
    
    print(f"📌 实际存在的文件夹: {len(actual_folders)}")
    
    # 1. 检查名单中的角色是否都有拼音映射
    print("\n" + "=" * 70)
    print("1️⃣ 检查拼音映射")
    print("=" * 70)
    
    missing_mapping = []
    for role in role_list:
        if role['name'] not in PINYIN_MAPPING:
            missing_mapping.append(role)
    
    if missing_mapping:
        print(f"❌ 缺少拼音映射的角色 ({len(missing_mapping)}个):")
        for role in missing_mapping:
            print(f"   第{role['line']}行: {role['name']}")
    else:
        print("✅ 所有角色都有拼音映射")
    
    # 2. 检查名单中的角色是否都有对应的文件夹
    print("\n" + "=" * 70)
    print("2️⃣ 检查文件夹存在性")
    print("=" * 70)
    
    missing_folder = []
    for role in role_list:
        pinyin = PINYIN_MAPPING.get(role['name'])
        if pinyin and pinyin not in actual_folders:
            missing_folder.append((role, pinyin))
    
    if missing_folder:
        print(f"❌ 缺少文件夹的角色 ({len(missing_folder)}个):")
        for role, pinyin in missing_folder:
            print(f"   {role['name']} ({pinyin})")
    else:
        print("✅ 所有角色都有对应的文件夹")
    
    # 3. 检查实际文件夹是否都在名单中
    print("\n" + "=" * 70)
    print("3️⃣ 检查额外文件夹")
    print("=" * 70)
    
    # 建立反向映射
    pinyin_to_name = {v: k for k, v in PINYIN_MAPPING.items()}
    
    extra_folders = []
    for folder in actual_folders:
        if folder not in pinyin_to_name:
            extra_folders.append(folder)
    
    if extra_folders:
        print(f"⚠️ 不在名单中的文件夹 ({len(extra_folders)}个):")
        for folder in sorted(extra_folders):
            # 统计图片数量
            folder_path = img_dir / folder
            img_count = len(list(folder_path.glob('*.jpg'))) + \
                       len(list(folder_path.glob('*.png'))) + \
                       len(list(folder_path.glob('*.webp')))
            print(f"   {folder}: {img_count}张图片")
    else:
        print("✅ 所有文件夹都在名单中")
    
    # 4. 检查重复映射
    print("\n" + "=" * 70)
    print("4️⃣ 检查重复映射")
    print("=" * 70)
    
    pinyin_count = {}
    for name, pinyin in PINYIN_MAPPING.items():
        if pinyin not in pinyin_count:
            pinyin_count[pinyin] = []
        pinyin_count[pinyin].append(name)
    
    duplicates = {k: v for k, v in pinyin_count.items() if len(v) > 1}
    if duplicates:
        print(f"❌ 重复的拼音映射 ({len(duplicates)}个):")
        for pinyin, names in duplicates.items():
            print(f"   {pinyin}: {', '.join(names)}")
    else:
        print("✅ 没有重复的拼音映射")
    
    # 5. 检查角色名是否重复
    print("\n" + "=" * 70)
    print("5️⃣ 检查角色名重复")
    print("=" * 70)
    
    name_count = {}
    for role in role_list:
        name = role['name']
        if name not in name_count:
            name_count[name] = []
        name_count[name].append(role['line'])
    
    dup_names = {k: v for k, v in name_count.items() if len(v) > 1}
    if dup_names:
        print(f"❌ 重复的角色名 ({len(dup_names)}个):")
        for name, lines in dup_names.items():
            print(f"   {name}: 第{', '.join(map(str, lines))}行")
    else:
        print("✅ 没有重复的角色名")
    
    # 6. 生成统计报告
    print("\n" + "=" * 70)
    print("📊 统计汇总")
    print("=" * 70)
    
    # 统计各角色的图片数量
    role_stats = []
    for role in role_list:
        pinyin = PINYIN_MAPPING.get(role['name'])
        if pinyin:
            folder = img_dir / pinyin
            if folder.exists():
                img_count = len(list(folder.glob('*.jpg'))) + \
                           len(list(folder.glob('*.png'))) + \
                           len(list(folder.glob('*.webp')))
                role_stats.append({
                    'name': role['name'],
                    'pinyin': pinyin,
                    'count': img_count,
                    'source': role['source']
                })
    
    # 按图片数量排序
    role_stats.sort(key=lambda x: x['count'], reverse=True)
    
    # 统计各范围
    above_50 = [r for r in role_stats if r['count'] >= 50]
    range_20_50 = [r for r in role_stats if 20 <= r['count'] < 50]
    below_20 = [r for r in role_stats if r['count'] < 20]
    
    print(f"✅ 图片≥50张: {len(above_50)}个角色")
    print(f"⚠️ 图片20-49张: {len(range_20_50)}个角色")
    print(f"❌ 图片<20张: {len(below_20)}个角色")
    
    if below_20:
        print(f"\n图片不足20张的角色:")
        for r in sorted(below_20, key=lambda x: x['count']):
            print(f"   {r['name']}: {r['count']}张")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()

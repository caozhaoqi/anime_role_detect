#!/usr/bin/env python3
"""核对角色名单与实际文件夹"""
from pathlib import Path

# 读取拼音映射
with open('spider_image_system/src/run/constants.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 提取 PINYIN_MAPPING
start = content.find('PINYIN_MAPPING = {')
end = content.find('}', start) + 1
mapping_str = content[start:end]
exec(mapping_str)

# 读取角色名单
with open('auto_spider_img/loli-role.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
role_names = [line.split()[0] for line in lines if line.strip()]

# 获取实际文件夹
img_dir = Path('data/organized_images')
actual_folders = set()
for item in img_dir.iterdir():
    if item.is_dir() and item.name not in ['trash', 'trash_nsfw', 'trash_multi_face', '其他', '.', '..']:
        actual_folders.add(item.name)

# 核对
missing = []
present = []
for name in role_names:
    pinyin = PINYIN_MAPPING.get(name)
    if pinyin and pinyin not in actual_folders:
        missing.append((name, pinyin))
    else:
        present.append(name)

print("=" * 70)
print("📋 角色名单核对")
print("=" * 70)
print(f"名单总数: {len(role_names)}")
print(f"已存在: {len(present)}")
print(f"缺失: {len(missing)}")
print("=" * 70)

if missing:
    print("\n❌ 缺失的角色:")
    for name, pinyin in missing:
        print(f"  • {name} ({pinyin})")

# 找出实际存在但不在名单中的文件夹
extra_folders = []
for folder in actual_folders:
    # 检查是否是拼音映射中的角色
    is_mapped = False
    for name, pinyin in PINYIN_MAPPING.items():
        if pinyin == folder:
            is_mapped = True
            break
    if not is_mapped:
        extra_folders.append(folder)

if extra_folders:
    print("\n⚠️ 实际存在但不在名单中的文件夹:")
    for folder in sorted(extra_folders):
        print(f"  • {folder}")

print("\n" + "=" * 70)

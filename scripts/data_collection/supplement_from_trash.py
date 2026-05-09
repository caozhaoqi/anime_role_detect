import os
import shutil
import json
from pathlib import Path

LOW_DATA_ROLES = {
    'fu2li4xi1ya4': '芙丽希娅',    # 16张 -> 需要14张
    'ai4li4er3': '爱丽儿',          # 20张 -> 需要10张
    'yue4qian1ye4': '月千夜',      # 20张 -> 需要10张
    'zao3wu4': '早雾',              # 24张 -> 需要6张
    'xiao3shan3': '小闪',          # 24张 -> 需要6张
    'fu2lan2': '芙兰',              # 24张 -> 需要6张
    'lei2mu3': '雷姆',              # 26张 -> 需要4张
    'a1ni4ya4': '阿尼亚',           # 26张 -> 需要4张
}

SOURCE_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images/trash_nsfw'
TARGET_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'
MIN_IMAGES = 30

def find_images_in_trash(role_pinyin):
    """从trash_nsfw目录中查找特定角色的图片"""
    found = []
    if not os.path.exists(SOURCE_DIR):
        return found
    
    for filename in os.listdir(SOURCE_DIR):
        if role_pinyin in filename.lower():
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                found.append(filename)
    return found

def supplement_from_trash():
    print("="*60)
    print("从trash_nsfw目录补充角色数据")
    print("="*60)
    print(f"源目录: {SOURCE_DIR}")
    print(f"目标目录: {TARGET_DIR}")
    print(f"最低图片数: {MIN_IMAGES}")
    print()

    total_supplemented = 0

    for pinyin, role_name in LOW_DATA_ROLES.items():
        target_dir = os.path.join(TARGET_DIR, pinyin)
        
        if not os.path.exists(target_dir):
            print(f"  {role_name} ({pinyin}): 目标目录不存在，跳过")
            continue

        existing_files = set()
        for f in os.listdir(target_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                existing_files.add(f)

        current_count = len(existing_files)
        needed = MIN_IMAGES - current_count

        if needed <= 0:
            print(f"  {role_name} ({pinyin}): 已满足 ({current_count}张)")
            continue

        print(f"  {role_name} ({pinyin}): 当前 {current_count} 张, 需要补充 {needed} 张")
        
        available_files = find_images_in_trash(pinyin)
        print(f"    在trash_nsfw中找到 {len(available_files)} 张候选图片")
        
        supplemented = 0
        for filename in available_files:
            if supplemented >= needed:
                break
            src = os.path.join(SOURCE_DIR, filename)
            dst = os.path.join(target_dir, filename)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                supplemented += 1

        new_count = current_count + supplemented
        total_supplemented += supplemented
        print(f"    成功补充 {supplemented} 张, 现在 {new_count} 张")

    print()
    print(f"补充完成！共补充 {total_supplemented} 张图片")

    stats = []
    for pinyin in sorted(os.listdir(TARGET_DIR)):
        role_dir = os.path.join(TARGET_DIR, pinyin)
        if os.path.isdir(role_dir):
            count = len([f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
            stats.append({'pinyin': pinyin, 'count': count})

    stats.sort(key=lambda x: x['count'])

    with open(os.path.join(TARGET_DIR, 'dataset_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n更新后的数据集统计:")
    for s in stats:
        print(f"  {s['pinyin']}: {s['count']} 张")

    total = sum(s['count'] for s in stats)
    print(f"\n总计: {len(stats)} 个角色, {total} 张图片")

if __name__ == '__main__':
    supplement_from_trash()
#!/usr/bin/env python3
"""
优先采集图片数量不足100张的角色
按照图片数量从少到多排序，优先采集数量最少的角色
"""

import os
import subprocess
from pathlib import Path

# 定义目录路径
BASE_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect")
DATASET_DIR = BASE_DIR / "data/merged_english_dataset"

# 目标图片数量
TARGET_COUNT = 100

def get_role_image_counts():
    """获取每个角色的图片数量"""
    role_counts = {}
    
    if not DATASET_DIR.exists():
        return role_counts
        
    for role_dir in DATASET_DIR.iterdir():
        if not role_dir.is_dir():
            continue
            
        count = len(list(role_dir.glob("*.jpg"))) + len(list(role_dir.glob("*.png")))
        role_counts[role_dir.name] = count
        
    return role_counts

def get_priority_roles():
    """获取需要补充采集的角色，按图片数量升序排序"""
    role_counts = get_role_image_counts()
    
    # 筛选图片数量 < 100 的角色
    need_collect = {k: v for k, v in role_counts.items() if v < TARGET_COUNT}
    
    # 按图片数量升序排序（数量少的优先）
    sorted_roles = sorted(need_collect.items(), key=lambda x: x[1])
    
    return sorted_roles

def generate_search_keywords(role_name):
    """生成角色搜索关键词"""
    keywords = [role_name]
    
    # 添加一些通用的搜索后缀
    suffixes = ["", " anime", " fanart", " wallpaper", " character"]
    
    return [f"{role_name}{suffix}" for suffix in suffixes]

def print_priority_list():
    """打印优先级采集列表"""
    sorted_roles = get_priority_roles()
    
    print(f"=== 优先采集角色列表 (目标: 每个角色 {TARGET_COUNT} 张图片) ===")
    print()
    
    if not sorted_roles:
        print("✓ 所有角色图片数量已达到或超过 100 张！")
        return
    
    # 按数量分组显示
    print("📊 优先级分组：")
    print("-" * 60)
    
    # 第一组：0-20张（最需优先采集）
    print("\n🔥 第一优先级 (0-20张) - 急需补充：")
    for role, count in sorted_roles:
        if count <= 20:
            needed = TARGET_COUNT - count
            print(f"   [{count:3d}/{TARGET_COUNT}] {role} (还需: {needed}张)")
    
    # 第二组：21-50张
    print("\n⚡ 第二优先级 (21-50张) - 需要补充：")
    for role, count in sorted_roles:
        if 21 <= count <= 50:
            needed = TARGET_COUNT - count
            print(f"   [{count:3d}/{TARGET_COUNT}] {role} (还需: {needed}张)")
    
    # 第三组：51-80张
    print("\n🔄 第三优先级 (51-80张) - 建议补充：")
    for role, count in sorted_roles:
        if 51 <= count <= 80:
            needed = TARGET_COUNT - count
            print(f"   [{count:3d}/{TARGET_COUNT}] {role} (还需: {needed}张)")
    
    # 第四组：81-99张
    print("\n📈 第四优先级 (81-99张) - 少量补充：")
    for role, count in sorted_roles:
        if 81 <= count <= 99:
            needed = TARGET_COUNT - count
            print(f"   [{count:3d}/{TARGET_COUNT}] {role} (还需: {needed}张)")
    
    print("\n" + "-" * 60)
    print(f"总计需要补充的角色: {len(sorted_roles)} 个")
    
    return sorted_roles

def get_top_priority_roles(n=10):
    """获取最需要采集的前N个角色"""
    sorted_roles = get_priority_roles()
    return sorted_roles[:n]

if __name__ == "__main__":
    priority_roles = print_priority_list()
    
    if priority_roles:
        print("\n💡 建议操作：")
        print("1. 使用爬虫脚本采集上述角色")
        print("2. 优先采集第一、第二优先级的角色")
        print("3. 可以手动搜索图片补充数量较少的角色")
        
        # 输出最需要采集的角色关键词
        top_roles = get_top_priority_roles(10)
        print("\n🔍 最需要采集的角色关键词：")
        for role, count in top_roles:
            print(f"   - {role}")

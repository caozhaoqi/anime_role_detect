#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一采集脚本 - 使用英文名采集规避拼音问题
"""

import os
import requests
import time
from .config import *
from .utils import *

def collect_all_roles(use_english=True):
    """采集所有角色的URL"""
    print("=" * 70)
    print("🚀 开始采集角色URL")
    print(f"使用{'英文名' if use_english else '中文名'}采集")
    print("=" * 70)
    
    # 检查爬虫服务
    try:
        response = requests.get(f"{API_BASE}/spider/status", timeout=5)
        print("✅ 爬虫服务连接成功")
    except Exception as e:
        print(f"❌ 爬虫服务未运行: {str(e)}")
        return
    
    roles = read_role_file(ROLE_FILE)
    print(f"\n📋 共读取到 {len(roles)} 个角色")
    
    success_count = 0
    fail_count = 0
    
    for i, line in enumerate(roles, 1):
        role = parse_role_line(line)
        name = role['name']
        keyword = role['en'] if use_english and role['en'] else name
        
        print(f"\n[{i}/{len(roles)}] {name} ({keyword})")
        
        if spider_single_role(keyword):
            print(f"  ✅ 采集成功")
            success_count += 1
        else:
            print(f"  ❌ 采集失败")
            fail_count += 1
        
        time.sleep(2)
    
    print("\n" + "=" * 70)
    print(f"采集完成: 成功 {success_count} 个, 失败 {fail_count} 个")
    print("=" * 70)

def download_all_roles(use_english=True):
    """下载所有角色的图片"""
    print("=" * 70)
    print("🚀 开始下载角色图片")
    print(f"目标: 每个角色 {TARGET_COUNT} 张")
    print("=" * 70)
    
    roles = read_role_file(ROLE_FILE)
    print(f"\n📋 共读取到 {len(roles)} 个角色")
    
    total_downloaded = 0
    completed_count = 0
    remaining_roles = []
    
    for i, line in enumerate(roles, 1):
        role = parse_role_line(line)
        name = role['name']
        pinyin = get_role_info(name)['pinyin']
        keyword = role['en'] if use_english and role['en'] else name
        
        role_dir = os.path.join(REORGANIZED_DATASET, pinyin)
        ensure_dir(role_dir)
        
        current_count = get_image_count(role_dir)
        
        if current_count >= TARGET_COUNT:
            print(f"  ✅ [{i}/{len(roles)}] {name}: {current_count} 张 (已达标)")
            completed_count += 1
            continue
        
        need_count = TARGET_COUNT - current_count
        print(f"\n📦 [{i}/{len(roles)}] {name} ({pinyin}): 当前 {current_count} 张, 需要补充 {need_count} 张")
        
        # 查找URL文件（支持多种关键词）
        url_file = None
        for identifier in [keyword, name, pinyin]:
            url_file = find_url_file(identifier)
            if url_file:
                print(f"  📋 找到URL文件: {os.path.basename(url_file)}")
                break
        
        if url_file:
            existing_files = get_image_files(role_dir)
            downloaded = download_images(url_file, role_dir, need_count, existing_files)
            
            if downloaded > 0:
                print(f"  📥 成功下载: {downloaded} 张")
                total_downloaded += downloaded
        else:
            print(f"  ⚠️ 未找到URL文件")
        
        current_count = get_image_count(role_dir)
        
        if current_count >= TARGET_COUNT:
            print(f"  ✅ 当前: {current_count} 张 (已达标)")
            completed_count += 1
        else:
            print(f"  ⚠️ 当前: {current_count} 张 (仍需 {TARGET_COUNT - current_count} 张)")
            remaining_roles.append((name, current_count))
        
        time.sleep(1)
    
    print("\n" + "=" * 70)
    print(f"下载完成: 总共下载 {total_downloaded} 张图片")
    print(f"已达标角色: {completed_count}/{len(roles)}")
    print("=" * 70)
    
    total_images = sum(get_image_count(os.path.join(REORGANIZED_DATASET, get_role_info(r)['pinyin'])) for r in roles)
    
    print(f"\n📊 最终统计:")
    print(f"角色总数: {len(roles)}")
    print(f"图片总数: {total_images}")
    print(f"平均每角色: {total_images / len(roles):.1f} 张")
    
    if remaining_roles:
        print(f"\n⚠️ 仍有 {len(remaining_roles)} 个角色未达标")
        for name, count in remaining_roles[:10]:
            print(f"  {name}: {count}/{TARGET_COUNT}")
        if len(remaining_roles) > 10:
            print(f"  ...还有 {len(remaining_roles) - 10} 个")
    else:
        print("\n🎉 所有角色均已达标！")

def check_status():
    """检查当前状态"""
    print("=" * 90)
    print("📊 数据集状态报告")
    print("=" * 90)
    
    roles = read_role_file(ROLE_FILE)
    total_images = 0
    completed_count = 0
    in_progress_count = 0
    not_started_count = 0
    
    print(f"{'角色名':<12} {'英文名':<15} {'图片数':<6} {'状态'}")
    print("-" * 90)
    
    for line in roles:
        role = parse_role_line(line)
        name = role['name']
        en_name = role['en']
        pinyin = get_role_info(name)['pinyin']
        
        role_dir = os.path.join(REORGANIZED_DATASET, pinyin)
        img_count = get_image_count(role_dir)
        total_images += img_count
        
        if img_count >= TARGET_COUNT:
            status = "✅ 已达标"
            completed_count += 1
        elif img_count > 0:
            status = "🔄 进行中"
            in_progress_count += 1
        else:
            status = "⏳ 未开始"
            not_started_count += 1
        
        print(f"{name:<12} {en_name:<15} {img_count:<6} {status}")
    
    print("-" * 90)
    print(f"\n📈 汇总统计:")
    print(f"  角色总数: {len(roles)}")
    print(f"  图片总数: {total_images}")
    print(f"  平均每角色: {total_images / len(roles):.1f} 张")
    print(f"  已达标: {completed_count} 个")
    print(f"  进行中: {in_progress_count} 个")
    print(f"  未开始: {not_started_count} 个")
    print(f"\n🎯 还需下载: {(len(roles) * TARGET_COUNT) - total_images} 张")
    print("=" * 90)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='角色图片采集与下载工具')
    parser.add_argument('action', choices=['collect', 'download', 'status'],
                        help='操作类型: collect(采集URL), download(下载图片), status(查看状态)')
    parser.add_argument('--english', action='store_true', default=True,
                        help='使用英文名采集（默认启用）')
    
    args = parser.parse_args()
    
    if args.action == 'collect':
        collect_all_roles(use_english=args.english)
    elif args.action == 'download':
        download_all_roles(use_english=args.english)
    elif args.action == 'status':
        check_status()

if __name__ == '__main__':
    main()

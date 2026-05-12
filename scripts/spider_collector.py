#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一采集脚本 - 使用多语言角色映射机制
"""

import os
import sys
import requests
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import *
from scripts.utils import *
from scripts.role_mapping import init_manager, find_role, find_url_files_for_role

def collect_all_roles_multi_lang():
    """使用多语言关键词采集所有角色的URL"""
    print("=" * 70)
    print("🚀 开始多语言采集角色URL")
    print("策略: 依次尝试英文名 -> 中文名 -> 日文名")
    print("=" * 70)
    
    # 检查爬虫服务
    try:
        response = requests.get(f"{API_BASE}/spider/status", timeout=5)
        print("✅ 爬虫服务连接成功")
    except Exception as e:
        print(f"❌ 爬虫服务未运行: {str(e)}")
        return
    
    # 初始化角色管理器
    init_manager(ROLE_FILE)
    
    roles = read_role_file(ROLE_FILE)
    print(f"\n📋 共读取到 {len(roles)} 个角色")
    
    success_count = 0
    fail_count = 0
    already_have_url = 0
    
    for i, line in enumerate(roles, 1):
        role = parse_role_line(line)
        name = role['name']
        en_name = role['en']
        jp_name = role['jp']
        
        print(f"\n[{i}/{len(roles)}] {name}")
        
        # 检查是否已有URL文件
        url_files = find_url_files_for_role(name)
        if url_files:
            print(f"  ⏭️ 已有 {len(url_files)} 个URL文件，跳过采集")
            already_have_url += 1
            success_count += 1
            continue
        
        # 多语言采集策略
        collected = False
        keywords = []
        
        # 优先使用英文名
        if en_name:
            keywords.append(('英文名', en_name))
        
        # 然后使用中文名
        keywords.append(('中文名', name))
        
        # 最后使用日文名
        if jp_name:
            keywords.append(('日文名', jp_name))
        
        for lang_type, keyword in keywords:
            print(f"  尝试{lang_type}: {keyword}")
            
            if spider_single_role(keyword):
                print(f"  ✅ 采集成功")
                collected = True
                break
            else:
                print(f"  ⚠️ {lang_type}采集失败")
        
        if collected:
            success_count += 1
        else:
            print(f"  ❌ 所有语言均采集失败")
            fail_count += 1
        
        time.sleep(2)
    
    print("\n" + "=" * 70)
    print(f"采集完成:")
    print(f"  - 成功: {success_count} 个")
    print(f"  - 失败: {fail_count} 个")
    print(f"  - 已存在URL: {already_have_url} 个")
    print("=" * 70)

def download_all_roles_unified():
    """使用统一角色识别下载所有角色的图片"""
    print("=" * 70)
    print("🚀 开始统一下载角色图片")
    print(f"目标: 每个角色 {TARGET_COUNT} 张")
    print("=" * 70)
    
    # 初始化角色管理器
    init_manager(ROLE_FILE)
    
    roles = read_role_file(ROLE_FILE)
    print(f"\n📋 共读取到 {len(roles)} 个角色")
    
    total_downloaded = 0
    completed_count = 0
    remaining_roles = []
    
    for i, line in enumerate(roles, 1):
        role = parse_role_line(line)
        name = role['name']
        pinyin = get_role_info(name)['pinyin']
        
        role_dir = os.path.join(REORGANIZED_DATASET, pinyin)
        ensure_dir(role_dir)
        
        current_count = get_image_count(role_dir)
        
        if current_count >= TARGET_COUNT:
            print(f"  ✅ [{i}/{len(roles)}] {name}: {current_count} 张 (已达标)")
            completed_count += 1
            continue
        
        need_count = TARGET_COUNT - current_count
        print(f"\n📦 [{i}/{len(roles)}] {name} ({pinyin}): 当前 {current_count} 张, 需要补充 {need_count} 张")
        
        # 使用角色映射机制查找所有相关URL文件
        url_files = find_url_files_for_role(name)
        
        if url_files:
            print(f"  📋 找到 {len(url_files)} 个URL文件")
            
            existing_files = get_image_files(role_dir)
            total_downloaded_for_role = 0
            
            for url_file in url_files:
                print(f"    处理: {os.path.basename(url_file)}")
                downloaded = download_images(url_file, role_dir, need_count, existing_files)
                
                if downloaded > 0:
                    print(f"    📥 下载: {downloaded} 张")
                    total_downloaded_for_role += downloaded
                    need_count -= downloaded
                    existing_files = get_image_files(role_dir)
                
                if need_count <= 0:
                    break
            
            if total_downloaded_for_role > 0:
                print(f"  📥 总共下载: {total_downloaded_for_role} 张")
                total_downloaded += total_downloaded_for_role
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
    
    total_images = sum(get_image_count(os.path.join(REORGANIZED_DATASET, get_role_info(parse_role_line(r)['name'])['pinyin'])) for r in roles)
    
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

def check_status_with_mapping():
    """使用角色映射检查当前状态"""
    print("=" * 90)
    print("📊 数据集状态报告 (含角色映射)")
    print("=" * 90)
    
    # 初始化角色管理器
    init_manager(ROLE_FILE)
    
    roles = read_role_file(ROLE_FILE)
    total_images = 0
    completed_count = 0
    in_progress_count = 0
    not_started_count = 0
    
    print(f"{'角色名':<12} {'英文名':<15} {'图片数':<6} {'URL文件':<6} {'状态'}")
    print("-" * 90)
    
    for line in roles:
        role = parse_role_line(line)
        name = role['name']
        en_name = role['en']
        pinyin = get_role_info(name)['pinyin']
        
        role_dir = os.path.join(REORGANIZED_DATASET, pinyin)
        img_count = get_image_count(role_dir)
        total_images += img_count
        
        # 检查URL文件
        url_files = find_url_files_for_role(name)
        url_count = len(url_files)
        
        if img_count >= TARGET_COUNT:
            status = "✅ 已达标"
            completed_count += 1
        elif img_count > 0:
            status = "🔄 进行中"
            in_progress_count += 1
        else:
            status = "⏳ 未开始"
            not_started_count += 1
        
        print(f"{name:<12} {en_name:<15} {img_count:<6} {url_count:<6} {status}")
    
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
    
    parser = argparse.ArgumentParser(description='角色图片采集与下载工具（多语言版）')
    parser.add_argument('action', choices=['collect', 'download', 'status'],
                        help='操作类型: collect(多语言采集URL), download(统一下载), status(查看状态)')
    
    args = parser.parse_args()
    
    if args.action == 'collect':
        collect_all_roles_multi_lang()
    elif args.action == 'download':
        download_all_roles_unified()
    elif args.action == 'status':
        check_status_with_mapping()

if __name__ == '__main__':
    main()

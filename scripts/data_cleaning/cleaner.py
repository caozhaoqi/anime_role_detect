#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据清理工具 - 使用工具库
整合：去重 + 质量检查 + 过滤
"""

import os
import sys
import argparse
from tqdm import tqdm
from collections import defaultdict

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils


def main():
    parser = argparse.ArgumentParser(description='统一数据清理工具')
    parser.add_argument('--data-dir', required=True, help='数据目录')
    parser.add_argument('--output-report', default=None, help='输出报告路径')
    parser.add_argument('--auto-delete', action='store_true', help='自动删除不合格文件')
    parser.add_argument('--skip-size', action='store_true', help='跳过大小检查')
    parser.add_argument('--skip-quality', action='store_true', help='跳过质量检查')
    parser.add_argument('--skip-duplicates', action='store_true', help='跳过去重')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 统一数据清理工具")
    print("=" * 70)
    print(f"数据目录: {args.data_dir}")
    print()
    
    # ==================== 1. 扫描 ====================
    print("📁 阶段1: 扫描图片...")
    all_files = utils.scan_images(args.data_dir)
    role_images = utils.scan_role_images(args.data_dir)
    
    total_roles = len(role_images)
    total_files = len(all_files)
    print(f"   发现: {total_roles} 个角色, {total_files} 个文件")
    
    if total_files == 0:
        print("❌ 没有找到图片文件")
        return
    
    # ==================== 2. 大小过滤 ====================
    filtered_by_size = []
    removed_by_size = []
    
    if not args.skip_size:
        print(f"\n📐 阶段2: 大小过滤 (最小 {utils.MIN_FILE_SIZE_KB}KB)...")
        for file_path in tqdm(all_files, desc="大小检查"):
            try:
                file_size_kb = os.path.getsize(file_path) / 1024
                if file_size_kb >= utils.MIN_FILE_SIZE_KB:
                    filtered_by_size.append(file_path)
                else:
                    removed_by_size.append(file_path)
            except:
                removed_by_size.append(file_path)
        
        print(f"   合格: {len(filtered_by_size)}, 移除: {len(removed_by_size)}")
    else:
        filtered_by_size = all_files
        print(f"\n📐 阶段2: 跳过大小检查")
    
    # ==================== 3. 质量检查 ====================
    passed_quality = []
    removed_quality = {}
    
    if not args.skip_quality:
        print(f"\n✨ 阶段3: 质量检查 (最小 {utils.MIN_IMAGE_WIDTH}x{utils.MIN_IMAGE_HEIGHT})...")
        passed_quality, removed_quality = utils.batch_quality_check(filtered_by_size)
        print(f"   合格: {len(passed_quality)}, 移除: {len(removed_quality)}")
    else:
        passed_quality = filtered_by_size
        print(f"\n✨ 阶段3: 跳过质量检查")
    
    # ==================== 4. 去重 ====================
    unique_files = []
    removed_duplicates = []
    
    if not args.skip_duplicates:
        print(f"\n🔍 阶段4: 去重...")
        duplicates = utils.find_duplicate_files(passed_quality)
        
        if duplicates:
            removed_duplicates = utils.get_deletion_candidates(duplicates)
            unique_files = [f for f in passed_quality if f not in removed_duplicates]
            print(f"   发现: {len(duplicates)} 组重复")
        else:
            unique_files = passed_quality
        
        print(f"   保留: {len(unique_files)}, 移除: {len(removed_duplicates)}")
    else:
        unique_files = passed_quality
        print(f"\n🔍 阶段4: 跳过去重")
    
    # ==================== 5. 报告 ====================
    print("\n" + "=" * 70)
    print("📊 清理报告")
    print("=" * 70)
    
    print(f"\n📈 统计:")
    print(f"   原始文件: {total_files}")
    print(f"   最终文件: {len(unique_files)}")
    print(f"   移除文件: {total_files - len(unique_files)}")
    print(f"   保留率: {(len(unique_files)/total_files*100):.1f}%")
    
    print(f"\n🗑️ 移除原因:")
    if not args.skip_size:
        print(f"   大小不合格: {len(removed_by_size)}")
    if not args.skip_quality:
        print(f"   质量不合格: {len(removed_quality)}")
    if not args.skip_duplicates:
        print(f"   重复文件: {len(removed_duplicates)}")
    
    # 按角色统计
    print(f"\n👥 角色统计:")
    final_role_stats = defaultdict(int)
    for file_path in unique_files:
        for role, files in role_images.items():
            if file_path in files:
                final_role_stats[role] += 1
    
    sorted_roles = sorted(final_role_stats.items(), key=lambda x: x[1], reverse=True)
    for i, (role, count) in enumerate(sorted_roles[:10], 1):
        print(f"   {i}. {role}: {count}")
    if len(sorted_roles) > 10:
        print(f"   ... 还有 {len(sorted_roles) - 10} 个角色")
    
    # 保存报告
    report = {
        'summary': {
            'total_files': total_files,
            'final_files': len(unique_files),
            'removed_files': total_files - len(unique_files),
        },
        'removed_by_size': removed_by_size,
        'removed_by_quality': removed_quality,
        'removed_duplicates': removed_duplicates,
        'role_stats': dict(sorted_roles),
    }
    
    if args.output_report:
        utils.save_json(report, args.output_report)
        print(f"\n💾 报告已保存到: {args.output_report}")
    
    # ==================== 6. 删除 ====================
    if args.auto_delete:
        print("\n" + "=" * 70)
        print("🗑️ 开始删除...")
        print("=" * 70)
        
        all_to_delete = []
        if not args.skip_size:
            all_to_delete.extend(removed_by_size)
        if not args.skip_quality:
            all_to_delete.extend(removed_quality.keys())
        if not args.skip_duplicates:
            all_to_delete.extend(removed_duplicates)
        
        if all_to_delete:
            success, failed = utils.delete_files(all_to_delete)
            print(f"\n✅ 删除完成: 成功 {success}, 失败 {failed}")
        else:
            print("\n✅ 没有需要删除的文件")
    else:
        print(f"\n⚠️ 未启用自动删除，请使用 --auto-delete 参数执行删除")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能去重脚本 - 根据筛选报告执行清理
"""

import os
import json
import argparse
from tqdm import tqdm


def load_report(report_path):
    """加载筛选报告"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def delete_files(file_list, desc="删除文件"):
    """批量删除文件"""
    deleted = 0
    failed = 0
    skipped = 0
    
    for item in tqdm(file_list, desc=desc):
        filepath = item.get('path', item.get('duplicate'))
        
        if not filepath or not os.path.exists(filepath):
            skipped += 1
            continue
        
        try:
            os.remove(filepath)
            deleted += 1
        except Exception as e:
            failed += 1
            print(f"❌ 删除失败 {filepath}: {e}")
    
    return deleted, failed, skipped


def get_keep_strategy(original, duplicate):
    """确定保留策略"""
    # 策略1: 优先保留文件名较短的
    if len(os.path.basename(original)) < len(os.path.basename(duplicate)):
        return original, duplicate
    
    # 策略2: 优先保留不在裁剪目录中的
    if 'crop' in duplicate and 'crop' not in original:
        return original, duplicate
    
    # 策略3: 优先保留在更干净的目录中
    clean_dirs = ['merged_dataset', 'training_dataset']
    orig_in_clean = any(d in original for d in clean_dirs)
    dup_in_clean = any(d in duplicate for d in clean_dirs)
    
    if orig_in_clean and not dup_in_clean:
        return original, duplicate
    
    # 默认保留原始的，删除重复的
    return original, duplicate


def deduplicate(report, dry_run=True):
    """执行去重"""
    stats = {
        'size_failed': 0,
        'quality_failed': 0,
        'duplicates': 0,
        'total': 0
    }
    
    # 1. 删除大小过滤的文件
    if report.get('failed_size'):
        print(f"\n📐 删除大小过滤文件 ({len(report['failed_size'])} 个)...")
        if not dry_run:
            d, f, s = delete_files(report['failed_size'], "大小过滤")
            stats['size_failed'] = d
    
    # 2. 删除质量过滤的文件
    if report.get('failed_quality'):
        print(f"\n✨ 删除质量过滤文件 ({len(report['failed_quality'])} 个)...")
        if not dry_run:
            d, f, s = delete_files(report['failed_quality'], "质量过滤")
            stats['quality_failed'] = d
    
    # 3. 删除重复文件
    if report.get('duplicates'):
        print(f"\n🔍 处理重复文件 ({len(report['duplicates'])} 个)...")
        to_delete = []
        
        for dup in report['duplicates']:
            keep, delete = get_keep_strategy(dup['original'], dup['duplicate'])
            to_delete.append({'path': delete})
        
        if not dry_run:
            d, f, s = delete_files(to_delete, "重复文件")
            stats['duplicates'] = d
    
    stats['total'] = stats['size_failed'] + stats['quality_failed'] + stats['duplicates']
    return stats


def main():
    parser = argparse.ArgumentParser(description='智能去重脚本')
    parser.add_argument('--report', type=str, default='data/quick_filter_report.json', help='筛选报告')
    parser.add_argument('--dry-run', action='store_true', help='试运行模式，不实际删除')
    parser.add_argument('--auto', action='store_true', help='自动模式，无需确认')
    args = parser.parse_args()
    
    print("🚀 智能去重开始")
    print("=" * 60)
    
    if not os.path.exists(args.report):
        print(f"❌ 报告文件不存在: {args.report}")
        return
    
    # 加载报告
    report = load_report(args.report)
    
    print(f"\n📊 报告统计:")
    print(f"   大小过滤: {len(report.get('failed_size', []))} 个")
    print(f"   质量过滤: {len(report.get('failed_quality', []))} 个")
    print(f"   重复文件: {len(report.get('duplicates', []))} 个")
    
    if args.dry_run:
        print("\n⚠️ 试运行模式 - 不会实际删除文件")
    elif not args.auto:
        print("\n⚠️ 警告：即将删除文件！")
        confirm = input("确认继续？(yes/no): ")
        if confirm.lower() != 'yes':
            print("操作取消")
            return
    
    # 执行去重
    stats = deduplicate(report, dry_run=args.dry_run)
    
    print("\n" + "=" * 60)
    print("✅ 去重完成")
    print("=" * 60)
    
    if args.dry_run:
        print("\n📋 试运行统计:")
        print(f"   计划删除大小过滤: {len(report.get('failed_size', []))} 个")
        print(f"   计划删除质量过滤: {len(report.get('failed_quality', []))} 个")
        print(f"   计划删除重复文件: {len(report.get('duplicates', []))} 个")
    else:
        print(f"\n📋 删除统计:")
        print(f"   大小过滤: {stats['size_failed']} 个")
        print(f"   质量过滤: {stats['quality_failed']} 个")
        print(f"   重复文件: {stats['duplicates']} 个")
        print(f"   总计删除: {stats['total']} 个")


if __name__ == '__main__':
    main()

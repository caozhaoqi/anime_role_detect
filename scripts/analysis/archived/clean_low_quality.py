#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理 final_dataset 中的低质量数据
包括：SVG文件、过小图片、损坏图片、非图片文件
"""

import os
from pathlib import Path
from PIL import Image
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect")

# 目标目录
TARGET_DIR = PROJECT_ROOT / "data" / "final_dataset"

# 角色列表文件
ROLE_LIST_FILE = PROJECT_ROOT / "auto_spider_img" / "loli-role.txt"

# 图片扩展名
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}

# 最小图片尺寸（宽或高）
MIN_IMAGE_SIZE = 64

# 最小文件大小（字节）
MIN_FILE_SIZE = 1024  # 1KB


def parse_role_list():
    """解析角色列表文件"""
    role_info = {}
    
    if not ROLE_LIST_FILE.exists():
        return role_info
    
    with open(ROLE_LIST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                chinese_name = parts[0]
                game = parts[1] if len(parts) > 1 else "未知"
                english_name = parts[2] if len(parts) > 2 else chinese_name
                japanese_name = parts[3] if len(parts) > 3 else ""
                
                role_info[english_name] = {
                    'chinese_name': chinese_name,
                    'english_name': english_name,
                    'japanese_name': japanese_name,
                    'game': game
                }
    
    return role_info


def clean_low_quality_images():
    """清理低质量图片"""
    if not TARGET_DIR.exists():
        print(f"目标目录不存在: {TARGET_DIR}")
        return
    
    # 解析角色列表
    role_info = parse_role_list()
    
    # 统计信息
    stats = {
        'total_files': 0,
        'svg_files': 0,
        'invalid_extensions': 0,
        'too_small_files': 0,
        'too_small_images': 0,
        'corrupted_images': 0,
        'removed_files': 0,
        'kept_files': 0,
    }
    
    # 记录每个角色的清理情况
    role_cleanup = defaultdict(lambda: {
        'total': 0,
        'removed': 0,
        'removed_reasons': defaultdict(int)
    })
    
    # 遍历所有角色目录
    for role_dir in TARGET_DIR.iterdir():
        if not role_dir.is_dir():
            continue
        
        role_name = role_dir.name
        
        # 遍历角色目录中的文件
        for file_path in role_dir.iterdir():
            if not file_path.is_file():
                continue
            
            stats['total_files'] += 1
            role_cleanup[role_name]['total'] += 1
            
            ext = file_path.suffix.lower()
            
            # 检查文件扩展名
            if ext not in VALID_EXTENSIONS:
                if ext == '.svg':
                    stats['svg_files'] += 1
                    reason = 'SVG文件'
                else:
                    stats['invalid_extensions'] += 1
                    reason = '无效扩展名'
                
                file_path.unlink()
                stats['removed_files'] += 1
                role_cleanup[role_name]['removed'] += 1
                role_cleanup[role_name]['removed_reasons'][reason] += 1
                continue
            
            # 检查文件大小
            file_size = file_path.stat().st_size
            if file_size < MIN_FILE_SIZE:
                stats['too_small_files'] += 1
                reason = '文件过小'
                
                file_path.unlink()
                stats['removed_files'] += 1
                role_cleanup[role_name]['removed'] += 1
                role_cleanup[role_name]['removed_reasons'][reason] += 1
                continue
            
            # 检查图片尺寸和质量
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    
                    # 检查图片尺寸
                    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                        stats['too_small_images'] += 1
                        reason = '图片尺寸过小'
                        
                        file_path.unlink()
                        stats['removed_files'] += 1
                        role_cleanup[role_name]['removed'] += 1
                        role_cleanup[role_name]['removed_reasons'][reason] += 1
                        continue
                    
                    # 检查图片是否损坏（尝试加载像素数据）
                    img.verify()
                    
            except Exception as e:
                stats['corrupted_images'] += 1
                reason = f'图片损坏({type(e).__name__})'
                
                file_path.unlink()
                stats['removed_files'] += 1
                role_cleanup[role_name]['removed'] += 1
                role_cleanup[role_name]['removed_reasons'][reason] += 1
                continue
            
            # 保留文件
            stats['kept_files'] += 1
    
    # 输出统计报告
    print("\n" + "="*60)
    print("低质量数据清理完成!")
    print("="*60)
    print(f"扫描文件总数: {stats['total_files']}")
    print(f"删除文件总数: {stats['removed_files']}")
    print(f"保留文件总数: {stats['kept_files']}")
    print("\n删除原因统计:")
    print(f"  - SVG文件: {stats['svg_files']}")
    print(f"  - 无效扩展名: {stats['invalid_extensions']}")
    print(f"  - 文件过小: {stats['too_small_files']}")
    print(f"  - 图片尺寸过小: {stats['too_small_images']}")
    print(f"  - 图片损坏: {stats['corrupted_images']}")
    print("="*60)
    
    # 输出每个角色的清理情况
    print("\n各角色清理情况:")
    print("-" * 60)
    for role_name in sorted(role_cleanup.keys()):
        cleanup = role_cleanup[role_name]
        if cleanup['removed'] > 0:
            info = role_info.get(role_name, {'chinese_name': role_name, 'game': '未知'})
            print(f"{info['chinese_name']} ({role_name}):")
            print(f"  删除 {cleanup['removed']} / {cleanup['total']} 个文件")
            for reason, count in cleanup['removed_reasons'].items():
                print(f"    - {reason}: {count}")
    
    # 生成清理报告
    report_file = PROJECT_ROOT / "docs" / "低质量数据清理报告.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 低质量数据清理报告\n\n")
        f.write(f"**清理时间**: {os.popen('date').read().strip()}\n")
        f.write(f"**数据集位置**: `{TARGET_DIR}`\n\n")
        
        f.write("## 清理统计\n\n")
        f.write(f"- 扫描文件总数: {stats['total_files']}\n")
        f.write(f"- 删除文件总数: {stats['removed_files']}\n")
        f.write(f"- 保留文件总数: {stats['kept_files']}\n")
        f.write(f"- 删除率: {stats['removed_files'] / stats['total_files'] * 100:.2f}%\n\n")
        
        f.write("## 删除原因统计\n\n")
        f.write("| 原因 | 数量 |\n")
        f.write("|------|------|\n")
        f.write(f"| SVG文件 | {stats['svg_files']} |\n")
        f.write(f"| 无效扩展名 | {stats['invalid_extensions']} |\n")
        f.write(f"| 文件过小 | {stats['too_small_files']} |\n")
        f.write(f"| 图片尺寸过小 | {stats['too_small_images']} |\n")
        f.write(f"| 图片损坏 | {stats['corrupted_images']} |\n")
        f.write(f"| **总计** | **{stats['removed_files']}** |\n\n")
        
        f.write("## 各角色清理详情\n\n")
        f.write("| 中文名 | 英文名 | 游戏 | 删除数 | 总数 | 删除率 |\n")
        f.write("|--------|--------|------|--------|------|--------|\n")
        
        for role_name in sorted(role_cleanup.keys()):
            cleanup = role_cleanup[role_name]
            info = role_info.get(role_name, {'chinese_name': role_name, 'game': '未知'})
            delete_rate = cleanup['removed'] / cleanup['total'] * 100 if cleanup['total'] > 0 else 0
            f.write(f"| {info['chinese_name']} | {role_name} | {info['game']} | {cleanup['removed']} | {cleanup['total']} | {delete_rate:.2f}% |\n")
        
        f.write("\n## 清理标准\n\n")
        f.write("- **最小文件大小**: 1 KB\n")
        f.write("- **最小图片尺寸**: 64x64 像素\n")
        f.write("- **有效格式**: JPG, JPEG, PNG, GIF, WebP, BMP\n")
        f.write("- **排除格式**: SVG（矢量图）\n")
        f.write("- **损坏检测**: 尝试加载并验证图片完整性\n")
    
    print(f"\n报告已保存到: {report_file}")
    
    # 统计清理后的数据
    print("\n清理后数据统计:")
    print("-" * 40)
    total_files = 0
    for role_dir in TARGET_DIR.iterdir():
        if role_dir.is_dir():
            file_count = len([f for f in role_dir.iterdir() if f.is_file()])
            if file_count > 0:
                total_files += file_count
                info = role_info.get(role_dir.name, {'chinese_name': role_dir.name})
                print(f"{info['chinese_name']} ({role_dir.name}): {file_count} 张")
    
    print(f"\n总计: {total_files} 张图片")


if __name__ == "__main__":
    clean_low_quality_images()
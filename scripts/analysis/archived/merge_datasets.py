#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 organized_images 和 merged_english_dataset 目录
使用 loli-role.txt 中的标准英文名作为角色名
"""

import os
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect")

# 来源目录
SOURCE_DIRS = [
    PROJECT_ROOT / "data" / "organized_images",
    PROJECT_ROOT / "data" / "merged_english_dataset",
]

# 目标目录
TARGET_DIR = PROJECT_ROOT / "data" / "final_dataset"

# 角色列表文件
ROLE_LIST_FILE = PROJECT_ROOT / "auto_spider_img" / "loli-role.txt"


def parse_role_list():
    """解析角色列表文件，返回标准英文名映射"""
    role_mapping = {}
    
    if not ROLE_LIST_FILE.exists():
        print(f"警告: 角色列表文件不存在: {ROLE_LIST_FILE}")
        return role_mapping
    
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
                
                # 存储标准英文名
                role_mapping[english_name] = {
                    'chinese_name': chinese_name,
                    'english_name': english_name,
                    'japanese_name': japanese_name,
                    'game': game
                }
    
    return role_mapping


def get_file_md5(file_path):
    """计算文件MD5值用于去重"""
    md5_hash = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except:
        return None


def match_role_to_standard(dir_name, role_mapping):
    """将目录名匹配到标准英文名"""
    # 直接匹配
    if dir_name in role_mapping:
        return dir_name
    
    # 尝试清理后匹配
    clean_name = dir_name.replace('_', ' ').replace('-', ' ').strip()
    if clean_name in role_mapping:
        return clean_name
    
    # 尝试小写匹配
    lower_name = dir_name.lower()
    for standard_name in role_mapping:
        if standard_name.lower() == lower_name:
            return standard_name
    
    # 尝试部分匹配（去除空格和特殊字符）
    normalized = ''.join(c.lower() for c in dir_name if c.isalnum())
    for standard_name in role_mapping:
        standard_normalized = ''.join(c.lower() for c in standard_name if c.isalnum())
        if normalized == standard_normalized:
            return standard_name
    
    # 如果都不匹配，返回原目录名
    return dir_name


def merge_datasets():
    """合并两个数据集"""
    # 解析角色列表
    role_mapping = parse_role_list()
    print(f"从角色列表中解析到 {len(role_mapping)} 个标准角色名")
    
    # 创建目标目录
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {
        'total_files_scanned': 0,
        'total_files_copied': 0,
        'duplicates_found': 0,
        'roles_processed': 0,
        'roles_matched': 0,
        'roles_unmatched': 0,
        'errors': 0,
    }
    
    # 用于去重的MD5缓存
    seen_md5 = set()
    
    # 图片扩展名
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'}
    
    # 记录每个角色的文件数
    role_file_counts = defaultdict(int)
    unmatched_roles = set()
    
    # 遍历所有来源目录
    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            print(f"跳过不存在的目录: {source_dir}")
            continue
        
        print(f"\n正在处理目录: {source_dir}")
        
        for role_dir in source_dir.iterdir():
            if not role_dir.is_dir():
                continue
            
            stats['roles_processed'] += 1
            
            # 匹配到标准英文名
            standard_name = match_role_to_standard(role_dir.name, role_mapping)
            
            if standard_name == role_dir.name and standard_name not in role_mapping:
                # 未匹配到标准角色名
                stats['roles_unmatched'] += 1
                unmatched_roles.add(role_dir.name)
                print(f"  未匹配: {role_dir.name}")
            else:
                stats['roles_matched'] += 1
            
            # 创建目标角色目录
            target_role_dir = TARGET_DIR / standard_name
            target_role_dir.mkdir(parents=True, exist_ok=True)
            
            # 遍历角色目录中的文件
            file_count = 0
            for file_path in role_dir.iterdir():
                if not file_path.is_file():
                    continue
                
                ext = file_path.suffix.lower()
                if ext not in valid_extensions:
                    continue
                
                stats['total_files_scanned'] += 1
                
                # 计算MD5去重
                md5 = get_file_md5(file_path)
                if md5 and md5 in seen_md5:
                    stats['duplicates_found'] += 1
                    continue
                
                if md5:
                    seen_md5.add(md5)
                
                # 复制文件到目标目录
                try:
                    target_file = target_role_dir / file_path.name
                    # 如果目标文件已存在但MD5不同，添加序号
                    if target_file.exists():
                        base_name = file_path.stem
                        counter = 1
                        while target_file.exists():
                            target_file = target_role_dir / f"{base_name}_{counter}{ext}"
                            counter += 1
                    
                    shutil.copy2(file_path, target_file)
                    stats['total_files_copied'] += 1
                    file_count += 1
                except Exception as e:
                    stats['errors'] += 1
                    print(f"  复制失败: {file_path.name}, 错误: {e}")
            
            if file_count > 0:
                role_file_counts[standard_name] += file_count
                print(f"  {role_dir.name} -> {standard_name}: {file_count} 个文件")
    
    # 输出统计报告
    print("\n" + "="*60)
    print("数据集合并完成!")
    print("="*60)
    print(f"扫描文件总数: {stats['total_files_scanned']}")
    print(f"成功复制文件: {stats['total_files_copied']}")
    print(f"发现重复文件: {stats['duplicates_found']}")
    print(f"处理角色目录: {stats['roles_processed']}")
    print(f"匹配到标准名: {stats['roles_matched']}")
    print(f"未匹配标准名: {stats['roles_unmatched']}")
    print(f"错误数量: {stats['errors']}")
    print("="*60)
    
    # 输出未匹配的角色
    if unmatched_roles:
        print("\n未匹配到标准英文名的角色:")
        for role_name in sorted(unmatched_roles):
            print(f"  - {role_name}")
    
    # 输出角色文件数量统计
    print("\n各角色图片数量统计:")
    print("-" * 40)
    for role_name, count in sorted(role_file_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{role_name}: {count} 张")
    
    # 保存统计报告
    report_file = PROJECT_ROOT / "docs" / "数据集合并报告.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 数据集合并报告\n\n")
        f.write(f"**合并时间**: {os.popen('date').read().strip()}\n\n")
        f.write("## 统计概览\n\n")
        f.write(f"- 扫描文件总数: {stats['total_files_scanned']}\n")
        f.write(f"- 成功复制文件: {stats['total_files_copied']}\n")
        f.write(f"- 发现重复文件: {stats['duplicates_found']}\n")
        f.write(f"- 处理角色目录: {stats['roles_processed']}\n")
        f.write(f"- 匹配到标准名: {stats['roles_matched']}\n")
        f.write(f"- 未匹配标准名: {stats['roles_unmatched']}\n")
        f.write(f"- 错误数量: {stats['errors']}\n\n")
        
        f.write("## 各角色图片数量\n\n")
        f.write("| 角色 | 图片数量 | 所属游戏 |\n")
        f.write("|------|----------|----------|\n")
        for role_name, count in sorted(role_file_counts.items(), key=lambda x: x[1], reverse=True):
            game = role_mapping.get(role_name, {}).get('game', '未知')
            f.write(f"| {role_name} | {count} | {game} |\n")
        
        if unmatched_roles:
            f.write("\n## 未匹配的角色\n\n")
            for role_name in sorted(unmatched_roles):
                f.write(f"- {role_name}\n")
    
    print(f"\n报告已保存到: {report_file}")
    print(f"合并后的数据集位置: {TARGET_DIR}")


if __name__ == "__main__":
    merge_datasets()

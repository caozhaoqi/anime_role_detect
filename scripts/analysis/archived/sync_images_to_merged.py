#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片数据定时同步脚本
将所有来源的图片按角色整理到 merged_english_dataset 目录
支持定时任务调用
"""

import os
import shutil
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect")

# 来源目录
SOURCE_DIRS = [
    PROJECT_ROOT / "spider_image_system" / "src" / "run" / "data" / "downloaded_images",
    PROJECT_ROOT / "data" / "images",
    PROJECT_ROOT / "data" / "organized_images",
]

# 目标目录
TARGET_DIR = PROJECT_ROOT / "data" / "merged_english_dataset"

# 角色名称映射表（拼音 -> 英文名）
ROLE_MAPPING = {
    'a1luo4na4': 'Arona',
    'a1ni4ya4': 'Anya',
    'ai4li4er3': 'Alice',
    'an1ke3': 'Anke',
    'bai2shang4chui1xue3': 'Shirogane',
    'bu4luo4ni2ya4': 'Bronya',
    'de2li4sha1': 'Theresa',
    'di2ao4na4': 'Diona',
    'duo1li4': 'Dori',
    'fei1mi3li4si1': 'Fimilis',
    'fu2lan2': 'Fran',
    'fu2xuan2': 'Fu Xuan',
    'hei1ta3': 'Herta',
    'hua1huo3': 'Sparkle',
    'ka3qi2na4': 'Kachina',
    'ke3li4': 'Klee',
    'qi1qi1': 'Qiqi',
    'qing1que4': 'Qingque',
    'yaoyao': 'Yaoyao',
    'yaoyao yuan2shen2': 'Yaoyao',
}


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


def get_standard_role_name(role_dir_name):
    """获取标准化的角色名称（优先英文名）"""
    # 先尝试映射表
    name = role_dir_name.lower().strip()
    if name in ROLE_MAPPING:
        return ROLE_MAPPING[name]
    
    # 如果已经是英文名格式（只有字母、空格、下划线）
    clean_name = ''.join(c for c in role_dir_name if c.isalnum() or c in [' ', '_']).strip()
    if clean_name and clean_name == role_dir_name:
        return clean_name.replace(' ', '_')
    
    # 对于拼音格式，保留原样但清理特殊字符
    clean_pinyin = ''.join(c for c in role_dir_name if c.isalnum() or c == '_').strip()
    return clean_pinyin if clean_pinyin else role_dir_name


def sync_images(dry_run=False, verbose=False):
    """同步图片到 merged_english_dataset"""
    # 确保目标目录存在
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {
        'total_files_scanned': 0,
        'total_files_copied': 0,
        'duplicates_found': 0,
        'roles_processed': 0,
        'errors': 0,
    }
    
    # 已存在文件的MD5缓存（用于去重）
    existing_md5 = set()
    
    # 先收集目标目录中已有的文件MD5
    if verbose:
        print("收集已存在文件的MD5...")
    for role_dir in TARGET_DIR.iterdir():
        if not role_dir.is_dir():
            continue
        for file_path in role_dir.iterdir():
            if file_path.is_file():
                md5 = get_file_md5(file_path)
                if md5:
                    existing_md5.add(md5)
    
    if verbose:
        print(f"已存在 {len(existing_md5)} 个文件的MD5记录")
    
    # 图片扩展名
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'}
    
    # 记录每个角色新增的文件数
    role_file_counts = defaultdict(int)
    
    # 遍历所有来源目录
    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            if verbose:
                print(f"跳过不存在的目录: {source_dir}")
            continue
        
        if verbose:
            print(f"\n正在处理目录: {source_dir}")
        
        for role_dir in source_dir.iterdir():
            if not role_dir.is_dir():
                continue
            
            role_name = get_standard_role_name(role_dir.name)
            
            # 创建目标角色目录
            target_role_dir = TARGET_DIR / role_name
            if not dry_run:
                target_role_dir.mkdir(parents=True, exist_ok=True)
            
            stats['roles_processed'] += 1
            
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
                if md5 and md5 in existing_md5:
                    stats['duplicates_found'] += 1
                    if verbose:
                        print(f"  跳过重复: {file_path.name}")
                    continue
                
                if md5:
                    existing_md5.add(md5)
                
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
                    
                    if not dry_run:
                        shutil.copy2(file_path, target_file)
                    stats['total_files_copied'] += 1
                    file_count += 1
                    
                    if verbose:
                        print(f"  复制: {file_path.name} -> {target_file.name}")
                except Exception as e:
                    stats['errors'] += 1
                    print(f"  复制失败: {file_path} -> {target_file}, 错误: {e}")
            
            if file_count > 0:
                role_file_counts[role_name] += file_count
                if verbose:
                    print(f"  {role_dir.name} -> {role_name}: {file_count} 个新文件")
    
    # 输出统计报告
    print("\n" + "="*60)
    print("图片同步完成!")
    print("="*60)
    print(f"扫描文件总数: {stats['total_files_scanned']}")
    print(f"成功复制文件: {stats['total_files_copied']}")
    print(f"发现重复文件: {stats['duplicates_found']}")
    print(f"处理角色目录: {stats['roles_processed']}")
    print(f"错误数量: {stats['errors']}")
    print("="*60)
    
    if role_file_counts:
        print("\n新增文件统计:")
        print("-" * 40)
        for role_name, count in sorted(role_file_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{role_name}: +{count} 个文件")
    
    # 保存统计报告
    report_file = PROJECT_ROOT / "docs" / "图片同步报告.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 图片同步报告\n\n")
        f.write(f"**同步时间**: {os.popen('date').read().strip()}\n")
        f.write(f"**同步模式**: {'模拟运行' if dry_run else '实际运行'}\n\n")
        f.write("## 统计概览\n\n")
        f.write(f"- 扫描文件总数: {stats['total_files_scanned']}\n")
        f.write(f"- 成功复制文件: {stats['total_files_copied']}\n")
        f.write(f"- 发现重复文件: {stats['duplicates_found']}\n")
        f.write(f"- 处理角色目录: {stats['roles_processed']}\n")
        f.write(f"- 错误数量: {stats['errors']}\n\n")
        f.write("## 新增文件统计\n\n")
        f.write("| 角色 | 新增数量 |\n")
        f.write("|------|----------|\n")
        for role_name, count in sorted(role_file_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {role_name} | +{count} |\n")
    
    print(f"\n报告已保存到: {report_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='同步图片到 merged_english_dataset')
    parser.add_argument('--dry-run', '-n', action='store_true', help='模拟运行，不实际复制文件')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    args = parser.parse_args()
    
    sync_images(dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()

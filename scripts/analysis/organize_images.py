#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片数据整理脚本
将多个目录的图片合并到统一位置，去除重复，统一命名规范
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
    PROJECT_ROOT / "data" / "merged_english_dataset",
    PROJECT_ROOT / "spider_image_system" / "src" / "run" / "data" / "downloaded_images",
]

# 目标目录
TARGET_DIR = PROJECT_ROOT / "data" / "organized_images"

# 角色名称映射表（拼音 -> 英文名）
ROLE_MAPPING = {
    'a1luo4na4': 'Arona',
    'a1ni4ya4': 'Anya',
    'ai4li4er3': 'Alice',
    'an1ka3xi1ya3': 'Aneka',
    'an1ke3': 'Anke',
    'bai2shang4chui1xue3': 'Shirogane',
    'bu4luo4ni2ya4': 'Bronya',
    'cong2yu3': 'Congyu',
    'de2li4sha1': 'Theresa',
    'di2ao4na4': 'Diona',
    'duo1li4': 'Dori',
    'fei1mi3li4si1': 'Fimilis',
    'fei1xie4er3': 'Feixieer',
    'fu2lan2': 'Fran',
    'fu2li4xi1ya4': 'Furixiya',
    'fu2xuan2': 'Fu Xuan',
    'gu3ming2di4lian4': 'Gumingdiliana',
    'hei1ta3': 'Herta',
    'hua1huo3': 'Sparkle',
    'ka3qi2na4': 'Kachina',
    'kai3lu4': 'Kai Lu',
    'ke3lin2': 'Kelin',
    'ke3li4': 'Klee',
    'qi1qi1': 'Qiqi',
    'qing1que4': 'Qingque',
    'sha1wu4': 'Shawu',
    'shen1yue4': 'Shen Yue',
    'xia4ke4li3': 'Xia Keli',
    'xue4xiao3ban3': 'Xue Xiaoban',
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
    """获取标准化的角色名称"""
    # 先尝试映射表
    name = role_dir_name.lower().strip()
    if name in ROLE_MAPPING:
        return ROLE_MAPPING[name]
    
    # 移除特殊字符和空格
    clean_name = ''.join(c for c in role_dir_name if c.isalnum() or c in [' ', '_']).strip()
    clean_name = clean_name.replace(' ', '_')
    
    # 如果是拼音形式，尝试转换
    if name != clean_name:
        return clean_name
    
    return role_dir_name


def organize_images():
    """整理图片数据"""
    # 创建目标目录
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {
        'total_files_scanned': 0,
        'total_files_copied': 0,
        'duplicates_found': 0,
        'roles_processed': 0,
        'roles_created': 0,
        'errors': 0,
    }
    
    # 用于去重的MD5缓存
    seen_md5 = set()
    
    # 图片扩展名
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'}
    
    # 记录每个角色的文件数
    role_file_counts = defaultdict(int)
    
    # 遍历所有来源目录
    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            print(f"跳过不存在的目录: {source_dir}")
            continue
        
        print(f"\n正在处理目录: {source_dir}")
        
        for role_dir in source_dir.iterdir():
            if not role_dir.is_dir():
                continue
            
            role_name = get_standard_role_name(role_dir.name)
            
            # 创建目标角色目录
            target_role_dir = TARGET_DIR / role_name
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
                    print(f"复制失败: {file_path} -> {target_file}, 错误: {e}")
            
            role_file_counts[role_name] += file_count
            print(f"  {role_dir.name} -> {role_name}: {file_count} 个文件")
    
    # 统计创建的角色目录数
    stats['roles_created'] = len(list(TARGET_DIR.iterdir()))
    
    # 输出统计报告
    print("\n" + "="*60)
    print("图片整理完成!")
    print("="*60)
    print(f"扫描文件总数: {stats['total_files_scanned']}")
    print(f"成功复制文件: {stats['total_files_copied']}")
    print(f"发现重复文件: {stats['duplicates_found']}")
    print(f"处理角色目录: {stats['roles_processed']}")
    print(f"创建目标目录: {stats['roles_created']}")
    print(f"错误数量: {stats['errors']}")
    print("="*60)
    
    # 输出角色文件数量统计
    print("\n各角色图片数量统计:")
    print("-" * 40)
    for role_name, count in sorted(role_file_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{role_name}: {count} 张")
    
    # 保存统计报告
    report_file = PROJECT_ROOT / "docs" / "图片整理报告.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 图片数据整理报告\n\n")
        f.write(f"**整理时间**: {os.popen('date').read().strip()}\n\n")
        f.write("## 统计概览\n\n")
        f.write(f"- 扫描文件总数: {stats['total_files_scanned']}\n")
        f.write(f"- 成功复制文件: {stats['total_files_copied']}\n")
        f.write(f"- 发现重复文件: {stats['duplicates_found']}\n")
        f.write(f"- 处理角色目录: {stats['roles_processed']}\n")
        f.write(f"- 创建目标目录: {stats['roles_created']}\n")
        f.write(f"- 错误数量: {stats['errors']}\n\n")
        f.write("## 各角色图片数量\n\n")
        f.write("| 角色 | 图片数量 |\n")
        f.write("|------|----------|\n")
        for role_name, count in sorted(role_file_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {role_name} | {count} |\n")
    
    print(f"\n报告已保存到: {report_file}")


if __name__ == "__main__":
    organize_images()

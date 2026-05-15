#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计所有角色的URL下载图片数据并生成报告
"""

import os
import json

def count_urls_in_file(file_path):
    """统计文件中的URL数量"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            urls = [line.strip() for line in lines if line.strip()]
            return len(urls)
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return 0

def count_images_in_dir(dir_path):
    """统计目录中的图片数量"""
    if not os.path.exists(dir_path):
        return 0
    count = 0
    for f in os.listdir(dir_path):
        if f.lower().endswith('.jpg'):
            count += 1
    return count

def parse_role_list(role_list_path):
    """解析角色列表"""
    roles = {}
    with open(role_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                chinese_name = parts[0]
                english_name = parts[2]
                roles[english_name] = chinese_name
    return roles

def generate_report():
    """生成统计报告"""
    # 配置路径
    role_list_path = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'
    img_url_dirs = [
        '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url',
        '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url_english'
    ]
    dataset_path = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
    output_path = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/url_download_stats.md'
    
    # 解析角色列表
    roles = parse_role_list(role_list_path)
    
    # 统计URL
    url_stats = {}
    for img_url_dir in img_url_dirs:
        if not os.path.exists(img_url_dir):
            continue
        for filename in os.listdir(img_url_dir):
            if filename.endswith('_img.txt'):
                role_name = filename[:-8]  # 去掉 '_img.txt'
                file_path = os.path.join(img_url_dir, filename)
                url_count = count_urls_in_file(file_path)
                
                if role_name not in url_stats:
                    url_stats[role_name] = 0
                url_stats[role_name] += url_count
    
    # 统计已下载图片
    download_stats = {}
    if os.path.exists(dataset_path):
        for dir_name in os.listdir(dataset_path):
            dir_path = os.path.join(dataset_path, dir_name)
            if os.path.isdir(dir_path):
                img_count = count_images_in_dir(dir_path)
                download_stats[dir_name] = img_count
    
    # 生成报告
    report = []
    report.append("# 角色URL下载图片数据统计报告")
    report.append("")
    report.append("## 概览")
    report.append("")
    
    total_urls = sum(url_stats.values())
    total_images = sum(download_stats.values())
    total_roles = len(roles)
    
    report.append(f"- **角色总数**: {total_roles} 个")
    report.append(f"- **URL总数**: {total_urls} 条")
    report.append(f"- **已下载图片**: {total_images} 张")
    report.append(f"- **平均下载率**: {((total_images / total_urls) * 100) if total_urls > 0 else 0:.2f}%")
    report.append("")
    
    # 详细统计表格
    report.append("## 详细统计")
    report.append("")
    report.append("| 英文角色名 | 中文角色名 | URL数量 | 已下载图片 | 下载率 |")
    report.append("|-----------|-----------|--------|-----------|--------|")
    
    for english_name in sorted(roles.keys()):
        chinese_name = roles[english_name]
        url_count = url_stats.get(english_name, 0)
        img_count = download_stats.get(english_name, 0)
        download_rate = ((img_count / url_count) * 100) if url_count > 0 else 0
        
        report.append(f"| {english_name} | {chinese_name} | {url_count} | {img_count} | {download_rate:.1f}% |")
    
    report.append("")
    
    # URL文件列表（拼音格式）
    report.append("## URL文件列表（拼音格式）")
    report.append("")
    report.append("| 拼音文件名 | URL数量 |")
    report.append("|-----------|--------|")
    
    for pinyin_name in sorted(url_stats.keys()):
        if pinyin_name not in roles:
            url_count = url_stats[pinyin_name]
            report.append(f"| {pinyin_name} | {url_count} |")
    
    report.append("")
    
    # 未找到URL的角色
    report.append("## 未找到URL的角色")
    report.append("")
    missing_roles = []
    for english_name in roles:
        if english_name not in url_stats:
            missing_roles.append(f"- {english_name} ({roles[english_name]})")
    
    if missing_roles:
        report.extend(missing_roles)
    else:
        report.append("- 无")
    
    # 写入报告文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"报告已生成: {output_path}")
    print(f"角色总数: {total_roles}")
    print(f"URL总数: {total_urls}")
    print(f"已下载图片: {total_images}")

if __name__ == "__main__":
    generate_report()

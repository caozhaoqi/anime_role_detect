#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计已有角色URL数量并生成汇总报告
"""

import os
import sys
from pypinyin import lazy_pinyin, Style

# 配置
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'
ROLE_LIST_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'
OUTPUT_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/data_collection/downloaders/url_statistics_summary.txt'


def get_all_roles():
    """获取完整角色列表"""
    roles = []
    with open(ROLE_LIST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ')
                chinese_name = parts[0]
                roles.append(chinese_name)
    return roles


def get_pinyin(name):
    """获取角色拼音（用于匹配文件名）"""
    return ''.join(lazy_pinyin(name, style=Style.TONE3))


def count_urls():
    """统计每个角色的URL数量"""
    # 获取所有角色
    all_roles = get_all_roles()
    
    # 获取已有的URL文件
    existing_files = {}
    if os.path.exists(URL_DIR):
        for filename in os.listdir(URL_DIR):
            if filename.endswith('_img.txt'):
                role_pinyin = filename.replace('_img.txt', '')
                existing_files[role_pinyin] = filename
    
    # 统计每个角色的URL数量
    role_url_counts = []
    missing_roles = []
    
    for role in all_roles:
        pinyin = get_pinyin(role)
        url_file = os.path.join(URL_DIR, f"{pinyin}_img.txt")
        
        if os.path.exists(url_file):
            with open(url_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                count = len([l for l in lines if l.strip()])
            role_url_counts.append({
                'role': role,
                'pinyin': pinyin,
                'count': count
            })
        else:
            missing_roles.append(role)
    
    # 按URL数量降序排序
    role_url_counts.sort(key=lambda x: x['count'], reverse=True)
    
    return role_url_counts, missing_roles


def generate_report(role_url_counts, missing_roles):
    """生成汇总报告"""
    total_urls = sum(item['count'] for item in role_url_counts)
    avg_urls = round(total_urls / len(role_url_counts), 1) if role_url_counts else 0
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("        角色URL统计汇总报告")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append(f"统计时间: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"总角色数: {len(role_url_counts) + len(missing_roles)}")
    report_lines.append(f"已采集角色: {len(role_url_counts)}")
    report_lines.append(f"缺失角色: {len(missing_roles)}")
    report_lines.append(f"总URL数: {total_urls:,}")
    report_lines.append(f"平均每个角色URL数: {avg_urls}")
    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("   URL数量排名（降序）")
    report_lines.append("=" * 60)
    report_lines.append(f"{'排名':<6} {'角色':<10} {'拼音':<20} {'URL数量':>10}")
    report_lines.append("-" * 60)
    
    for i, item in enumerate(role_url_counts, 1):
        report_lines.append(f"{i:<6} {item['role']:<10} {item['pinyin']:<20} {item['count']:>10}")
    
    if missing_roles:
        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("   缺失角色列表")
        report_lines.append("=" * 60)
        for role in missing_roles:
            report_lines.append(f"  • {role}")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("                    报告结束")
    report_lines.append("=" * 60)
    
    return '\n'.join(report_lines)


def main():
    print("=== 开始统计角色URL ===")
    
    # 统计URL数量
    role_url_counts, missing_roles = count_urls()
    
    # 生成报告
    report = generate_report(role_url_counts, missing_roles)
    
    # 打印报告
    print(report)
    
    # 输出到文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 报告已保存到: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
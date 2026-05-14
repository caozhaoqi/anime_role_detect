#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色URL和图片分布统计报告生成器
基于 loli-role.txt 中的核心角色列表统计URL和图片分布
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect")

# 数据目录
SPIDER_DATA_DIR = PROJECT_ROOT / "spider_image_system" / "data"
IMG_URL_DIR = SPIDER_DATA_DIR / "img_url"
IMG_URL_ENGLISH_DIR = SPIDER_DATA_DIR / "img_url_english"
HREF_URL_DIR = SPIDER_DATA_DIR / "href_url"
CONFIG_DIR = SPIDER_DATA_DIR / "config"

# 角色列表文件
ROLE_LIST_FILE = PROJECT_ROOT / "auto_spider_img" / "loli-role.txt"

# 图片存储目录
IMAGE_DIRS = [
    PROJECT_ROOT / "data" / "images",
    PROJECT_ROOT / "data" / "organized_images",
    PROJECT_ROOT / "data" / "merged_english_dataset",
    PROJECT_ROOT / "spider_image_system" / "data" / "images",
    PROJECT_ROOT / "spider_image_system" / "src" / "run" / "data" / "downloaded_images",
]


def parse_role_list():
    """解析角色列表文件，返回角色信息字典"""
    roles = {}
    
    if not ROLE_LIST_FILE.exists():
        print(f"警告: 角色列表文件不存在: {ROLE_LIST_FILE}")
        return roles
    
    with open(ROLE_LIST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                chinese_name = parts[0]
                game = parts[1] if len(parts) > 1 else "未知"
                english_name = parts[2] if len(parts) > 2 else chinese_name
                japanese_name = parts[3] if len(parts) > 3 else ""
                
                # 生成可能的role_key（拼音或英文小写）
                role_key = english_name.lower().replace(' ', '_').replace('-', '_')
                
                roles[role_key] = {
                    'chinese_name': chinese_name,
                    'english_name': english_name,
                    'japanese_name': japanese_name,
                    'game': game,
                    'role_key': role_key
                }
    
    return roles


def count_lines_in_file(file_path):
    """统计文件行数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return len([line for line in f if line.strip()])
    except Exception:
        return 0


def find_url_files(role_key):
    """查找角色的URL文件并统计"""
    stats = {
        'img_url_count': 0,
        'href_url_count': 0,
        'total_urls': 0
    }
    
    # 查找 img_url 文件
    img_file = IMG_URL_DIR / f"{role_key}_img.txt"
    if img_file.exists():
        stats['img_url_count'] += count_lines_in_file(img_file)
    
    # 查找 img_url_english 文件
    img_english_file = IMG_URL_ENGLISH_DIR / f"{role_key}_img.txt"
    if img_english_file.exists():
        stats['img_url_count'] += count_lines_in_file(img_english_file)
    
    # 查找 href_url 文件
    href_file = HREF_URL_DIR / f"{role_key}_url.txt"
    if href_file.exists():
        stats['href_url_count'] += count_lines_in_file(href_file)
    
    href_result_file = HREF_URL_DIR / f"{role_key}_result_url.txt"
    if href_result_file.exists():
        stats['href_url_count'] += count_lines_in_file(href_result_file)
    
    stats['total_urls'] = stats['img_url_count'] + stats['href_url_count']
    
    return stats


def count_images_for_role(role_key, english_name=None):
    """统计角色的图片数量，支持多种目录命名方式"""
    total_images = 0
    searched_dirs = set()
    
    # 可能的目录名列表
    possible_names = [role_key]
    if english_name:
        possible_names.append(english_name)
        # 处理英文名中的空格和特殊字符
        possible_names.append(english_name.replace(' ', '_').replace('-', '_'))
        possible_names.append(english_name.replace(' ', '').replace('-', ''))
    
    for image_dir in IMAGE_DIRS:
        if not image_dir.exists():
            continue
        
        for name in possible_names:
            if name in searched_dirs:
                continue
            searched_dirs.add(name)
            
            role_dir = image_dir / name
            if role_dir.exists() and role_dir.is_dir():
                image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'}
                try:
                    count = sum(1 for f in role_dir.iterdir() 
                               if f.is_file() and f.suffix.lower() in image_extensions)
                    total_images += count
                except Exception:
                    pass
    
    return total_images


def collect_role_stats(roles):
    """收集所有角色的URL统计信息"""
    role_stats = {}
    
    for role_key, role_info in roles.items():
        url_stats = find_url_files(role_key)
        image_count = count_images_for_role(role_key, role_info['english_name'])
        
        role_stats[role_key] = {
            'name': role_info['chinese_name'],
            'english_name': role_info['english_name'],
            'japanese_name': role_info['japanese_name'],
            'game': role_info['game'],
            'img_url_count': url_stats['img_url_count'],
            'href_url_count': url_stats['href_url_count'],
            'total_urls': url_stats['total_urls'],
            'image_count': image_count
        }
    
    return role_stats


def generate_markdown_report(role_stats):
    """生成Markdown格式的报告"""
    report_lines = []
    
    # 报告标题
    report_lines.append("# 项目角色URL和图片分布统计报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**数据来源**: `{ROLE_LIST_FILE}`")
    report_lines.append("")
    
    # 统计概览
    total_roles = len(role_stats)
    total_urls = sum(stats['total_urls'] for stats in role_stats.values())
    total_img_urls = sum(stats['img_url_count'] for stats in role_stats.values())
    total_href_urls = sum(stats['href_url_count'] for stats in role_stats.values())
    total_images = sum(stats['image_count'] for stats in role_stats.values())
    
    # 统计有数据的角色
    roles_with_data = sum(1 for s in role_stats.values() if s['total_urls'] > 0)
    roles_without_data = total_roles - roles_with_data
    
    report_lines.append("## 📊 统计概览")
    report_lines.append("")
    report_lines.append(f"- **角色总数**: {total_roles}")
    report_lines.append(f"- **有URL数据的角色**: {roles_with_data}")
    report_lines.append(f"- **无URL数据的角色**: {roles_without_data}")
    report_lines.append(f"- **URL总数**: {total_urls}")
    report_lines.append(f"  - 图片URL (img_url): {total_img_urls}")
    report_lines.append(f"  - 作品URL (href_url): {total_href_urls}")
    report_lines.append(f"- **已下载图片总数**: {total_images}")
    report_lines.append("")
    
    # 按URL数量排序
    sorted_roles = sorted(role_stats.items(), key=lambda x: x[1]['total_urls'], reverse=True)
    
    # 详细统计表格
    report_lines.append("## 📋 角色详细统计")
    report_lines.append("")
    report_lines.append("| 序号 | 角色(中文) | 角色(英文) | 所属游戏 | 图片URL | 作品URL | URL总数 | 已下载图片 | 状态 |")
    report_lines.append("|------|------------|------------|----------|---------|---------|---------|------------|------|")
    
    for idx, (role_key, stats) in enumerate(sorted_roles, 1):
        name = stats['name']
        english = stats['english_name']
        game = stats['game']
        img_count = stats['img_url_count']
        href_count = stats['href_url_count']
        total = stats['total_urls']
        images = stats['image_count']
        
        # 状态判断
        if total >= 200:
            status = "✅ 充足"
        elif total >= 100:
            status = "🟡 良好"
        elif total >= 50:
            status = "🟠 一般"
        elif total > 0:
            status = "🔴 不足"
        else:
            status = "⚪ 无数据"
        
        report_lines.append(f"| {idx} | {name} | {english} | {game} | {img_count} | {href_count} | {total} | {images} | {status} |")
    
    report_lines.append("")
    
    # 按游戏分组统计
    report_lines.append("## 🎮 按游戏分组统计")
    report_lines.append("")
    
    game_stats = defaultdict(lambda: {'count': 0, 'total_urls': 0, 'total_images': 0})
    for stats in role_stats.values():
        game = stats['game']
        game_stats[game]['count'] += 1
        game_stats[game]['total_urls'] += stats['total_urls']
        game_stats[game]['total_images'] += stats['image_count']
    
    # 按角色数量排序
    sorted_games = sorted(game_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    report_lines.append("| 游戏 | 角色数 | URL总数 | 图片总数 |")
    report_lines.append("|------|--------|---------|----------|")
    
    for game, stats in sorted_games:
        report_lines.append(f"| {game} | {stats['count']} | {stats['total_urls']} | {stats['total_images']} |")
    
    report_lines.append("")
    
    # 分布统计
    report_lines.append("## 📈 URL数量分布")
    report_lines.append("")
    
    ranges = [
        (200, float('inf'), "200+", "充足"),
        (100, 199, "100-199", "良好"),
        (50, 99, "50-99", "一般"),
        (1, 49, "1-49", "不足"),
        (0, 0, "0", "无数据")
    ]
    
    report_lines.append("| URL数量范围 | 角色数量 | 占比 | 状态 |")
    report_lines.append("|-------------|----------|------|------|")
    
    for min_val, max_val, label, status_text in ranges:
        if max_val == float('inf'):
            count = sum(1 for s in role_stats.values() if s['total_urls'] >= min_val)
        elif min_val == max_val:
            count = sum(1 for s in role_stats.values() if s['total_urls'] == min_val)
        else:
            count = sum(1 for s in role_stats.values() if min_val <= s['total_urls'] <= max_val)
        percentage = (count / total_roles * 100) if total_roles > 0 else 0
        report_lines.append(f"| {label} | {count} | {percentage:.1f}% | {status_text} |")
    
    report_lines.append("")
    
    # 需要关注的角色（URL数量少于100的）
    report_lines.append("## ⚠️ 需要关注的角色（URL < 100）")
    report_lines.append("")
    
    low_url_roles = [(k, v) for k, v in sorted_roles if 0 < v['total_urls'] < 100]
    no_data_roles = [(k, v) for k, v in sorted_roles if v['total_urls'] == 0]
    
    if low_url_roles:
        report_lines.append("### URL不足100的角色")
        report_lines.append("")
        report_lines.append("| 角色(中文) | 角色(英文) | 所属游戏 | 当前URL数 | 建议操作 |")
        report_lines.append("|------------|------------|----------|-----------|----------|")
        
        for role_key, stats in low_url_roles:
            name = stats['name']
            english = stats['english_name']
            game = stats['game']
            count = stats['total_urls']
            needed = 100 - count
            report_lines.append(f"| {name} | {english} | {game} | {count} | 需补充 {needed} 个URL |")
        
        report_lines.append("")
    
    if no_data_roles:
        report_lines.append("### 无URL数据的角色")
        report_lines.append("")
        report_lines.append("| 角色(中文) | 角色(英文) | 所属游戏 | 建议操作 |")
        report_lines.append("|------------|------------|----------|----------|")
        
        for role_key, stats in no_data_roles:
            name = stats['name']
            english = stats['english_name']
            game = stats['game']
            report_lines.append(f"| {name} | {english} | {game} | 需采集URL |")
        
        report_lines.append("")
    
    # 数据存储位置说明
    report_lines.append("## 📁 数据存储位置")
    report_lines.append("")
    report_lines.append("### URL文件位置")
    report_lines.append(f"- **图片URL**: `{IMG_URL_DIR}`")
    report_lines.append(f"- **英文图片URL**: `{IMG_URL_ENGLISH_DIR}`")
    report_lines.append(f"- **作品URL**: `{HREF_URL_DIR}`")
    report_lines.append("")
    report_lines.append("### 配置文件位置")
    report_lines.append(f"- **角色配置**: `{CONFIG_DIR}`")
    report_lines.append("")
    
    # 文件命名规则
    report_lines.append("## 📝 文件命名规则")
    report_lines.append("")
    report_lines.append("- 图片URL文件: `{role_key}_img.txt`")
    report_lines.append("- 作品URL文件: `{role_key}_url.txt` 或 `{role_key}_result_url.txt`")
    report_lines.append("- 配置文件: `{role_key}_config.json`")
    report_lines.append("")
    
    return "\n".join(report_lines)


def main():
    """主函数"""
    print("开始统计角色URL和图片分布...")
    print(f"角色列表文件: {ROLE_LIST_FILE}")
    
    # 解析角色列表
    roles = parse_role_list()
    print(f"从角色列表中解析到 {len(roles)} 个角色")
    
    # 收集URL统计
    role_stats = collect_role_stats(roles)
    
    # 生成报告
    print("生成Markdown报告...")
    report = generate_markdown_report(role_stats)
    
    # 保存报告
    output_file = PROJECT_ROOT / "docs" / "角色URL和图片分布统计报告.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存到: {output_file}")
    
    # 输出简要统计
    total_roles = len(role_stats)
    roles_with_data = sum(1 for s in role_stats.values() if s['total_urls'] > 0)
    total_urls = sum(s['total_urls'] for s in role_stats.values())
    
    print(f"\n{'='*60}")
    print(f"统计摘要:")
    print(f"  - 角色总数: {total_roles}")
    print(f"  - 有URL数据的角色: {roles_with_data}")
    print(f"  - 无URL数据的角色: {total_roles - roles_with_data}")
    print(f"  - URL总数: {total_urls}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

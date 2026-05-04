#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析角色图片数目分布"""

import os
from pathlib import Path
from collections import defaultdict

ORGANIZED_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images')

# 角色拼音到中文名的映射
PINYIN_TO_CHINESE = {
    'a1luo4na4': '阿洛娜', 'pu3la1na4': '普拉娜', 'na4xi1da2': '纳西妲', 'ke3li4': '可莉',
    'di2ao4na4': '迪奥娜', 'yao2yao2': '瑶瑶', 'hei1ta3': '黑塔', 'fu2xuan2': '符玄',
    'qi1qi1': '七七', 'zao3you4': '早柚', 'duo1li4': '多莉', 'ka3qi2na4': '卡齐娜',
    'san1yue4qi1': '三月七', 'hua1huo3': '花火', 'yin2lang2': '银狼', 'lei2mu3': '雷姆',
    'la1mu3': '拉姆', 'ti2bao3': '缇宝', 'wei2li3nai4': '维里奈', 'an1ke3': '安可',
    'you4hu2': '釉壶', 'luo4ke3ke3': '洛可可', 'lu4mu4yuan2': '鹿目圆', 'xiao3mei3yan4': '晓美焰',
    'xue4xiao3ban3': '血小板', 'kang1na4': '康娜', 'si4mi4nai3': '四糸乃', 'kai3lu4': '凯露',
    'ke4luo2luo2': '克萝萝', 'xiao3shan3': '小闪', 'yi1li4ya3': '伊莉雅', 'ren3ye3ren3': '忍野忍',
    'zhi4nai3': '智乃', 'xiao3mai2': '小埋', 'sha1wu4': '纱雾', 'mao1gong1you4nai4': '猫宫又奈',
    'de2li4sha1': '德丽莎', 'bu4luo4ni2ya4': '布洛妮娅', 'ke3lin2': '可琳', 'ai4li4er3': '爱丽儿',
    'shen2le4': '神乐', 'bai2shang4chui1xue3': '白上吹雪', 'yue4qian1ye4': '月千夜',
    'fu2li4xi1ya4': '芙丽希娅', 'li4ta3la1': '莉塔拉', 'wei2pu3lei3': '维普蕾', 'xia4ke4li3': '夏克里',
    'na4gan1': '纳甘', 'ke1xie4ni2ya4': '科谢尼娅', 'qi2ta3': '奇塔', 'kou4er3fu2': '寇尔芙',
    'ke4luo2li4ke1': '克罗丽科', 'pei4li3ti2ya4': '佩里缇亚', 'a1ni4ya4': '阿尼亚', 'luo4qian4': '洛茜',
    'mi3dou4zi5': '祢豆子', 'xi1er3': '希儿', 'xing4': '杏', 'yi1se4lin2': '伊瑟琳',
    'fu2lan2': '芙兰', 'fei1mi3li4si1': '菲米莉丝', 'tian1tong2ai4li4si1': '天童爱丽丝',
    'xi1ge2wen2': '希格雯', 'lei3bei4': '蕾贝', 'zao3wu4': '早雾',
}

def get_role_stats():
    """获取每个角色的图片数量"""
    stats = []
    for dir_path in ORGANIZED_DIR.iterdir():
        if dir_path.is_dir():
            role_name = dir_path.name
            img_count = len(list(dir_path.glob('*')))
            chinese_name = PINYIN_TO_CHINESE.get(role_name, role_name)
            stats.append((role_name, chinese_name, img_count))
    return sorted(stats, key=lambda x: x[2], reverse=True)

def main():
    print("=" * 70)
    print("📊 角色图片数目分布分析")
    print("=" * 70)
    
    stats = get_role_stats()
    
    if not stats:
        print("未找到任何角色目录")
        return
    
    total_images = sum(s[2] for s in stats)
    total_roles = len(stats)
    
    print(f"\n📁 总计: {total_roles} 个角色, {total_images:,} 张图片")
    print(f"📊 平均: {total_images // total_roles} 张/角色")
    
    # 按数量区间统计
    ranges = [
        (0, 50),
        (50, 100),
        (100, 200),
        (200, 300),
        (300, 500),
        (500, float('inf'))
    ]
    
    range_labels = [
        "0-49",
        "50-99",
        "100-199",
        "200-299",
        "300-499",
        "500+"
    ]
    
    range_counts = [0] * len(ranges)
    range_details = [[] for _ in ranges]
    
    for role_name, chinese_name, count in stats:
        for i, (low, high) in enumerate(ranges):
            if low <= count < high:
                range_counts[i] += 1
                range_details[i].append((chinese_name, count))
                break
    
    print("\n" + "=" * 70)
    print("📈 图片数量区间分布")
    print("=" * 70)
    
    for i, label in enumerate(range_labels):
        percent = (range_counts[i] / total_roles * 100) if total_roles > 0 else 0
        print(f"{label:<10} {range_counts[i]:3d} 个角色 ({percent:5.1f}%)")
    
    # 显示每个区间的角色详情
    print("\n" + "=" * 70)
    print("📋 各区间角色详情")
    print("=" * 70)
    
    for i, label in enumerate(range_labels):
        if range_details[i]:
            print(f"\n🔹 {label} 区间 ({len(range_details[i])} 个角色):")
            for chinese_name, count in sorted(range_details[i], key=lambda x: x[1]):
                print(f"   {chinese_name:12} {count:4d} 张")
    
    # 显示TOP 10
    print("\n" + "=" * 70)
    print("🏆 图片数量TOP 10")
    print("=" * 70)
    print(f"{'排名':<6} {'中文名':<12} {'拼音':<20} {'图片数':<8}")
    print("-" * 70)
    for i, (role_name, chinese_name, count) in enumerate(stats[:10], 1):
        print(f"{i:<6} {chinese_name:<12} {role_name:<20} {count:<8}")
    
    # 显示最少的10个
    print("\n" + "=" * 70)
    print("⚠️  图片数量最少的10个角色")
    print("=" * 70)
    print(f"{'排名':<6} {'中文名':<12} {'拼音':<20} {'图片数':<8}")
    print("-" * 70)
    for i, (role_name, chinese_name, count) in enumerate(reversed(stats[-10:]), 1):
        print(f"{i:<6} {chinese_name:<12} {role_name:<20} {count:<8}")
    
    # 统计达标情况
    sufficient = sum(1 for _, _, cnt in stats if cnt >= 100)
    insufficient = sum(1 for _, _, cnt in stats if cnt < 100)
    
    print("\n" + "=" * 70)
    print("✅ 达标情况统计")
    print("=" * 70)
    print(f"图片数 >= 100: {sufficient} 个角色 ({sufficient/total_roles*100:.1f}%)")
    print(f"图片数 < 100:  {insufficient} 个角色 ({insufficient/total_roles*100:.1f}%)")

if __name__ == '__main__':
    main()

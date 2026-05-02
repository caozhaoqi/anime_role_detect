#!/usr/bin/env python3
"""分析loli-role.txt名单与当前采集状态对比"""
import os
import sys
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from spider_image_system.src.run.constants import PINYIN_MAPPING, get_pinyin

def parse_loli_role_file(file_path):
    """解析loli-role.txt文件"""
    roles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                chinese_name = parts[0]
                game = parts[1] if len(parts) > 1 else ""
                roles.append({
                    'chinese': chinese_name,
                    'game': game,
                    'pinyin': get_pinyin(chinese_name)
                })
    return roles

def count_urls_from_files():
    """从img_url目录统计URL数量"""
    img_url_dir = project_root / "spider_image_system" / "data" / "img_url"
    if not img_url_dir.exists():
        return {}

    role_counts = {}
    for file in img_url_dir.glob("*_img.txt"):
        pinyin_name = file.stem.replace("_img", "")
        try:
            with open(file, 'r', encoding='utf-8') as f:
                count = len([line for line in f if line.strip()])
            role_counts[pinyin_name] = count
        except Exception:
            role_counts[pinyin_name] = 0
    return role_counts

def main():
    print("=" * 80)
    print(" Loli-Role.txt 名单采集状态分析")
    print("=" * 80)

    # 解析loli-role.txt
    loli_role_file = project_root / "auto_spider_img" / "loli-role.txt"
    if not loli_role_file.exists():
        print(f"❌ 文件不存在: {loli_role_file}")
        return

    roles = parse_loli_role_file(loli_role_file)
    print(f"\n📋 Loli-Role.txt 名单总数: {len(roles)} 个角色")

    # 统计URL数量
    url_counts = count_urls_from_files()

    # 分析每个角色
    collected = []
    not_collected = []
    insufficient = []  # < 200

    print("\n" + "=" * 80)
    print(" 📊 采集状态详情")
    print("=" * 80)

    print(f"\n{'序号':<4} {'角色名':<15} {'拼音':<20} {'游戏':<15} {'URL数':>8} {'状态':<10}")
    print("-" * 80)

    for idx, role in enumerate(roles, 1):
        chinese = role['chinese']
        pinyin = role['pinyin']
        game = role['game']

        # 查找URL数量
        count = url_counts.get(pinyin, 0)
        if count == 0:
            # 尝试通过中文名查找
            for py, cn in PINYIN_MAPPING.items():
                if cn == chinese or cn == pinyin:
                    count = url_counts.get(py, 0)
                    break

        if count >= 200:
            status = "✅充足"
            collected.append((chinese, pinyin, game, count))
        elif count > 0:
            status = "⚠️不足"
            insufficient.append((chinese, pinyin, game, count))
        else:
            status = "❌未采集"
            not_collected.append((chinese, pinyin, game, count))

        print(f"{idx:<4} {chinese:<15} {pinyin:<20} {game:<15} {count:>8} {status:<10}")

    # 汇总统计
    print("\n" + "=" * 80)
    print(" 📈 采集汇总")
    print("=" * 80)

    total = len(roles)
    print(f"\n  总角色数:     {total}")
    print(f"  ✅ 已采集充足: {len(collected)} ({len(collected)/total*100:.1f}%)")
    print(f"  ⚠️ 需要补充:   {len(insufficient)} ({len(insufficient)/total*100:.1f}%)")
    print(f"  ❌ 未采集:     {len(not_collected)} ({len(not_collected)/total*100:.1f}%)")

    # 未采集角色
    if not_collected:
        print("\n" + "=" * 80)
        print(" 🚨 未采集角色（需要立即采集）")
        print("=" * 80)
        for chinese, pinyin, game, count in sorted(not_collected, key=lambda x: x[1]):
            print(f"  {chinese:<15} {pinyin:<20} {game}")

    # 需要补充的角色
    if insufficient:
        print("\n" + "=" * 80)
        print(" ⚠️ 需要补充采集的角色（<200 URL）")
        print("=" * 80)
        for chinese, pinyin, game, count in sorted(insufficient, key=lambda x: x[3], reverse=True):
            bar = "█" * min(count // 10, 20)
            print(f"  {chinese:<15} {count:>5} │{bar}")

    # 生成待采集列表
    need_collection = not_collected + insufficient
    if need_collection:
        print("\n" + "=" * 80)
        print(" 📝 待采集角色列表（按优先级排序）")
        print("=" * 80)
        # 优先级：完全未采集的优先，然后按数量排序
        priority_list = sorted(need_collection, key=lambda x: (x[3] == 0, x[3]))
        for idx, (chinese, pinyin, game, count) in enumerate(priority_list, 1):
            priority = "🔴紧急" if count == 0 else "🟡补充"
            print(f"  {idx:>2}. {chinese:<15} {pinyin:<20} {count:>5} {priority}")

        # 保存到文件
        output_file = project_root / "auto_spider_img" / "pending_roles.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for chinese, pinyin, game, count in priority_list:
                f.write(f"{chinese}\n")
        print(f"\n💾 已保存待采集列表到: {output_file}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
角色数据统计报告生成脚本
"""

import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ORGANIZED_IMAGES = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images")
REPORT_PATH = ORGANIZED_IMAGES / "角色数据整理报告.md"

# 角色名映射（中文名 -> 拼音）
PINYIN_TO_NAME = {
    'a1luo4na4': '阿洛娜',
    'pu3la1na4': '普拉娜',
    'na4xi1da2': '纳西妲',
    'ti2bao3': '缇宝',
    'ke3li4': '可莉',
    'di2ao4na4': '迪奥娜',
    'yao2yao2': '瑶瑶',
    'xi1ge2wen2': '希格雯',
    'lei3bei4': '蕾贝',
    'hei1ta3': '黑塔',
    'fu2xuan2': '符玄',
    'qi1qi1': '七七',
    'zao3you4': '早柚',
    'duo1li4': '多莉',
    'ka3qi2na4': '卡齐娜',
    'san1yue4qi1': '三月七',
    'hua1huo3': '花火',
    'yin2lang2': '银狼',
    'tian1tong2ai4li4si1': '天童爱丽丝',
    'zao3wu4': '早雾',
    'wei2li3nai4': '维里奈',
    'an1ke3': '安可',
    'you4hu2': '釉壶',
    'luo4ke4ke4': '洛可可',
    'lu4mu4yuan2': '鹿目圆',
    'xiao3mei3yan4': '晓美焰',
    'xue4xiao3ban3': '血小板',
    'lei2mu3': '雷姆',
    'la1mu3': '拉姆',
    'kang1na4': '康娜',
    'si4mi4nai3': '四糸乃',
    'kai3lu4': '凯露',
    'ke4luo2luo2': '克萝萝',
    'xiao3shan3': '小闪',
    'yi1li4ya3': '伊莉雅',
    'ren3ye3ren3': '忍野忍',
    'zhi4nai3': '智乃',
    'xiao3mai2': '小埋',
    'sha1wu4': '纱雾',
    'mao1gong1you4nai4': '猫宫又奈',
    'de2li4sha1': '德丽莎',
    'bu4luo4ni2ya4': '布洛妮娅',
    'ke3lin2': '可琳',
    'ai4li4er3': '爱丽儿',
    'shen1yue4': '神乐',
    'bai2shang4chui1xue3': '白上吹雪',
    'yue4qian1ye4': '月千夜',
    'fu2li4xi1ya4': '芙丽希娅',
    'li4ta3la1': '莉塔拉',
    'wei2pu3lei3': '维普蕾',
    'xia4ke4li3': '夏克里',
    'na4gan1': '纳甘',
    'ke1xie4ni2ya4': '科谢尼娅',
    'qi2ta3': '奇塔',
    'kou4er3fu2': '寇尔芙',
    'ke4luo2li4ke1': '克罗丽科',
    'pei4li3ti2ya4': '佩里缇亚',
    'a1ni4ya4': '阿尼亚',
    'luo4qian4': '洛茜',
    'ni2dou4zi5': '祢豆子',
    'xi1er3': '希儿',
    'xing4': '杏',
    'yi1se4lin2': '伊瑟琳',
    'fu2lan2': '芙兰',
    'fei1mi3li4si1': '菲米莉丝',
    'luo4ke3ke3': '罗可可',
    'mi2dou4zi': '蜜豆子',
    'ke4la1la1': '克拉拉',
}

# 排除的文件夹
EXCLUDE_FOLDERS = {'trash', 'trash_nsfw', 'trash_multi_face', '其他', '.DS_Store'}

def count_images(folder_path):
    """计算文件夹中的图片数量"""
    if not folder_path.exists():
        return 0
    return len([
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']
    ])

def get_role_stats():
    """获取所有角色的统计数据"""
    stats = []

    for folder in ORGANIZED_IMAGES.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name

        # 跳过排除的文件夹
        if folder_name in EXCLUDE_FOLDERS or folder_name.startswith('.'):
            continue

        image_count = count_images(folder)
        chinese_name = PINYIN_TO_NAME.get(folder_name, '未知')

        stats.append({
            'folder': folder_name,
            'name': chinese_name,
            'count': image_count
        })

    # 按图片数量降序排序
    stats.sort(key=lambda x: x['count'], reverse=True)
    return stats

def generate_report():
    """生成统计报告"""
    stats = get_role_stats()

    # 计算统计数据
    total_images = sum(s['count'] for s in stats)
    total_roles = len(stats)
    avg_per_role = total_images / total_roles if total_roles > 0 else 0
    max_count = stats[0]['count'] if stats else 0
    max_role = stats[0]['name'] if stats else ''
    min_count = stats[-1]['count'] if stats else 0
    min_role = stats[-1]['name'] if stats else ''

    # 生成分布统计
    distribution = defaultdict(int)
    for s in stats:
        if s['count'] >= 100:
            distribution['100张以上'] += 1
        elif s['count'] >= 50:
            distribution['50-99张'] += 1
        elif s['count'] >= 20:
            distribution['20-49张'] += 1
        elif s['count'] >= 10:
            distribution['10-19张'] += 1
        else:
            distribution['1-9张'] += 1

    # 生成报告内容
    report = f"""# 角色数据整理报告

**统计时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 数据概览

| 项目 | 数量 |
|------|------|
| 总图片数 | {total_images:,} |
| 角色文件夹数 | {total_roles} |
| 平均每角色图片数 | {avg_per_role:.2f} |
| 最多图片角色 | {max_role} ({max_count}张) |
| 最少图片角色 | {min_role} ({min_count}张) |

---

## 各角色图片数量分布

| 排名 | 角色 | 拼音 | 图片数 | 占比 |
|------|------|------|--------|------|
"""

    for i, s in enumerate(stats, 1):
        percentage = s['count'] / total_images * 100 if total_images > 0 else 0
        report += f"| {i} | {s['name']} | {s['folder']} | {s['count']} | {percentage:.2f}% |\n"

    report += f"""

---

## 数量分布统计

| 区间 | 角色数 | 说明 |
|------|--------|------|
"""

    for range_name in ['100张以上', '50-99张', '20-49张', '10-19张', '1-9张']:
        count = distribution.get(range_name, 0)
        report += f"| {range_name} | {count} | - |\n"

    report += f"""

---

## 本次整理操作

### 1. 从"其他"目录合并的角色

| 角色 | 图片数 | 操作 |
|------|--------|------|
| 鹿目圆 (lu4mu4yuan2) | +182张 | 从"其他"合并到主目录 |
| 神乐 (shen1yue4) | +348张 | 从"其他"合并到主目录 |

### 2. 神乐文件夹合并

| 原文件夹 | 图片数 | 操作 |
|----------|--------|------|
| shen2le4 | 232张 | 合并到 shen1yue4 |
| shen1yue4 | 合并后 | 580张 |

---

## 其他目录保留角色

以下角色不在loli-role.txt名单中，保留在"其他"目录：

| 拼音 | 图片数 |
|------|--------|
"""

    # 列出其他目录中的角色
    other_dir = ORGANIZED_IMAGES / "其他"
    if other_dir.exists():
        other_stats = []
        for folder in other_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith('.'):
                image_count = count_images(folder)
                other_stats.append((folder.name, image_count))

        other_stats.sort(key=lambda x: x[1], reverse=True)
        for folder_name, image_count in other_stats:
            report += f"| {folder_name} | {image_count} |\n"

    report += f"""

---

## Trash 目录统计

| 文件夹 | 类型 | 说明 |
|--------|------|------|
| trash | 通用垃圾 | 低质量图片等 |
| trash_nsfw | NSFW内容 | 敏感内容图片 |
| trash_multi_face | 非单人 | 包含多人的图片 |

---

*报告生成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # 写入报告
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 报告已生成: {REPORT_PATH}")
    return report

if __name__ == "__main__":
    report = generate_report()
    print("\n" + "=" * 60)
    print(report)

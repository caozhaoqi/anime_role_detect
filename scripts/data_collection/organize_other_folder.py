#!/usr/bin/env python3
"""
角色数据整理脚本
将"其他"目录中符合loli-role.txt名单的角色移动到主目录
"""

import os
import shutil
from pathlib import Path

# 路径配置
ORGANIZED_IMAGES = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images")
OTHER_DIR = ORGANIZED_IMAGES / "其他"
ROLE_LIST_PATH = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt"

# PINYIN_MAPPING - 从constants.py复制
PINYIN_MAPPING = {
    '阿洛娜': 'a1luo4na4',
    '普拉娜': 'pu3la1na4',
    '纳西妲': 'na4xi1da2',
    '缇宝': 'ti2bao3',
    '可莉': 'ke3li4',
    '迪奥娜': 'di2ao4na4',
    '瑶瑶': 'yao2yao2',
    '希格雯': 'xi1ge2wen2',
    '蕾贝': 'lei3bei4',
    '黑塔': 'hei1ta3',
    '符玄': 'fu2xuan2',
    '七七': 'qi1qi1',
    '早柚': 'zao3you4',
    '多莉': 'duo1li4',
    '卡齐娜': 'ka3qi2na4',
    '三月七': 'san1yue4qi1',
    '花火': 'hua1huo3',
    '银狼': 'yin2lang2',
    '天童爱丽丝': 'tian1tong2ai4li4si1',
    '早雾': 'zao3wu4',
    '维里奈': 'wei2li3nai4',
    '安可': 'an1ke3',
    '釉壶': 'you4hu2',
    '洛可可': 'luo4ke4ke4',
    '鹿目圆': 'lu4mu4yuan2',
    '晓美焰': 'xiao3mei3yan4',
    '血小板': 'xue4xiao3ban3',
    '雷姆': 'lei2mu3',
    '拉姆': 'la1mu3',
    '康娜': 'kang1na4',
    '四糸乃': 'si4mi4nai3',
    '凯露': 'kai3lu4',
    '克萝萝': 'ke4luo2luo2',
    '小闪': 'xiao3shan3',
    '伊莉雅': 'yi1li4ya3',
    '忍野忍': 'ren3ye3ren3',
    '智乃': 'zhi4nai3',
    '小埋': 'xiao3mai2',
    '纱雾': 'sha1wu4',
    '猫宫又奈': 'mao1gong1you4nai4',
    '德丽莎': 'de2li4sha1',
    '布洛妮娅': 'bu4luo4ni2ya4',
    '可琳': 'ke3lin2',
    '爱丽儿': 'ai4li4er3',
    '神乐': 'shen1yue4',
    '白上吹雪': 'bai2shang4chui1xue3',
    '月千夜': 'yue4qian1ye4',
    '芙丽希娅': 'fu2li4xi1ya4',
    '莉塔拉': 'li4ta3la1',
    '维普蕾': 'wei2pu3lei3',
    '夏克里': 'xia4ke4li3',
    '纳甘': 'na4gan1',
    '科谢尼娅': 'ke1xie4ni2ya4',
    '奇塔': 'qi2ta3',
    '寇尔芙': 'kou4er3fu2',
    '克罗丽科': 'ke4luo2li4ke1',
    '佩里缇亚': 'pei4li3ti2ya4',
    '阿尼亚': 'a1ni4ya4',
    '洛茜': 'luo4qian4',
    '祢豆子': 'ni2dou4zi5',
    '希儿': 'xi1er3',
    '杏': 'xing4',
    '伊瑟琳': 'yi1se4lin2',
    '芙兰': 'fu2lan2',
    '菲米莉丝': 'fei1mi3li4si1',
    '罗可可': 'luo4ke3ke3',
    '蜜豆子': 'mi2dou4zi',
}

def read_role_list():
    """读取角色名单"""
    valid_pinyins = set()
    with open(ROLE_LIST_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 1:
                    name = parts[0]
                    if name in PINYIN_MAPPING:
                        valid_pinyins.add(PINYIN_MAPPING[name])
    return valid_pinyins

def analyze_other_dir():
    """分析其他目录中的文件夹"""
    if not OTHER_DIR.exists():
        print(f"❌ 目录不存在: {OTHER_DIR}")
        return {}, []

    folders = []
    for item in OTHER_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            folders.append(item.name)

    return folders

def match_folder_to_role(folder_name, valid_pinyins):
    """判断文件夹是否匹配名单中的角色"""
    # 直接匹配
    if folder_name in valid_pinyins:
        return True, folder_name

    # 部分匹配 (处理空格或特殊命名)
    folder_clean = folder_name.replace(' ', '').replace('_', '')

    for pinyin in valid_pinyins:
        pinyin_clean = pinyin.replace(' ', '').replace('_', '')
        if folder_clean == pinyin_clean:
            return True, pinyin

    return False, None

def get_folder_image_count(folder_path):
    """获取文件夹中的图片数量"""
    if not folder_path.exists():
        return 0
    count = 0
    for item in folder_path.iterdir():
        if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']:
            count += 1
    return count

def main():
    print("=" * 60)
    print("角色数据整理脚本")
    print("=" * 60)

    # 读取有效角色拼音列表
    valid_pinyins = read_role_list()
    print(f"\n✅ 名单中共有 {len(valid_pinyins)} 个有效角色")

    # 分析其他目录
    folders = analyze_other_dir()
    print(f"\n📁 '其他'目录中共有 {len(folders)} 个文件夹")

    # 分析每个文件夹
    to_move = []  # 需要移动的文件夹
    to_keep = []  # 保留在其他的文件夹

    print("\n" + "-" * 60)
    print("分析结果：")
    print("-" * 60)

    for folder in sorted(folders):
        folder_path = OTHER_DIR / folder
        is_match, matched_pinyin = match_folder_to_role(folder, valid_pinyins)
        image_count = get_folder_image_count(folder_path)

        if is_match:
            # 检查目标目录是否已存在
            target_path = ORGANIZED_IMAGES / matched_pinyin
            if target_path.exists():
                target_count = get_folder_image_count(target_path)
                total_count = image_count + target_count
                to_move.append((folder, matched_pinyin, image_count, target_count, total_count))
                print(f"  📦 {folder} → {matched_pinyin} ({image_count}张 + {target_count}张 = {total_count}张)")
            else:
                to_move.append((folder, matched_pinyin, image_count, 0, image_count))
                print(f"  ✅ {folder} → {matched_pinyin} ({image_count}张) [新目录]")
        else:
            to_keep.append((folder, image_count))
            print(f"  ❌ {folder} ({image_count}张) - 不在名单中")

    # 汇总
    print("\n" + "=" * 60)
    print("汇总：")
    print("=" * 60)
    print(f"  需要移动到主目录的文件夹: {len(to_move)} 个")
    print(f"  保留在其他目录的文件夹: {len(to_keep)} 个")

    if to_move:
        total_images_to_move = sum(item[2] for item in to_move)
        print(f"  涉及移动的图片数量: {total_images_to_move} 张")

    # 询问确认
    print("\n" + "-" * 60)
    response = input("是否执行移动操作? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ 操作已取消")
        return

    # 执行移动
    print("\n" + "=" * 60)
    print("执行移动操作...")
    print("=" * 60)

    moved_count = 0
    for folder, matched_pinyin, image_count, target_count, total_count in to_move:
        folder_path = OTHER_DIR / folder
        target_path = ORGANIZED_IMAGES / matched_pinyin

        # 如果目标目录不存在，先创建
        if not target_path.exists():
            target_path.mkdir(parents=True, exist_ok=True)
            print(f"\n📂 创建新目录: {matched_pinyin}")

        # 移动文件
        files_moved = 0
        for item in folder_path.iterdir():
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']:
                target_file = target_path / item.name
                # 如果目标文件已存在，跳过
                if target_file.exists():
                    print(f"    ⚠️ 文件已存在，跳过: {item.name}")
                    continue
                shutil.move(str(item), str(target_file))
                files_moved += 1

        # 删除空文件夹
        try:
            if folder_path.exists() and not any(folder_path.iterdir()):
                folder_path.rmdir()
                print(f"    🗑️ 删除空文件夹: {folder}")
        except Exception as e:
            print(f"    ⚠️ 无法删除文件夹: {e}")

        moved_count += 1
        print(f"    ✅ 已移动 {files_moved} 个文件到 {matched_pinyin}")

    print("\n" + "=" * 60)
    print(f"✅ 整理完成! 共处理 {moved_count} 个文件夹")
    print("=" * 60)

if __name__ == "__main__":
    main()

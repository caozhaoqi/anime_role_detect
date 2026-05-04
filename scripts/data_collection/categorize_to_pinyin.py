#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将中英文角色目录归类到拼音命名目录"""

import os
import sys
from pathlib import Path

ORGANIZED_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images')

# 中英文到拼音的映射表
NAME_MAPPING = {
    # 英文名 -> 拼音
    'Aris': 'tian1tong2ai4li4si1',      # 天童爱丽丝
    'Arona': 'a1luo4na4',               # 阿洛娜
    'Plana': 'pu3la1na4',               # 普拉娜
    'Nahida': 'na4xi1da2',              # 纳西妲
    'Princess': 'ti2bao3',              # 缇宝
    'Klee': 'ke3li4',                   # 可莉
    'Diona': 'di2ao4na4',               # 迪奥娜
    'Yaoyao': 'yao2yao2',               # 瑶瑶
    'Sigewinne': 'xi1ge2wen2',          # 希格雯
    'Rebe': 'lei3bei4',                 # 蕾贝
    'Herta': 'hei1ta3',                 # 黑塔
    'Fu Xuan': 'fu2xuan2',              # 符玄
    'Qiqi': 'qi1qi1',                   # 七七
    'Sayu': 'zao3you4',                 # 早柚
    'Dori': 'duo1li4',                  # 多莉
    'Kachina': 'ka3qi2na4',             # 卡齐娜
    'March 7th': 'san1yue4qi1',         # 三月七
    'Sparkle': 'hua1huo3',              # 花火
    'Silver Wolf': 'yin2lang2',         # 银狼
    'Rem': 'lei2mu3',                   # 雷姆
    'Ram': 'la1mu3',                    # 拉姆
    'Verina': 'wei2li3nai4',            # 维里奈
    'Encore': 'an1ke3',                 # 安可
    'Youhu': 'you4hu2',                 # 釉壶
    'Roccia': 'luo4ke3ke3',             # 洛可可
    'Madoka Kaname': 'lu4mu4yuan2',     # 鹿目圆
    'Homura Akemi': 'xiao3mei3yan4',    # 晓美焰
    'Platelet': 'xue4xiao3ban3',        # 血小板
    'Kanna': 'kang1na4',                # 康娜
    'Yoshino': 'si4mi4nai3',            # 四糸乃
    'Kyaru': 'kai3lu4',                 # 凯露
    'Klor': 'ke4luo2luo2',              # 克萝萝
    'Flash': 'xiao3shan3',              # 小闪
    'Illya': 'yi1li4ya3',               # 伊莉雅
    'Oshino Shinobu': 'ren3ye3ren3',    # 忍野忍
    'Chino': 'zhi4nai3',                # 智乃
    'Tsumugi': 'xiao3mai2',             # 小埋
    'Sagiri': 'sha1wu4',                # 纱雾
    'Yanagi': 'mao1gong1you4nai4',      # 猫宫又奈
    'Theresa': 'de2li4sha1',            # 德丽莎
    'Bronya': 'bu4luo4ni2ya4',          # 布洛妮娅
    'Kira': 'ke3lin2',                  # 可琳
    'Ariel': 'ai4li4er3',               # 爱丽儿
    'Kagura': 'shen2le4',               # 神乐
    'Shirogane Noel': 'bai2shang4chui1xue3',  # 白上吹雪
    'Tsukiyo': 'yue4qian1ye4',          # 月千夜
    'Furisia': 'fu2li4xi1ya4',          # 芙丽希娅
    'Lita': 'li4ta3la1',                # 莉塔拉
    'Viprey': 'wei2pu3lei3',            # 维普蕾
    'Shakri': 'xia4ke4li3',             # 夏克里
    'Nagan': 'na4gan1',                 # 纳甘
    'Koshenia': 'ke1xie4ni2ya4',        # 科谢尼娅
    'Kita': 'qi2ta3',                   # 奇塔
    'Korvu': 'kou4er3fu2',              # 寇尔芙
    'Krokri': 'ke4luo2li4ke1',          # 克罗丽科
    'Peritia': 'pei4li3ti2ya4',         # 佩里缇亚
    'Anya Forger': 'a1ni4ya4',          # 阿尼亚
    'Rosci': 'luo4qian4',               # 洛茜
    'Nezuko Kamado': 'mi3dou4zi5',      # 祢豆子
    'Seele Vollerei': 'xi1er3',         # 希儿
    'An Makhall': 'xing4',              # 杏
    'Iselin LeviSius': 'yi1se4lin2',    # 伊瑟琳
    'Fran': 'fu2lan2',                  # 芙兰
    'Fimilis': 'fei1mi3li4si1',         # 菲米莉丝
    # 混合格式处理
    'Yaoyao yuan2shen2': 'yao2yao2',
    'ke3lin2_wei1ke4si1': 'ke3lin2',
    'li4li4ya3_a1lin2': 'yi1li4ya3',
    'li4li4ya4·a1lin2': 'yi1li4ya3',
    'luo2sha1li4ya3_a1lin2': 'yi1li4ya3',
    'cong2yu3': 'gong1cong2',           # 丛雨 -> 丛雨
}

def find_non_pinyin_dirs(base_dir):
    """找出非拼音格式的目录"""
    non_pinyin = []
    for dir_path in base_dir.iterdir():
        if dir_path.is_dir():
            dir_name = dir_path.name
            # 检查是否是拼音格式（拼音格式包含数字1-4）
            if not any(c.isdigit() for c in dir_name):
                non_pinyin.append(dir_name)
            elif dir_name in NAME_MAPPING:
                non_pinyin.append(dir_name)
    return non_pinyin

def categorize_to_pinyin(base_dir):
    """将中英文目录归类到拼音目录"""
    categories = []
    
    for dir_path in base_dir.iterdir():
        if dir_path.is_dir():
            dir_name = dir_path.name
            
            # 检查是否需要归类
            if dir_name in NAME_MAPPING:
                target_pinyin = NAME_MAPPING[dir_name]
                target_path = base_dir / target_pinyin
                
                if target_path.exists() and target_path.is_dir():
                    # 合并到目标目录
                    print(f"📦 合并: {dir_name} -> {target_pinyin}")
                    moved_count = 0
                    for img_file in dir_path.glob('*'):
                        if img_file.is_file():
                            new_file = target_path / img_file.name
                            if not new_file.exists():
                                img_file.rename(new_file)
                                moved_count += 1
                    # 删除原目录
                    if moved_count > 0:
                        dir_path.rmdir()
                    categories.append((dir_name, target_pinyin, moved_count, '合并'))
                else:
                    # 重命名目录
                    print(f"📝 重命名: {dir_name} -> {target_pinyin}")
                    dir_path.rename(target_path)
                    categories.append((dir_name, target_pinyin, 0, '重命名'))
    
    return categories

def main():
    print("=" * 60)
    print("📦 将中英文角色目录归类到拼音命名目录")
    print("=" * 60)
    
    # 找出需要处理的目录
    print("\n🔍 查找需要归类的目录...")
    non_pinyin_dirs = find_non_pinyin_dirs(ORGANIZED_DIR)
    print(f"   找到 {len(non_pinyin_dirs)} 个非拼音格式目录")
    
    if non_pinyin_dirs:
        print("   需要处理的目录:", non_pinyin_dirs)
    
    # 执行归类
    print("\n📦 开始归类...")
    results = categorize_to_pinyin(ORGANIZED_DIR)
    
    # 统计结果
    print("\n" + "=" * 60)
    print("📊 归类结果")
    print("=" * 60)
    
    if results:
        print("\n处理详情:")
        print("-" * 50)
        for src, dest, count, action in results:
            if action == '合并':
                print(f"✓ {src} → {dest} (移动 {count} 个文件)")
            else:
                print(f"✓ {src} → {dest} (重命名)")
        print(f"\n总计处理: {len(results)} 个目录")
    else:
        print("没有需要归类的目录")
    
    # 最终统计
    print("\n" + "=" * 60)
    total_dirs = 0
    total_images = 0
    
    for dir_path in ORGANIZED_DIR.iterdir():
        if dir_path.is_dir():
            total_dirs += 1
            for img_file in dir_path.glob('*'):
                if img_file.is_file():
                    total_images += 1
    
    print(f"\n📁 角色目录数: {total_dirs}")
    print(f"🖼️  图片总数: {total_images}")
    print("\n✅ 归类完成!")

if __name__ == '__main__':
    main()

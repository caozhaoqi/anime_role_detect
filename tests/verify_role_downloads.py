#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对loli-role.txt角色名与已下载角色目录"""

from pathlib import Path

ROLE_FILE = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt')
ORGANIZED_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images')

# 拼音映射表
PINYIN_MAP = {
    '阿洛娜': 'a1luo4na4', '普拉娜': 'pu3la1na4', '纳西妲': 'na4xi1da2', '可莉': 'ke3li4',
    '迪奥娜': 'di2ao4na4', '瑶瑶': 'yao2yao2', '黑塔': 'hei1ta3', '符玄': 'fu2xuan2',
    '七七': 'qi1qi1', '早柚': 'zao3you4', '多莉': 'duo1li4', '卡齐娜': 'ka3qi2na4',
    '三月七': 'san1yue4qi1', '花火': 'hua1huo3', '银狼': 'yin2lang2', '雷姆': 'lei2mu3',
    '拉姆': 'la1mu3', '缇宝': 'ti2bao3', '维里奈': 'wei2li3nai4', '安可': 'an1ke3',
    '釉壶': 'you4hu2', '洛可可': 'luo4ke3ke3', '鹿目圆': 'lu4mu4yuan2', '晓美焰': 'xiao3mei3yan4',
    '血小板': 'xue4xiao3ban3', '康娜': 'kang1na4', '四糸乃': 'si4mi4nai3', '凯露': 'kai3lu4',
    '克萝萝': 'ke4luo2luo2', '小闪': 'xiao3shan3', '伊莉雅': 'yi1li4ya3', '忍野忍': 'ren3ye3ren3',
    '智乃': 'zhi4nai3', '小埋': 'xiao3mai2', '纱雾': 'sha1wu4', '猫宫又奈': 'mao1gong1you4nai4',
    '德丽莎': 'de2li4sha1', '布洛妮娅': 'bu4luo4ni2ya4', '可琳': 'ke3lin2', '爱丽儿': 'ai4li4er3',
    '神乐': 'shen2le4', '白上吹雪': 'bai2shang4chui1xue3', '月千夜': 'yue4qian1ye4',
    '芙丽希娅': 'fu2li4xi1ya4', '莉塔拉': 'li4ta3la1', '维普蕾': 'wei2pu3lei3', '夏克里': 'xia4ke4li3',
    '纳甘': 'na4gan1', '科谢尼娅': 'ke1xie4ni2ya4', '奇塔': 'qi2ta3', '寇尔芙': 'kou4er3fu2',
    '克罗丽科': 'ke4luo2li4ke1', '佩里缇亚': 'pei4li3ti2ya4', '阿尼亚': 'a1ni4ya4', '洛茜': 'luo4qian4',
    '祢豆子': 'mi3dou4zi5', '希儿': 'xi1er3', '杏': 'xing4', '伊瑟琳': 'yi1se4lin2',
    '芙兰': 'fu2lan2', '菲米莉丝': 'fei1mi3li4si1', '天童爱丽丝': 'tian1tong2ai4li4si1',
    '希格雯': 'xi1ge2wen2', '蕾贝': 'lei3bei4', '早雾': 'zao3wu4',
}

def get_listed_roles():
    """获取loli-role.txt中的角色列表"""
    roles = []
    with open(ROLE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ')
                chinese_name = parts[0]
                roles.append(chinese_name)
    return roles

def get_downloaded_roles():
    """获取已下载的角色目录列表"""
    roles = {}
    for dir_path in ORGANIZED_DIR.iterdir():
        if dir_path.is_dir():
            role_name = dir_path.name
            img_count = len(list(dir_path.glob('*')))
            roles[role_name] = img_count
    return roles

def main():
    print("=" * 70)
    print("✅ 核对loli-role.txt角色名与已下载角色目录")
    print("=" * 70)
    
    listed_roles = get_listed_roles()
    downloaded_roles = get_downloaded_roles()
    
    print(f"\n📋 loli-role.txt 角色数: {len(listed_roles)}")
    print(f"📁 已下载角色目录数: {len(downloaded_roles)}")
    
    # 找出列表中有但未下载的角色
    missing_in_download = []
    for role in listed_roles:
        pinyin = PINYIN_MAP.get(role, None)
        if pinyin and pinyin not in downloaded_roles:
            missing_in_download.append((role, pinyin))
        elif not pinyin:
            missing_in_download.append((role, '未知拼音'))
    
    # 找出下载了但列表中没有的角色
    extra_in_download = []
    for dir_name, count in downloaded_roles.items():
        # 检查是否是已知拼音
        found = False
        for cn, py in PINYIN_MAP.items():
            if py == dir_name:
                if cn in listed_roles:
                    found = True
                break
        if not found:
            extra_in_download.append((dir_name, count))
    
    # 统计列表中角色的下载情况
    download_stats = []
    for role in listed_roles:
        pinyin = PINYIN_MAP.get(role)
        if pinyin in downloaded_roles:
            download_stats.append((role, pinyin, downloaded_roles[pinyin]))
        else:
            download_stats.append((role, pinyin, 0))
    
    # 输出结果
    print("\n" + "=" * 70)
    print("📊 列表角色下载状态")
    print("=" * 70)
    print(f"{'序号':<4} {'中文名':<10} {'拼音':<20} {'图片数':<8} {'状态'}")
    print("-" * 70)
    
    sufficient_count = 0
    insufficient_count = 0
    missing_count = 0
    
    for i, (role, pinyin, count) in enumerate(download_stats, 1):
        if count == 0:
            status = "❌"
            missing_count += 1
        elif count < 100:
            status = "⚠️"
            insufficient_count += 1
        else:
            status = "✅"
            sufficient_count += 1
        print(f"{i:<4} {role:<10} {pinyin:<20} {count:<8} {status}")
    
    print("\n" + "=" * 70)
    print("📈 统计汇总")
    print("=" * 70)
    print(f"✅ 图片充足(>=100): {sufficient_count} 个")
    print(f"⚠️ 图片不足(<100): {insufficient_count} 个")
    print(f"❌ 尚未下载: {missing_count} 个")
    
    if missing_in_download:
        print("\n" + "=" * 70)
        print("❌ 列表中有但未下载的角色:")
        print("=" * 70)
        for role, pinyin in missing_in_download:
            print(f"  • {role} ({pinyin})")
    
    if extra_in_download:
        print("\n" + "=" * 70)
        print("⚠️  下载了但列表中没有的角色:")
        print("=" * 70)
        for dir_name, count in sorted(extra_in_download, key=lambda x: x[1], reverse=True):
            print(f"  • {dir_name}: {count} 张图片")

if __name__ == '__main__':
    main()

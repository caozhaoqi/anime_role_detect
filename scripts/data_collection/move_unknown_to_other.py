#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将不在loli-role.txt名单中的角色移动到其他目录"""
import shutil
from pathlib import Path

Loli_ROLE_FILE = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt')
IMG_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images')
OTHER_DIR = IMG_DIR / '其他'

CHINESE_TO_FOLDER = {
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
    '洛可可': 'luo4ke3ke3',
    '鹿目圆': 'luo4qian4',
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
    '神乐': 'shen2le4',
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
    '洛茜': 'luo4xi1',
    '祢豆子': 'mi2dou4zi',
    '希儿': 'xi1er3',
    '杏': 'xing4',
    '伊瑟琳': 'yi1se4lin2',
    '芙兰': 'fu2lan2',
    '菲米莉丝': 'fei1mi3li4si1',
}

ALLOWED_FOLDERS = set(CHINESE_TO_FOLDER.values())

def main():
    print("=" * 60)
    print("📁 将不在名单中的角色移到其他目录")
    print("=" * 60)

    print(f"\n📋 允许的文件夹数量: {len(ALLOWED_FOLDERS)}")

    folders_to_move = []
    folders_to_keep = []

    for item in IMG_DIR.iterdir():
        if not item.is_dir():
            continue
        if item.name in ['其他', 'manifest.txt']:
            continue

        if item.name in ALLOWED_FOLDERS:
            folders_to_keep.append(item.name)
        else:
            folders_to_move.append(item.name)

    print(f"\n✅ 需要保留的文件夹 ({len(folders_to_keep)}):")
    for f in sorted(folders_to_keep):
        print(f"   {f}")

    print(f"\n❌ 需要移动到其他的文件夹 ({len(folders_to_move)}):")
    for f in sorted(folders_to_move):
        print(f"   {f}")

    if not folders_to_move:
        print("\n✅ 所有文件夹都在名单中，无需移动")
        return

    OTHER_DIR.mkdir(exist_ok=True)
    print(f"\n📂 创建/使用目录: {OTHER_DIR}")

    for folder in folders_to_move:
        src = IMG_DIR / folder
        dst = OTHER_DIR / folder
        if dst.exists():
            print(f"   ⚠️ 目标已存在: {dst}")
            continue
        shutil.move(str(src), str(dst))
        print(f"   ✅ 移动: {folder} -> 其他/")

    print("\n" + "=" * 60)
    print("✅ 完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()
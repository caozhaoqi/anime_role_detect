#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计名单角色的完整图片数目"""
import os
from pathlib import Path
from pypinyin import lazy_pinyin, Style

ROLE_FILE = Path('auto_spider_img/loli-role.txt')
IMG_DIR = Path('data/organized_images')

# 读取名单角色
roles = []
with open(ROLE_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(' ')
            roles.append(parts[0])

# 手动映射
MAPPING = {
    '阿洛娜': 'a1luo4na4', '普拉娜': 'pu3la1na4', '纳西妲': 'na4xi1da2',
    '缇宝': 'ti2bao3', '可莉': 'ke3li4', '迪奥娜': 'di2ao4na4',
    '瑶瑶': 'yao2yao2', '希格雯': 'xi1ge2wen2', '蕾贝': 'lei3bei4',
    '黑塔': 'hei1ta3', '符玄': 'fu2xuan2', '七七': 'qi1qi1',
    '早柚': 'zao3you4', '多莉': 'duo1li4', '卡齐娜': 'ka3qi2na4',
    '三月七': 'san1yue4qi1', '花火': 'hua1huo3', '银狼': 'yin2lang2',
    '天童爱丽丝': 'tian1tong2ai4li4si1', '早雾': 'zao3wu4', '维里奈': 'wei2li3nai4',
    '安可': 'an1ke3', '釉壶': 'you4hu2', '洛可可': 'luo4ke3ke3',
    '鹿目圆': 'luo4qian4', '晓美焰': 'xiao3mei3yan4', '血小板': 'xue4xiao3ban3',
    '雷姆': 'lei2mu3', '拉姆': 'la1mu3', '康娜': 'kang1na4',
    '四糸乃': 'si4mi4nai3', '凯露': 'kai3lu4', '克萝萝': 'ke4luo2luo2',
    '小闪': 'xiao3shan3', '伊莉雅': 'yi1li4ya3', '忍野忍': 'ren3ye3ren3',
    '智乃': 'zhi4nai3', '小埋': 'xiao3mai2', '纱雾': 'sha1wu4',
    '猫宫又奈': 'mao1gong1you4nai4', '德丽莎': 'de2li4sha1', '布洛妮娅': 'bu4luo4ni2ya4',
    '可琳': 'ke3lin2', '爱丽儿': 'ai4li4er3', '神乐': 'shen2le4',
    '白上吹雪': 'bai2shang4chui1xue3', '月千夜': 'yue4qian1ye4', '芙丽希娅': 'fu2li4xi1ya4',
    '莉塔拉': 'li4ta3la1', '维普蕾': 'wei2pu3lei3', '夏克里': 'xia4ke4li3',
    '纳甘': 'na4gan1', '科谢尼娅': 'ke1xie4ni2ya4', '奇塔': 'qi2ta3',
    '寇尔芙': 'kou4er3fu2', '克罗丽科': 'ke4luo2li4ke1', '佩里缇亚': 'pei4li3ti2ya4',
    '阿尼亚': 'a1ni4ya4', '洛茜': 'luo4xi1', '祢豆子': 'mi2dou4zi',
    '希儿': 'xi1er3', '杏': 'xing4', '伊瑟琳': 'yi1se4lin2',
    '芙兰': 'fu2lan2', '菲米莉丝': 'fei1mi3li4si1'
}

stats = []
for role in roles:
    folder = MAPPING.get(role)
    if not folder:
        folder = ''.join(lazy_pinyin(role, style=Style.TONE3))
    dir_path = IMG_DIR / folder
    count = len(list(dir_path.glob('*'))) if dir_path.exists() else 0
    stats.append((role, folder, count))

stats.sort(key=lambda x: x[2], reverse=True)

print(f'共 {len(stats)} 个角色')
print()
print(f'{"角色":<10} {"文件夹":<25} {"图片数":>8}')
print('-' * 50)
for role, folder, cnt in stats:
    status = '✅' if cnt >= 100 else '⚠️' if cnt > 0 else '❌'
    print(f'{status} {role:<8} {folder:<25} {cnt:>6} 张')

good = len([s for s in stats if s[2] >= 100])
low = len([s for s in stats if 0 < s[2] < 100])
missing = len([s for s in stats if s[2] == 0])
print('-' * 50)
print(f'✅ 充足: {good} | ⚠️ 不足: {low} | ❌ 缺失: {missing}')

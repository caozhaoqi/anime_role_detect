#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计额外角色图片分布"""
from pathlib import Path

IMG_DIR = Path('data/organized_images')

# 65个角色的文件夹
ALLOWED_FOLDERS = {
    'a1luo4na4', 'pu3la1na4', 'na4xi1da2', 'ti2bao3', 'ke3li4', 'di2ao4na4',
    'yao2yao2', 'xi1ge2wen2', 'lei3bei4', 'hei1ta3', 'fu2xuan2', 'qi1qi1',
    'zao3you4', 'duo1li4', 'ka3qi2na4', 'san1yue4qi1', 'hua1huo3', 'yin2lang2',
    'tian1tong2ai4li4si1', 'zao3wu4', 'wei2li3nai4', 'an1ke3', 'you4hu2', 'luo4ke3ke3',
    'luo4qian4', 'xiao3mei3yan4', 'xue4xiao3ban3', 'lei2mu3', 'la1mu3', 'kang1na4',
    'si4mi4nai3', 'kai3lu4', 'ke4luo2luo2', 'xiao3shan3', 'yi1li4ya3', 'ren3ye3ren3',
    'zhi4nai3', 'xiao3mai2', 'sha1wu4', 'mao1gong1you4nai4', 'de2li4sha1', 'bu4luo4ni2ya4',
    'ke3lin2', 'ai4li4er3', 'shen2le4', 'bai2shang4chui1xue3', 'yue4qian1ye4', 'fu2li4xi1ya4',
    'li4ta3la1', 'wei2pu3lei3', 'xia4ke4li3', 'na4gan1', 'ke1xie4ni2ya4', 'qi2ta3',
    'kou4er3fu2', 'ke4luo2li4ke1', 'pei4li3ti2ya4', 'a1ni4ya4', 'luo4xi1', 'mi2dou4zi',
    'xi1er3', 'xing4', 'yi1se4lin2', 'fu2lan2', 'fei1mi3li4si1'
}

# 额外文件夹
extra_folders = []
for d in IMG_DIR.iterdir():
    if d.is_dir() and d.name not in ALLOWED_FOLDERS:
        count = len([f for f in d.glob('*') if f.is_file()])
        extra_folders.append((d.name, count))

# 排序
extra_folders.sort(key=lambda x: x[1], reverse=True)

print('=' * 60)
print('📊 额外角色图片分布 (共 {} 个角色)'.format(len(extra_folders)))
print('=' * 60)
print('{:<4} {:<30} {:>8}'.format('序号', '文件夹', '图片数'))
print('-' * 60)

total = 0
for i, (folder, count) in enumerate(extra_folders, 1):
    print('{:<4} {:<30} {:>6} 张'.format(i, folder, count))
    total += count

print('-' * 60)
print('总计: {} 个角色, {} 张图片'.format(len(extra_folders), total))

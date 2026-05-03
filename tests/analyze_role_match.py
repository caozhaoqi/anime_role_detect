#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析loli-role.txt与URL文件的完整匹配"""

from pathlib import Path

URL_DIR = Path('spider_image_system/data/img_url')
ROLE_FILE = Path('auto_spider_img/loli-role.txt')

existing = {}
for f in URL_DIR.glob('*_img.txt'):
    role = f.stem.replace('_img', '')
    with open(f, 'r', encoding='utf-8') as fp:
        existing[role] = len([l for l in fp if l.strip()])

name_map = {
    '阿洛娜': 'a1luo4na4',
    '普拉娜': 'pu3la1na4',
    '纳西妲': 'na4xi1da2',
    '可莉': 'ke3li4',
    '迪奥娜': 'di2ao4na4',
    '瑶瑶': 'yao2yao2',
    '黑塔': 'hei1ta3',
    '符玄': 'fu2xuan2',
    '七七': 'qi2qi2',
    '早柚': 'zao3you4',
    '多莉': 'duo1li4',
    '卡齐娜': 'ka3qi2na4',
    '三月七': 'san1yue4qi1',
    '花火': 'hua1huo3',
    '银狼': 'yin2lang2',
    '雷姆': 'lei2mu3',
    '拉姆': 'la1mu3',
    '缇宝': 'ti2bao3',
    '维里奈': 'wei2li3nai4',
}

print('=' * 60)
print('📋 名单角色URL匹配分析')
print('=' * 60)

matched = []
unmatched = []
for cn in name_map:
    py = name_map[cn]
    cnt = existing.get(py, 0)
    if cnt > 0:
        matched.append((cn, py, cnt))
    else:
        unmatched.append((cn, py, cnt))

print(f'\n✅ 已匹配到URL: {len(matched)} 个')
print('-' * 50)
for cn, py, cnt in sorted(matched, key=lambda x: x[2], reverse=True):
    status = '充足' if cnt >= 100 else '不足'
    print(f'  {cn}: {cnt} URL [{status}]')

print(f'\n❌ 未匹配到URL: {len(unmatched)} 个')
print('-' * 50)
for cn, py, cnt in unmatched:
    print(f'  {cn}: {py}')

all_file_pinyin = set(existing.keys())
matched_pinyin = set(name_map.values())
remaining_pinyin = all_file_pinyin - matched_pinyin

print(f'\n📁 URL文件中未在名单映射的角色({len(remaining_pinyin)}个):')
print('-' * 50)
for py in sorted(remaining_pinyin)[:30]:
    cnt = existing[py]
    print(f'  {py}: {cnt} URL')
if len(remaining_pinyin) > 30:
    print(f'  ... 还有 {len(remaining_pinyin) - 30} 个')

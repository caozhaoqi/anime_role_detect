#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析loli-role.txt角色名单中缺少的角色 - 支持中文名匹配"""

from pathlib import Path

URL_DIR = Path('spider_image_system/data/img_url')
ROLE_FILE = Path('auto_spider_img/loli-role.txt')

name_mapping = {}

for f in URL_DIR.glob('*_img.txt'):
    with open(f, 'r', encoding='utf-8') as fp:
        urls = [l.strip() for l in fp if l.strip()]
    pinyin = f.stem.replace('_img', '')

    chinese_names = []
    for line in open(ROLE_FILE, 'r', encoding='utf-8'):
        parts = line.strip().split()
        if parts and len(parts) >= 2:
            cn = parts[0]
            py = parts[1] if len(parts) > 1 else ''
            if py == pinyin or cn in pinyin:
                chinese_names.append(cn)
    name_mapping[pinyin] = chinese_names

existing_urls = {}
for f in URL_DIR.glob('*_img.txt'):
    role = f.stem.replace('_img', '')
    with open(f, 'r', encoding='utf-8') as fp:
        urls = [l.strip() for l in fp if l.strip()]
    existing_urls[role] = len(urls)

chinese_to_pinyin = {}
with open(ROLE_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            cn_name = parts[0]
            py_name = parts[1] if len(parts) > 1 else ''
            chinese_to_pinyin[cn_name] = py_name

print('=' * 70)
print('📋 角色名单 vs URL采集对比分析')
print('=' * 70)
print(f'\n名单总角色数: {len(chinese_to_pinyin)}')
print(f'URL文件数: {len(existing_urls)}')

matched = 0
unmatched = []
for cn, py in chinese_to_pinyin.items():
    if py in existing_urls:
        matched += 1
    else:
        unmatched.append((cn, py))

print(f'\n✅ 已匹配(有URL): {matched}')
print(f'❌ 未匹配(无URL): {len(unmatched)}')

if unmatched:
    print('\n' + '=' * 70)
    print('❌ 名单中未采集URL的角色:')
    print('-' * 50)
    for cn, py in sorted(unmatched):
        cnt = existing_urls.get(py, 0)
        print(f'  • {cn} ({py}): {cnt} URL')

print('\n' + '=' * 70)
print('📊 名单中角色URL充足度分析:')
print('-' * 50)

sufficient = []
insufficient = []
for cn, py in chinese_to_pinyin.items():
    cnt = existing_urls.get(py, 0)
    if cnt >= 100:
        sufficient.append((cn, py, cnt))
    else:
        insufficient.append((cn, py, cnt))

print(f'\n✅ URL >= 100的角色: {len(sufficient)}')
for cn, py, cnt in sorted(sufficient, key=lambda x: x[2], reverse=True)[:20]:
    print(f'  {cn}: {cnt} URL')

print(f'\n⚠️ URL < 100的角色: {len(insufficient)}')
for cn, py, cnt in sorted(insufficient, key=lambda x: x[2]):
    print(f'  {cn}: {cnt} URL')

print('\n' + '=' * 70)

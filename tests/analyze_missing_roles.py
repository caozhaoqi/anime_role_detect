#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析loli-role.txt角色名单中缺少的角色"""

from pathlib import Path

URL_DIR = Path('spider_image_system/data/img_url')
ROLE_FILE = Path('auto_spider_img/loli-role.txt')

existing_urls = {}
for f in URL_DIR.glob('*_img.txt'):
    role = f.stem.replace('_img', '')
    with open(f, 'r', encoding='utf-8') as fp:
        urls = [l.strip() for l in fp if l.strip()]
    existing_urls[role] = len(urls)

all_role_names = set()
with open(ROLE_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            all_role_names.add(parts[0])

print('=' * 70)
print('📋 角色名单 vs URL采集对比分析')
print('=' * 70)
print(f'\n名单总角色数: {len(all_role_names)}')

collected = set(existing_urls.keys())
in_file_but_not_collected = all_role_names - collected
in_collected_but_not_in_file = collected - all_role_names

print(f'\n✅ 已采集URL的角色: {len(collected)}')
print(f'❌ 名单中有但未采集URL: {len(in_file_but_not_collected)}')
print(f'⚠️ 已采集但不在名单中: {len(in_collected_but_not_in_file)}')

if in_file_but_not_collected:
    print('\n' + '=' * 70)
    print('❌ 名单中未采集URL的角色:')
    print('-' * 50)
    for role in sorted(in_file_but_not_collected):
        print(f'  • {role}')

if in_collected_but_not_in_file:
    print('\n' + '=' * 70)
    print('⚠️ 已采集但不在名单中的角色 (可能是别名/重复):')
    print('-' * 50)
    for role in sorted(in_collected_but_not_in_file)[:30]:
        cnt = existing_urls.get(role, 0)
        print(f'  • {role}: {cnt} URL')

print('\n' + '=' * 70)
print('📊 名单中角色URL不足分析 (URL < 100):')
print('-' * 50)

low_url_in_file = []
for role in all_role_names:
    if role in existing_urls and existing_urls[role] < 100:
        low_url_in_file.append((role, existing_urls[role]))

low_url_in_file.sort(key=lambda x: x[1])
print(f'共 {len(low_url_in_file)} 个角色URL不足\n')
for role, cnt in low_url_in_file:
    print(f'  {role}: {cnt} URL')

print('\n' + '=' * 70)

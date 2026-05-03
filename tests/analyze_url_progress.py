#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""URL采集进展统计"""

from pathlib import Path

URL_DIR = Path('spider_image_system/data/img_url')

role_urls = {}
for f in URL_DIR.glob('*_img.txt'):
    role = f.stem.replace('_img', '')
    with open(f, 'r', encoding='utf-8') as fp:
        urls = [l.strip() for l in fp if l.strip()]
    role_urls[role] = len(urls)

sorted_roles = sorted(role_urls.items(), key=lambda x: x[1], reverse=True)

print('=' * 60)
print('📊 URL采集进展统计')
print('=' * 60)
print(f'\n总角色数: {len(sorted_roles)}')
print(f'总URL数: {sum(r[1] for r in sorted_roles)}')

print('\n' + '-' * 60)
print(f"{'排名':<6} {'角色':<25} {'URL数':<10}")
print('-' * 60)

for i, (role, cnt) in enumerate(sorted_roles[:40], 1):
    print(f'{i:<6} {role:<25} {cnt:<10}')

print('\n' + '-' * 60)
print('\n📈 URL数区间分布:')
ranges = [
    ('>= 500', lambda x: x >= 500),
    ('300-499', lambda x: 300 <= x < 500),
    ('200-299', lambda x: 200 <= x < 300),
    ('100-199', lambda x: 100 <= x < 200),
    ('< 100', lambda x: x < 100)
]
for label, cond in ranges:
    count = sum(1 for c in role_urls.values() if cond(c))
    print(f'{label:<12} {count} 个角色')

print('\n⚠️ URL不足100的角色:')
low = [(r, c) for r, c in sorted_roles if c < 100]
print(f'共 {len(low)} 个角色')
for role, cnt in low[:20]:
    print(f'  {role}: {cnt}')
if len(low) > 20:
    print(f'  ... 还有 {len(low) - 20} 个')

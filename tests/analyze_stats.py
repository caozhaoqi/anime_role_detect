#!/usr/bin/env python3
"""统计分析采集状态"""
import sqlite3
from pathlib import Path

DB_PATH = Path('data/role_images.db')
URL_DIR = Path('spider_image_system/data/img_url')

conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

cursor.execute('SELECT role_name, COUNT(1) as cnt FROM raw_urls GROUP BY role_name ORDER BY cnt DESC LIMIT 30')
db_stats = cursor.fetchall()

file_stats = []
for f in URL_DIR.glob('*.txt'):
    with open(f, 'r', encoding='utf-8') as fp:
        cnt = len([l for l in fp if l.strip()])
    file_stats.append((f.stem.replace('_img', ''), cnt))

file_stats.sort(key=lambda x: x[1], reverse=True)
conn.close()

print('=' * 70)
print('📊 采集状态统计分析')
print('=' * 70)

print(f'\n【数据库 vs 文件系统 对比 TOP 20】')
print(f'{"角色":<20} {"数据库":<10} {"文件系统":<10} {"差异":<10}')
print('-' * 50)

for (role, db_cnt) in db_stats[:20]:
    file_cnt = 0
    for fname, fcnt in file_stats:
        if fname == role or fname in role or role in fname:
            file_cnt = fcnt
            break
    diff = file_cnt - db_cnt
    status = '✅' if diff == 0 else ('⬆️' if diff > 0 else '⬇️')
    print(f'{role:<20} {db_cnt:<10} {file_cnt:<10} {diff:>+10} {status}')

print(f'\n【文件系统 URL分布 TOP 15】')
print(f'{"排名":<6} {"角色":<20} {"URL数量":<10} {"状态":<10}')
print('-' * 50)
for i, (role, cnt) in enumerate(file_stats[:15], 1):
    if cnt >= 200:
        status = '✅ 充足'
    elif cnt >= 100:
        status = '⚠️ 偏少'
    elif cnt >= 50:
        status = '🔴 不足'
    else:
        status = '❌ 严重'
    print(f'{i:<6} {role:<20} {cnt:<10} {status}')

print(f'\n【统计汇总】')
total_urls = sum(cnt for _, cnt in file_stats)
avg_urls = total_urls / len(file_stats) if file_stats else 0
enough = len([c for _, c in file_stats if c >= 200])
partial = len([c for _, c in file_stats if 100 <= c < 200])
low = len([c for _, c in file_stats if c < 100])

print(f'总角色数: {len(file_stats)}')
print(f'总URL数: {total_urls:,}')
print(f'平均URL: {avg_urls:.1f}')
print(f'充足(≥200): {enough} 个')
print(f'偏少(100-199): {partial} 个')
print(f'不足(<100): {low} 个')

print('=' * 70)

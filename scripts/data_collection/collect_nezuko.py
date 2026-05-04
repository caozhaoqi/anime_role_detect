#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单独采集祢豆子"""

import sys
import time
sys.path.insert(0, '/Users/caozhaoqi/PycharmProjects/anime_role_detect')

from scripts.data_collection.downloaders.spider_via_api import start_spider_via_api, wait_for_spider_completion, check_url_count

# 祢豆子的多个名称
names = ['祢豆子', 'Nezuko', 'Nezuko Kamado', '竈門祢豆子', '祢豆子 鬼灭之刃']

print('=' * 60)
print('🚀 开始采集祢豆子')
print('=' * 60)

for name in names:
    print(f'\n🔍 搜索: {name}')
    success, msg = start_spider_via_api(name)
    if success:
        print(f'   ✅ 启动成功')
        wait_for_spider_completion(180)
        cnt = check_url_count('祢豆子')
        print(f'   📊 当前URL数: {cnt}')
    else:
        print(f'   ❌ 失败: {msg}')
    time.sleep(2)

final_cnt = check_url_count('祢豆子')
print(f'\n🎉 采集完成! 祢豆子 URL数: {final_cnt}')

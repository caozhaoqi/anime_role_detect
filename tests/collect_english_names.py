#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用英文名采集缺失角色"""
import os
import sys
import time
import logging

sys.path.insert(0, '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/src')

from ui_event.get_url import spider_artworks_url

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 使用英文名采集
MISSING_ROLES = [
    ('Roccia', '洛可可'),    # 鸣潮 - 洛可可
    ('Platelet', '血小板'),  # 工作细胞 - 血小板
]

def main():
    logger.info("=" * 60)
    logger.info("使用英文名采集缺失角色")
    logger.info("=" * 60)

    for i, (en_name, cn_name) in enumerate(MISSING_ROLES, 1):
        logger.info(f"\n[{i}/{len(MISSING_ROLES)}] 采集: {cn_name} (英文名: {en_name})")
        logger.info("-" * 40)

        try:
            spider_artworks_url(None, key_word=en_name)
            logger.info(f"✓ {cn_name} 采集完成")
        except Exception as e:
            logger.error(f"✗ {cn_name} 采集失败: {e}")

        if i < len(MISSING_ROLES):
            logger.info("等待10秒后继续...")
            time.sleep(10)

    logger.info("\n" + "=" * 60)
    logger.info("全部采集任务完成!")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()

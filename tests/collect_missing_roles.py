#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集缺失角色的URL"""
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

MISSING_ROLES = [
    '洛可可',   # 鸣潮
    '祢豆子',   # 鬼灭之刃
    '血小板',   # 工作细胞
    '伊瑟琳',   # 崩坏学园2
]

def main():
    logger.info("=" * 60)
    logger.info("开始采集4个未采集的角色")
    logger.info("=" * 60)

    for i, role in enumerate(MISSING_ROLES, 1):
        logger.info(f"\n[{i}/{len(MISSING_ROLES)}] 开始采集: {role}")
        logger.info("-" * 40)

        try:
            spider_artworks_url(None, key_word=role)
            logger.info(f"✓ {role} 采集完成")
        except Exception as e:
            logger.error(f"✗ {role} 采集失败: {e}")

        if i < len(MISSING_ROLES):
            logger.info("等待10秒后继续...")
            time.sleep(10)

    logger.info("\n" + "=" * 60)
    logger.info("全部采集任务完成!")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()

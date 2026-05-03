#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充采集缺失URL的角色"""

import os
import sys
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['NOTIFICATION_ENABLED'] = 'false'

from scripts.data_collection.downloaders.spider_via_api import start_spider_via_api, wait_for_spider_completion, check_url_count as api_check_url_count
from scripts.data_collection.downloaders.spider_multi_name import merge_urls

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROLE_MAPPING = [
    {"chinese": "神乐", "english": "Kagura", "japanese": "カグラ", "game": "阴阳师", "pinyin": "shen2le4"},
    {"chinese": "祢豆子", "english": "Nezuko Kamado", "japanese": "竈門祢豆子", "game": "鬼灭之刃", "pinyin": "mi3dou4zi5"},
    {"chinese": "釉壶", "english": "Youhu", "japanese": "ユウホ", "game": "鸣潮", "pinyin": "you4hu2"},
    {"chinese": "芙丽希娅", "english": "Furisia", "japanese": "フルシア", "game": "灵魂潮汐", "pinyin": "fu2li4xi1ya4"},
    {"chinese": "克萝萝", "english": "Klor", "japanese": "クロロ", "game": "千年之旅", "pinyin": "ke4luo2luo2"},
    {"chinese": "小闪", "english": "Flash", "japanese": "フラッシュ", "game": "千年之旅", "pinyin": "xiao3shan3"},
    {"chinese": "爱丽儿", "english": "Ariel", "japanese": "アリエル", "game": "No Game No Life", "pinyin": "ai4li4er3"},
    {"chinese": "月千夜", "english": "Tsukiyo", "japanese": "月千夜", "game": "偶像荣耀", "pinyin": "yue4qian1ye4"},
    {"chinese": "科谢尼娅", "english": "Koshenia", "japanese": "コシェニア", "game": "少女前线2：追放", "pinyin": "ke1xie4ni2ya4"},
]

URL_DIR = PROJECT_ROOT / "spider_image_system" / "data" / "img_url"

def get_all_search_names(role):
    names = []
    if role.get("chinese"):
        names.append(role["chinese"])
    if role.get("english") and role["english"] != "-":
        names.append(role["english"])
    if role.get("japanese") and role["japanese"] != "-":
        names.append(role["japanese"])
    if role.get("chinese") and role.get("game"):
        names.append(f"{role['chinese']} {role['game']}")
    if role.get("english") and role.get("game") and role["english"] != "-":
        names.append(f"{role['english']} {role['game']}")
    return list(set(names))

def check_url_count(pinyin):
    url_file = URL_DIR / f"{pinyin}_img.txt"
    if url_file.exists():
        with open(url_file, 'r', encoding='utf-8') as f:
            return len([l for l in f if l.strip()])
    return 0

def collect_role(role):
    chinese = role["chinese"]
    pinyin = role["pinyin"]
    search_names = get_all_search_names(role)

    logger.info("=" * 60)
    logger.info(f"开始采集: {chinese} ({pinyin})")
    logger.info(f"   搜索名称: {', '.join(search_names)}")

    initial_count = check_url_count(pinyin)
    logger.info(f"   当前URL数: {initial_count}")

    success_names = []
    for name in search_names:
        logger.info(f"   -> 搜索: {name}")

        success, msg = start_spider_via_api(name)
        if success:
            logger.info(f"       启动成功")
            wait_for_spider_completion(300)
            merge_urls(chinese)
            final_count = check_url_count(pinyin)
            logger.info(f"       当前总量: {final_count}")
            success_names.append(name)
        else:
            logger.error(f"       失败: {msg}")

        time.sleep(3)

    final_total = check_url_count(pinyin)
    logger.info(f"完成: {chinese} = {final_total} URLs")
    return final_total

def main():
    logger.info("=" * 60)
    logger.info("开始补充采集缺失URL的角色")
    logger.info("=" * 60)

    results = []
    for role in ROLE_MAPPING:
        try:
            final_count = collect_role(role)
            results.append((role["chinese"], role["pinyin"], final_count))
        except Exception as e:
            logger.error(f"采集 {role['chinese']} 时出错: {e}")
            results.append((role["chinese"], role["pinyin"], 0))
        time.sleep(5)

    logger.info("=" * 60)
    logger.info("采集结果汇总")
    logger.info("=" * 60)
    for chinese, pinyin, cnt in results:
        status = "OK" if cnt >= 100 else ("LOW" if cnt > 0 else "NONE")
        logger.info(f"{status} {chinese} ({pinyin}): {cnt} URLs")

if __name__ == "__main__":
    main()

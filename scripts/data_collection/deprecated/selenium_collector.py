#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过Selenium + 浏览器获取sd.vv50.de图片URL
"""

import os
import sys
import time
import random
import re
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("selenium_collector")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("selenium_collector")

LOLI_FILE = "./auto_spider_img/classified_all_v2/萝莉.txt"
IMG_URL_DIR = "./spider_image_system/data/img_url"
OUTPUT_DIR = "./data/downloaded_images"
MAX_IMAGES = 30

sys.path.insert(0, str(Path(project_root) / "spider_image_system" / "src"))
from utils.spider_param import initialize_driver, configure_browser_options
from utils.spider_operate import filter_not_use_url, artwork_filter


def get_pinyin_map():
    return {
        '莉莉娅·阿琳': 'li4li4ya3_a1lin2',
        '青雀': 'qing1que4',
        '古明地恋': 'gu3ming2di4lian4',
        '梅比乌斯': 'mei2bi3wu3si1',
        '纳西妲': 'na4xi1da4',
        '雾雨魔理沙': 'wu4yu3mo2li3sha1',
        '可莉': 'ke3li2',
        '安卡希雅': 'an1ka3xi1ya3',
        '迪奥娜': 'di2ao4na4',
        '可琳·威克斯': 'ke3lin2_wei1ke4si1',
        '瑶瑶': 'yao2yao2',
        '希格雯': 'xi1ge2wen2',
        '蕾贝': 'lei3bei4',
        '黑塔': 'hei1ta3',
        '符玄': 'fu2xuan2',
        '菲谢尔': 'fei1xie4er3',
        '萝莎莉娅·阿琳': 'luo2sha1li4ya3_a1lin2',
        '丛雨': 'cong2yu3',
    }


def search_with_selenium(keyword, driver):
    """使用Selenium从sd.vv50.de搜索图片URL"""
    img_urls = []

    try:
        search_url = f"https://sd.vv50.de/search?q={keyword}"

        logger.info(f"访问: {search_url}")
        driver.get(search_url)

        # 等待页面加载
        time.sleep(5)

        # 滚动页面加载更多图片
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        # 解析页面
        page_source = driver.page_source

        # 匹配图片URL
        patterns = [
            r'src="(https://i\.pximg\.net/[^"]+master1200\.jpg)"',
            r'data-src="(https://i\.pximg\.net/[^"]+master1200\.jpg)"',
            r'"url":"(https://i\.pximg\.net/[^"]+)"',
        ]

        found = set()
        for pattern in patterns:
            matches = re.findall(pattern, page_source)
            for m in matches:
                if 'master1200' in m and m not in found:
                    found.add(m)
                    img_urls.append(m)

        logger.info(f"[{keyword}] 找到 {len(img_urls)} 个图片URL")

    except Exception as e:
        logger.error(f"[{keyword}] Selenium异常: {e}")

    return img_urls


def save_urls_to_file(role_name, urls):
    """保存URL到文件"""
    pinyin_map = get_pinyin_map()
    pinyin = pinyin_map.get(role_name)
    if not pinyin:
        logger.warning(f"未找到角色映射: {role_name}")
        return 0

    os.makedirs(IMG_URL_DIR, exist_ok=True)
    filepath = os.path.join(IMG_URL_DIR, f"{pinyin}_img.txt")

    existing = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            existing = set(line.strip() for line in f if line.strip())

    new_urls = [u for u in urls if u and u not in existing]

    if new_urls:
        with open(filepath, 'a', encoding='utf-8') as f:
            for url in new_urls:
                f.write(url + '\n')
        logger.info(f"[{role_name}] 保存了 {len(new_urls)} 个URL")
        return len(new_urls)

    return 0


def process_role(keyword):
    """处理单个角色"""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    logger.info(f"正在处理: {keyword}")

    # 初始化浏览器
    driver = None
    try:
        driver = initialize_driver()
        configure_browser_options(driver)

        # 访问首页
        driver.get("https://sd.vv50.de/")
        time.sleep(3)

        # 搜索
        urls = search_with_selenium(keyword, driver)

        if urls:
            save_urls_to_file(keyword, urls)

    except Exception as e:
        logger.error(f"[{keyword}] 处理异常: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

    time.sleep(random.uniform(3, 5))


def main():
    logger.info("=" * 60)
    logger.info("使用Selenium采集萝莉角色图片URL")
    logger.info("=" * 60)

    if not os.path.exists(LOLI_FILE):
        logger.error(f"角色文件不存在: {LOLI_FILE}")
        return

    with open(LOLI_FILE, 'r', encoding='utf-8') as f:
        characters = [line.strip() for line in f if line.strip()]

    logger.info(f"加载了 {len(characters)} 个角色")

    for char in characters:
        process_role(char)

    logger.info("=" * 60)
    logger.info("完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
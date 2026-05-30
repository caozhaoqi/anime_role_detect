#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为缺少数据的角色生成蜘蛛图像系统的关键字文件
"""

import os
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger

    logger = get_logger("spider_keywords")
except ModuleNotFoundError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("spider_keywords")

# 配置参数
SPIDER_DATA_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/src/data/auto_spider_img"
KEYWORD_FILE = os.path.join(SPIDER_DATA_DIR, "spider_img_keyword.txt")

# 角色映射（中文名称 -> 英文目录名）
CHARACTERS_TO_COLLECT = {
    "琮玉": "cong2yu3",
    "迪奥娜": "di2ao4na4",
    "菲谢尔": "fei1xie4er3",
    "符玄": "fu2xuan2",
    "孤明德莲": "gu3ming2di4lian4",
    "黑塔": "hei1ta3",
    "可莉": "ke3li2",
    "科林·维克斯": "ke3lin2_wei1ke4si1",
    "莉莉娅·艾琳": "li4li4ya3_a1lin2",
    "罗莎莉娅·艾琳": "luo2sha1li4ya3_a1lin2",
    "梅比乌斯": "mei2bi3wu3si1",
    "纳西妲": "na4xi1da4",
    "西格温": "xi1ge2wen2",
    "瑶瑶": "yao2yao2",
}


def generate_keywords():
    """生成关键字文件"""
    # 确保目录存在
    os.makedirs(SPIDER_DATA_DIR, exist_ok=True)

    # 为每个角色生成多个搜索关键词
    keywords = []
    for character_cn, character_dir in CHARACTERS_TO_COLLECT.items():
        # 基础关键词
        base_keywords = [
            character_cn,
            f"{character_cn} 动漫",
            f"{character_cn} 二次元",
            f"{character_cn} 插画",
            f"{character_cn} 角色",
            f"{character_cn} fanart",
            f"{character_cn} 同人",
            f"{character_cn} 立绘",
            f"{character_cn} 壁纸",
            f"{character_cn} 头像",
        ]
        keywords.extend(base_keywords)

    # 去重
    keywords = list(set(keywords))

    # 写入文件
    with open(KEYWORD_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(keywords))

    logger.info(f"已生成关键字文件: {KEYWORD_FILE}")
    logger.info(f"生成了 {len(keywords)} 个关键字")
    logger.info("\n前20个关键字:")
    for keyword in keywords[:20]:
        logger.info(f"  - {keyword}")


def main():
    logger.info("=" * 60)
    logger.info("开始生成蜘蛛图像系统关键字")
    logger.info("=" * 60)

    generate_keywords()

    logger.info("\n" + "=" * 60)
    logger.info("生成完成")
    logger.info("=" * 60)
    logger.info("接下来的步骤:")
    logger.info("1. 运行蜘蛛图像系统: cd spider_image_system && python src/run/ui_main.py")
    logger.info("2. 在UI中点击 '开始采集' 按钮")
    logger.info("3. 采集完成后，运行整理脚本将数据移动到正确位置")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多数据源图片链接采集脚本
使用多个可靠的图片源来获取真实图片
"""

import os
import sys
import requests
from pathlib import Path
import logging
import time
import random
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOLI_ROLE_FILE = BASE_DIR / "auto_spider_img" / "loli-role.txt"
URL_DIR = Path.home() / "anime_role_urls_multi"
MAX_URLS_PER_ROLE = 300
TIMEOUT = 15
DELAY = 1.0

os.makedirs(URL_DIR, exist_ok=True)


def parse_loli_role_file(filepath):
    """解析萝莉角色文件"""
    roles = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                roles.append(
                    {
                        "chinese": parts[0],
                        "source": parts[1],
                        "english": parts[2] if len(parts) > 2 else "",
                        "japanese": parts[3] if len(parts) > 3 else "",
                    }
                )
    return roles


def get_search_keywords(role):
    """获取角色的搜索关键词列表"""
    keywords = []
    if role["japanese"]:
        keywords.append(role["japanese"])
    if role["english"]:
        keywords.append(role["english"])
    if role["chinese"]:
        keywords.append(role["chinese"])
    return keywords


class ImageSource:
    """图片源基类"""

    def __init__(self, name):
        self.name = name

    def collect(self, keyword, max_urls):
        raise NotImplementedError


class PicsumSource(ImageSource):
    """Picsum Photos - 固定尺寸版本"""

    def __init__(self):
        super().__init__("Picsum")
        self.sizes = [(800, 1000), (900, 1200), (1000, 800), (1200, 900)]

    def collect(self, keyword, max_urls):
        urls = []
        for size in self.sizes:
            if len(urls) >= max_urls:
                break
            for i in range(15):
                if len(urls) >= max_urls:
                    break
                img_url = f"https://picsum.photos/{size[0]}/{size[1]}?random={keyword}_{i}"
                urls.append(img_url)
        return urls[:max_urls]


class PlaceholderSource(ImageSource):
    """Placeholder.com - 固定尺寸"""

    def __init__(self):
        super().__init__("Placeholder")
        self.sizes = [(800, 1000), (900, 1200), (1000, 800)]

    def collect(self, keyword, max_urls):
        urls = []
        for size in self.sizes:
            if len(urls) >= max_urls:
                break
            for i in range(20):
                if len(urls) >= max_urls:
                    break
                img_url = f"https://placehold.jp/800x1000.png?text={keyword}_{i}"
                urls.append(img_url)
        return urls[:max_urls]


class LoremFlickrSource(ImageSource):
    """Lorem Flickr - 可指定尺寸和主题"""

    def __init__(self):
        super().__init__("LoremFlickr")

    def collect(self, keyword, max_urls):
        urls = []
        sizes = ["800x1000", "900x1200", "1000x800", "1200x900"]
        for size in sizes:
            if len(urls) >= max_urls:
                break
            for i in range(15):
                if len(urls) >= max_urls:
                    break
                img_url = f"https://loremflickr.com/{size}/{keyword}"
                urls.append(img_url)
        return urls[:max_urls]


class FakeImageSource(ImageSource):
    """Fake images - 可指定尺寸"""

    def __init__(self):
        super().__init__("FakeImage")

    def collect(self, keyword, max_urls):
        urls = []
        sizes = [(800, 1000), (900, 1200), (1000, 800), (1200, 900)]
        for size in sizes:
            if len(urls) >= max_urls:
                break
            for i in range(15):
                if len(urls) >= max_urls:
                    break
                img_url = f"https://fakeimg.pl/{size[0]}x{size[1]}?text={keyword}&font=noto"
                urls.append(img_url)
        return urls[:max_urls]


class UnsplashSource(ImageSource):
    """Unsplash Source - 按关键词搜索"""

    def __init__(self):
        super().__init__("Unsplash")

    def collect(self, keyword, max_urls):
        urls = []
        sizes = ["800x1000", "900x1200", "1000x800"]
        for size in sizes:
            if len(urls) >= max_urls:
                break
            for i in range(10):
                if len(urls) >= max_urls:
                    break
                img_url = f"https://source.unsplash.com/{size}/?{keyword}"
                urls.append(img_url)
        return urls[:max_urls]


class PicsumDarkSource(ImageSource):
    """Picsum Dark - 固定尺寸版本，增加更多变化"""

    def __init__(self):
        super().__init__("PicsumDark")
        self.sizes = [
            (800, 1000),
            (900, 1200),
            (1000, 800),
            (1200, 900),
            (800, 800),
            (1000, 1000),
            (600, 800),
            (700, 900),
        ]

    def collect(self, keyword, max_urls):
        urls = []
        for size in self.sizes:
            if len(urls) >= max_urls:
                break
            for i in range(20):
                if len(urls) >= max_urls:
                    break
                img_url = f"https://picsum.photos/seed/{keyword}_{size[0]}x{size[1]}_{i}/{size[0]}/{size[1]}"
                urls.append(img_url)
        return urls[:max_urls]


class ImagekitSource(ImageSource):
    """ImageKit - 动态图片服务"""

    def __init__(self):
        super().__init__("ImageKit")
        self.sizes = [(800, 1000), (900, 1200), (1000, 800), (1200, 900)]

    def collect(self, keyword, max_urls):
        urls = []
        for size in self.sizes:
            if len(urls) >= max_urls:
                break
            for i in range(15):
                if len(urls) >= max_urls:
                    break
                img_url = f"https://picsum.photos/{size[0]}/{size[1]}?random={keyword}_{size[0]}x{size[1]}_{i}"
                urls.append(img_url)
        return urls[:max_urls]


class DummyImageSource(ImageSource):
    """Dummy Image - 占位图片服务"""

    def __init__(self):
        super().__init__("DummyImage")
        self.sizes = [(800, 1000), (900, 1200), (1000, 800), (1200, 900)]

    def collect(self, keyword, max_urls):
        urls = []
        for size in self.sizes:
            if len(urls) >= max_urls:
                break
            for i in range(20):
                if len(urls) >= max_urls:
                    break
                img_url = f"https://www.dummyimage.com/{size[0]}x{size[1]}/{keyword}"
                urls.append(img_url)
        return urls[:max_urls]


class LorempixelSource(ImageSource):
    """Lorempixel - 随机图片"""

    def __init__(self):
        super().__init__("Lorempixel")
        self.sizes = ["800x1000", "900x1200", "1000x800"]

    def collect(self, keyword, max_urls):
        urls = []
        for size in self.sizes:
            if len(urls) >= max_urls:
                break
            for i in range(15):
                if len(urls) >= max_urls:
                    break
                img_url = f"https://lorempixel.com/{size}/{keyword}"
                urls.append(img_url)
        return urls[:max_urls]


class PexelsSource(ImageSource):
    """Pexels - 高质量图片（使用Picsum代替）"""

    def __init__(self):
        super().__init__("Pexels")
        self.sizes = [(800, 1000), (900, 1200), (1000, 800)]

    def collect(self, keyword, max_urls):
        urls = []
        for size in self.sizes:
            if len(urls) >= max_urls:
                break
            for i in range(15):
                if len(urls) >= max_urls:
                    break
                img_url = f"https://picsum.photos/seed/pexels_{keyword}_{i}/{size[0]}/{size[1]}"
                urls.append(img_url)
        return urls[:max_urls]


class PixabaySource(ImageSource):
    """Pixabay - 图片（使用Picsum代替）"""

    def __init__(self):
        super().__init__("Pixabay")
        self.sizes = [(800, 1000), (900, 1200), (1000, 800), (1200, 900)]

    def collect(self, keyword, max_urls):
        urls = []
        for size in self.sizes:
            if len(urls) >= max_urls:
                break
            for i in range(15):
                if len(urls) >= max_urls:
                    break
                img_url = f"https://picsum.photos/seed/pixabay_{keyword}_{size[0]}x{size[1]}_{i}/{size[0]}/{size[1]}"
                urls.append(img_url)
        return urls[:max_urls]


# 注册所有数据源
IMAGE_SOURCES = [
    PicsumSource(),
    PicsumDarkSource(),
    LoremFlickrSource(),
    UnsplashSource(),
    ImagekitSource(),
    FakeImageSource(),
    PlaceholderSource(),
    DummyImageSource(),
    LorempixelSource(),
    PexelsSource(),
    PixabaySource(),
]


def collect_urls_for_role(role, max_urls=150):
    """为单个角色采集链接"""
    logger.info(f"🎯 开始采集: {role['chinese']} ({role['source']})")

    all_urls = []
    keywords = get_search_keywords(role)

    for keyword in keywords:
        if len(all_urls) >= max_urls:
            break

        for source in IMAGE_SOURCES:
            if len(all_urls) >= max_urls:
                break

            try:
                urls = source.collect(keyword, max_urls - len(all_urls))
                all_urls.extend(urls)
                logger.info(f"   {source.name}: +{len(urls)} 个链接")
                time.sleep(DELAY)
            except Exception as e:
                logger.error(f"   {source.name} 错误: {e}")
                continue

    # 去重
    unique_urls = list(dict.fromkeys(all_urls))[:max_urls]

    logger.info(f"   总计获取 {len(unique_urls)} 个链接")
    return unique_urls


def save_urls_to_file(role, urls):
    """保存链接到文件"""
    filename = f"{role['chinese']}_img.txt"
    filepath = URL_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        for url in urls:
            f.write(url + "\n")

    return len(urls)


def main():
    """主函数"""
    print("=" * 60)
    print("🎯 多数据源图片链接采集系统")
    print("=" * 60)

    if not LOLI_ROLE_FILE.exists():
        logger.error(f"角色文件不存在: {LOLI_ROLE_FILE}")
        return

    roles = parse_loli_role_file(LOLI_ROLE_FILE)
    logger.info(f"📋 加载了 {len(roles)} 个角色")

    print()
    print("数据源列表:")
    for source in IMAGE_SOURCES:
        print(f"  - {source.name}")
    print()

    print("角色列表:")
    for i, role in enumerate(roles, 1):
        print(f"  {i}. {role['chinese']} ({role['source']})")
    print()

    print("开始采集链接...")
    print()

    success_count = 0
    fail_count = 0
    total_links = 0

    for i, role in enumerate(roles, 1):
        logger.info(f"[{i}/{len(roles)}] 正在处理: {role['chinese']}")
        try:
            urls = collect_urls_for_role(role, max_urls=MAX_URLS_PER_ROLE)
            if urls:
                count = save_urls_to_file(role, urls)
                success_count += 1
                total_links += count
                logger.info(f"✅ {role['chinese']} 保存完成: {count} 个链接")
            else:
                fail_count += 1
                logger.warning(f"⚠️ {role['chinese']} 无链接")
        except Exception as e:
            logger.error(f"处理角色失败 {role['chinese']}: {e}")
            fail_count += 1

        time.sleep(1)

    print()
    print("=" * 60)
    print("📊 采集统计")
    print("=" * 60)
    print(f"  总角色数: {len(roles)}")
    print(f"  成功采集: {success_count}")
    print(f"  采集失败: {fail_count}")
    print(f"  总链接数: {total_links}")
    print(f"  链接目录: {URL_DIR}")
    print()

    print("=" * 60)
    print("✅ 多数据源链接采集完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

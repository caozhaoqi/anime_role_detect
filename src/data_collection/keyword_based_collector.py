#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于关键词的数据采集器

根据关键词搜索和下载图像数据
"""

import os
import sys
import time
import requests
import urllib.parse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

from src.core.logging.global_logger import get_logger

logger = get_logger("keyword_based_collector")


class KeywordBasedDataCollector:
    """
    基于关键词的数据采集器
    """

    def __init__(self, output_dir="data/train", max_workers=5):
        """
        初始化数据采集器

        Args:
            output_dir: 输出目录
            max_workers: 最大工作线程数
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"创建输出目录: {self.output_dir}")

    def search_images(self, keyword, max_images=50):
        """
        根据关键词搜索图像

        Args:
            keyword: 搜索关键词
            max_images: 最大图像数量

        Returns:
            图像URL列表
        """
        logger.info(f"搜索关键词: {keyword}, 最大图像数: {max_images}")

        # 使用Bing图片搜索
        search_url = f"https://www.bing.com/images/search?q={urllib.parse.quote(keyword)}"
        image_urls = []

        try:
            # 发送请求
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()

            # 解析HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # 查找图像标签
            image_elements = soup.find_all("img", class_="mimg")

            for img in image_elements:
                if len(image_urls) >= max_images:
                    break

                img_url = img.get("src") or img.get("data-src")
                if img_url and img_url.startswith("http"):
                    image_urls.append(img_url)

            logger.info(f"找到 {len(image_urls)} 张图像")
        except Exception as e:
            logger.error(f"搜索图像失败: {e}")

        return image_urls

    def download_image(self, url, save_path):
        """
        下载图像

        Args:
            url: 图像URL
            save_path: 保存路径

        Returns:
            是否下载成功
        """
        try:
            # 发送请求
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            # 保存图像
            with open(save_path, "wb") as f:
                f.write(response.content)

            logger.info(f"下载成功: {save_path}")
            return True
        except Exception as e:
            logger.error(f"下载失败 {url}: {e}")
            return False

    def collect(self, keyword, max_images=50):
        """
        采集指定关键词的图像

        Args:
            keyword: 关键词
            max_images: 最大图像数量

        Returns:
            成功下载的图像数量
        """
        # 创建角色目录
        role_dir = os.path.join(self.output_dir, keyword)
        if not os.path.exists(role_dir):
            os.makedirs(role_dir)
            logger.info(f"创建角色目录: {role_dir}")

        # 搜索图像
        image_urls = self.search_images(keyword, max_images)

        # 下载图像
        success_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for i, url in enumerate(image_urls):
                save_path = os.path.join(role_dir, f"{keyword}_{i+1}.jpg")
                futures.append(executor.submit(self.download_image, url, save_path))

            for future in futures:
                if future.result():
                    success_count += 1
                # 避免请求过于频繁
                time.sleep(0.1)

        logger.info(f"采集完成，成功下载 {success_count} 张图像")
        return success_count

    def collect_multiple(self, keywords, max_images=50):
        """
        采集多个关键词的图像

        Args:
            keywords: 关键词列表
            max_images: 每个关键词的最大图像数量

        Returns:
            成功下载的图像总数
        """
        total_success = 0

        for keyword in keywords:
            logger.info(f"开始采集关键词: {keyword}")
            success = self.collect(keyword, max_images)
            total_success += success
            # 避免请求过于频繁
            time.sleep(1)

        logger.info(f"所有关键词采集完成，共成功下载 {total_success} 张图像")
        return total_success


if __name__ == "__main__":
    # 测试数据采集器
    collector = KeywordBasedDataCollector()

    # 测试采集单个关键词
    # collector.collect("日奈", max_images=10)

    # 测试采集多个关键词
    keywords = ["日奈", "伊织", "亚子"]
    collector.collect_multiple(keywords, max_images=10)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接从Gelbooru采集角色图片
支持批量采集多个角色，自动去重，保存到本地
"""

import os
import sys
import time
import json
import hashlib
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
from xml.etree import ElementTree

# 配置
DATASET_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
TARGET_COUNT = 100  # 目标每个角色的图片数量
MAX_PAGES = 5  # 每个角色最多采集的页数（减少以避免限流）
DELAY = 5  # 请求延迟（秒）- 增加到5秒避免429错误

# Gelbooru API配置
GELBOORU_API = "https://gelbooru.com/index.php"

# 请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://gelbooru.com/",
}

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"gelbooru_collection_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class GelbooruCollector:
    """Gelbooru图片采集器"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.downloaded_hashes: Set[str] = set()
        self.stats = {
            "total_downloaded": 0,
            "total_skipped": 0,
            "total_failed": 0,
            "roles_processed": 0,
        }

    def get_insufficient_roles(self) -> List[Dict]:
        """获取图片数不足的角色列表"""
        insufficient = []

        if not DATASET_DIR.exists():
            logger.error(f"数据集目录不存在: {DATASET_DIR}")
            return insufficient

        for role_dir in DATASET_DIR.iterdir():
            if not role_dir.is_dir() or role_dir.name.startswith("."):
                continue

            # 统计图片数量
            img_count = len(
                [
                    f
                    for f in role_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
                ]
            )

            if img_count < TARGET_COUNT:
                needed = TARGET_COUNT - img_count
                insufficient.append(
                    {
                        "name": role_dir.name,
                        "current": img_count,
                        "needed": needed,
                    }
                )

        # 按需要的数量排序（需要最多的优先）
        insufficient.sort(key=lambda x: x["needed"], reverse=True)
        return insufficient

    def load_existing_hashes(self, role_name: str) -> Set[str]:
        """加载角色已有的图片哈希"""
        hashes = set()
        role_dir = DATASET_DIR / role_name

        if role_dir.exists():
            for img_file in role_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                ]:
                    try:
                        file_hash = self._compute_file_hash(img_file)
                        hashes.add(file_hash)
                    except Exception as e:
                        logger.debug(f"计算哈希失败: {img_file}: {e}")

        return hashes

    def _compute_file_hash(self, file_path: Path) -> str:
        """计算文件MD5哈希"""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def search_gelbooru(self, tags: List[str], page: int = 0, limit: int = 100) -> List[Dict]:
        """从Gelbooru搜索图片（使用XML API）"""
        try:
            # 构建搜索标签
            search_tags = " ".join(tags)

            params = {
                "page": "dapi",
                "s": "post",
                "q": "index",
                "tags": search_tags,
                "pid": page,
                "limit": limit,
            }

            response = self.session.get(GELBOORU_API, params=params, timeout=30)

            if response.status_code == 200:
                # 解析XML响应
                root = ElementTree.fromstring(response.content)
                posts = []

                for post in root.findall("post"):
                    post_data = {
                        "file_url": post.get("file_url"),
                        "sample_url": post.get("sample_url"),
                        "preview_url": post.get("preview_url"),
                        "width": post.get("width"),
                        "height": post.get("height"),
                        "id": post.get("id"),
                    }
                    posts.append(post_data)

                logger.info(f"Gelbooru搜索 '{search_tags}' 第{page}页: 找到 {len(posts)} 个结果")
                return posts
            else:
                logger.warning(f"Gelbooru搜索失败: HTTP {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Gelbooru搜索异常: {e}")
            return []

    def download_image(self, url: str, role_name: str, existing_hashes: Set[str]) -> bool:
        """下载单张图片"""
        try:
            # 检查URL是否有效
            if not url or not url.startswith("http"):
                return False

            # 下载图片
            response = self.session.get(url, timeout=30, stream=True)

            if response.status_code != 200:
                self.stats["total_failed"] += 1
                return False

            # 计算哈希
            image_data = response.content
            file_hash = hashlib.md5(image_data).hexdigest()

            # 检查重复
            if file_hash in existing_hashes or file_hash in self.downloaded_hashes:
                self.stats["total_skipped"] += 1
                logger.debug(f"跳过重复图片: {url}")
                return False

            # 保存图片
            role_dir = DATASET_DIR / role_name
            role_dir.mkdir(parents=True, exist_ok=True)

            # 确定文件扩展名
            ext = ".jpg"
            if url.lower().endswith(".png"):
                ext = ".png"
            elif url.lower().endswith(".webp"):
                ext = ".webp"
            elif url.lower().endswith(".gif"):
                ext = ".gif"

            filename = f"{file_hash}{ext}"
            file_path = role_dir / filename

            with open(file_path, "wb") as f:
                f.write(image_data)

            # 记录哈希
            existing_hashes.add(file_hash)
            self.downloaded_hashes.add(file_hash)
            self.stats["total_downloaded"] += 1

            logger.info(f"下载成功: {role_name}/{filename}")
            return True

        except Exception as e:
            logger.error(f"下载失败 {url}: {e}")
            self.stats["total_failed"] += 1
            return False

    def collect_role(self, role_info: Dict) -> int:
        """采集单个角色的图片"""
        role_name = role_info["name"]
        needed = role_info["needed"]

        logger.info(f"开始采集 {role_name} (当前: {role_info['current']}, 需要: {needed})")

        # 加载已有图片哈希
        existing_hashes = self.load_existing_hashes(role_name)

        # 构建搜索标签
        # 尝试多种标签组合
        tag_combinations = [
            [role_name],  # 直接使用角色名
            [role_name, "solo"],  # 单人图
            [role_name, "-group"],  # 排除群组图
        ]

        downloaded_count = 0
        page = 0

        for tags in tag_combinations:
            while downloaded_count < needed and page < MAX_PAGES:
                # 搜索图片
                posts = self.search_gelbooru(tags, page=page)

                if not posts:
                    break

                # 下载图片
                for post in posts:
                    # 获取图片URL（优先使用高质量图片）
                    url = None
                    if "file_url" in post and post["file_url"]:
                        url = post["file_url"]
                    elif "sample_url" in post and post["sample_url"]:
                        url = post["sample_url"]
                    elif "preview_url" in post and post["preview_url"]:
                        url = post["preview_url"]

                    if url:
                        if self.download_image(url, role_name, existing_hashes):
                            downloaded_count += 1

                            if downloaded_count >= needed:
                                break

                # 延迟
                time.sleep(DELAY)

                # 下一页
                page += 1

            # 重置页数
            page = 0

        self.stats["roles_processed"] += 1
        logger.info(f"完成 {role_name}: 下载 {downloaded_count} 张图片")
        return downloaded_count

    def run(self):
        """运行采集任务"""
        logger.info("=" * 60)
        logger.info("开始从Gelbooru采集图片")
        logger.info("=" * 60)

        # 获取不足的角色
        insufficient_roles = self.get_insufficient_roles()

        if not insufficient_roles:
            logger.info("🎉 所有角色图片数都已达标！")
            return

        logger.info(f"发现 {len(insufficient_roles)} 个角色需要补充图片")

        # 显示角色列表
        for i, role in enumerate(insufficient_roles[:20], 1):
            logger.info(
                f"{i}. {role['name']}: 当前 {role['current']} 张, 需要 {role['needed']} 张"
            )

        # 采集每个角色
        for role_info in insufficient_roles:
            try:
                self.collect_role(role_info)
                time.sleep(3)  # 角色间延迟
            except Exception as e:
                logger.error(f"采集 {role_info['name']} 失败: {e}")
                continue

        # 输出统计
        logger.info("=" * 60)
        logger.info("采集任务完成")
        logger.info(f"处理角色数: {self.stats['roles_processed']}")
        logger.info(f"下载图片数: {self.stats['total_downloaded']}")
        logger.info(f"跳过图片数: {self.stats['total_skipped']}")
        logger.info(f"失败图片数: {self.stats['total_failed']}")
        logger.info("=" * 60)


def main():
    """主函数"""
    collector = GelbooruCollector()
    collector.run()


if __name__ == "__main__":
    main()
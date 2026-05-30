#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共下载工具模块
提取所有下载脚本中的重复功能，提供统一的下载接口
"""

import os
import sys
import hashlib
import time
import random
import logging
from urllib.parse import urlparse
from typing import Optional, Set, Tuple, Dict, Any

import requests


# 默认请求头
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    "Referer": "https://www.pixiv.net/",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 可用的User-Agent列表
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# 允许的图片扩展名
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")

# 需要过滤的域名
BLOCKED_DOMAINS = {"vv50.de", "sd.vv50.de"}


def setup_logger(name: str = "download_utils", log_file: str = None) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（如果指定了日志文件）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_random_user_agent() -> str:
    """
    获取随机User-Agent

    Returns:
        随机选择的User-Agent字符串
    """
    return random.choice(USER_AGENTS)


def is_valid_image_url(url: str) -> bool:
    """
    检查URL是否为有效的图片URL

    Args:
        url: 待检查的URL

    Returns:
        True如果是有效图片URL，否则False
    """
    if not url or not url.startswith("http"):
        return False

    parsed = urlparse(url)

    # 检查是否在黑名单域名中
    if parsed.netloc in BLOCKED_DOMAINS:
        return False

    # 检查扩展名
    lower_url = url.lower()
    if any(lower_url.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return True

    return False


def get_filename_from_url(url: str) -> str:
    """
    从URL提取文件名

    Args:
        url: 图片URL

    Returns:
        文件名
    """
    parsed = urlparse(url)
    path = parsed.path
    filename = os.path.basename(path)

    if filename and "." in filename:
        name, ext = os.path.splitext(filename)
        if ext.lower() in ALLOWED_EXTENSIONS:
            return filename

    # 如果无法从URL提取有效文件名，使用URL哈希
    return f"{hash_url(url)}.jpg"


def hash_url(url: str) -> str:
    """
    计算URL的哈希值

    Args:
        url: 待哈希的URL

    Returns:
        URL的MD5哈希值（十六进制）
    """
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def compute_file_hash(file_path: str) -> str:
    """
    计算文件的MD5哈希

    Args:
        file_path: 文件路径

    Returns:
        文件的MD5哈希值（十六进制）
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_image(
    url: str,
    save_dir: str,
    headers: dict = None,
    timeout: int = 30,
    max_retries: int = 3,
    proxy: Optional[str] = None,
    check_exists: bool = True,
) -> Tuple[bool, str]:
    """
    下载单张图片

    Args:
        url: 图片URL
        save_dir: 保存目录
        headers: 请求头（可选）
        timeout: 超时时间（秒）
        max_retries: 最大重试次数
        proxy: 代理地址（可选）
        check_exists: 是否检查文件已存在

    Returns:
        (成功标志, 消息/文件名)
    """
    if not is_valid_image_url(url):
        return False, "无效的图片URL"

    # 使用提供的headers或默认headers
    request_headers = headers.copy() if headers else DEFAULT_HEADERS.copy()
    request_headers["User-Agent"] = get_random_user_agent()

    # 配置代理
    proxies = None
    if proxy:
        proxies = {"http": proxy, "https": proxy}

    filename = get_filename_from_url(url)
    filepath = os.path.join(save_dir, filename)

    # 检查文件是否已存在
    if check_exists and os.path.exists(filepath):
        return False, "文件已存在"

    retries = 0
    backoff_factor = 1

    while retries < max_retries:
        try:
            response = requests.get(
                url,
                headers=request_headers,
                timeout=timeout * (1 + retries * 0.5),
                proxies=proxies,
                stream=True,
                allow_redirects=True,
            )

            if response.status_code == 200:
                # 检查内容类型
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    return False, f"不是图片类型: {content_type}"

                # 读取内容
                content = response.content

                # 检查文件大小
                if len(content) < 1000:
                    return False, "文件太小，可能是无效图片"

                # 确保目录存在
                os.makedirs(save_dir, exist_ok=True)

                # 保存图片
                with open(filepath, "wb") as f:
                    f.write(content)

                return True, filename

            elif response.status_code in [429, 503, 504]:
                # 服务器繁忙，增加延迟后重试
                retries += 1
                delay = backoff_factor * (2**retries) + random.uniform(0, 1)
                time.sleep(delay)
                continue

            else:
                return False, f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            retries += 1
            if retries >= max_retries:
                return False, "请求超时"
            delay = backoff_factor * (2**retries) + random.uniform(0, 1)
            time.sleep(delay)

        except requests.exceptions.ConnectionError:
            retries += 1
            if retries >= max_retries:
                return False, "连接错误"
            delay = backoff_factor * (2**retries) + random.uniform(0, 1)
            time.sleep(delay)

        except Exception as e:
            retries += 1
            if retries >= max_retries:
                return False, str(e)
            delay = backoff_factor * (2**retries) + random.uniform(0, 1)
            time.sleep(delay)

    return False, "重试次数已达上限"


def load_urls_from_file(file_path: str) -> list:
    """
    从文件加载URL列表

    Args:
        file_path: URL文件路径

    Returns:
        URL列表
    """
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    # 过滤无效URL
    return [url for url in urls if is_valid_image_url(url)]


def load_local_hashes(directory: str) -> Set[str]:
    """
    加载目录中所有图片的哈希值（用于去重）

    Args:
        directory: 图片目录

    Returns:
        图片哈希集合
    """
    hashes = set()

    if not os.path.exists(directory):
        return hashes

    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(ALLOWED_EXTENSIONS):
                file_path = os.path.join(dirpath, filename)
                try:
                    hashes.add(compute_file_hash(file_path))
                except Exception:
                    pass

    return hashes


class DownloadStats:
    """
    下载统计类
    """

    def __init__(self):
        self.downloaded = 0
        self.skipped = 0
        self.failed = 0

    def __repr__(self):
        return f"DownloadStats(downloaded={self.downloaded}, skipped={self.skipped}, failed={self.failed})"

    def to_dict(self) -> Dict[str, int]:
        return {"downloaded": self.downloaded, "skipped": self.skipped, "failed": self.failed}

    def update(self, downloaded: int = 0, skipped: int = 0, failed: int = 0):
        """更新统计数据"""
        self.downloaded += downloaded
        self.skipped += skipped
        self.failed += failed


class DownloadConfig:
    """
    下载配置类
    """

    def __init__(self, **kwargs):
        self.download_dir: str = kwargs.get("download_dir", "./downloaded_images")
        self.url_dir: str = kwargs.get("url_dir", "./url_files")
        self.max_workers: int = kwargs.get("max_workers", 5)
        self.timeout: int = kwargs.get("timeout", 30)
        self.max_retries: int = kwargs.get("max_retries", 3)
        self.delay: float = kwargs.get("delay", 0.1)
        self.proxy: Optional[str] = kwargs.get("proxy", None)
        self.min_file_size: int = kwargs.get("min_file_size", 1000)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items()}


__all__ = [
    "DEFAULT_HEADERS",
    "USER_AGENTS",
    "ALLOWED_EXTENSIONS",
    "BLOCKED_DOMAINS",
    "setup_logger",
    "get_random_user_agent",
    "is_valid_image_url",
    "get_filename_from_url",
    "hash_url",
    "compute_file_hash",
    "download_image",
    "load_urls_from_file",
    "load_local_hashes",
    "DownloadStats",
    "DownloadConfig",
]

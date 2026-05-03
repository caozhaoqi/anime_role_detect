#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接从文件系统下载最新URL
直接从 spider_image_system/data/img_url/ 目录读取URL文件进行下载
"""

import os
import sys
import json
import time
import hashlib
import logging
import requests
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Set, List

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
URL_DIR = PROJECT_ROOT / "spider_image_system" / "data" / "img_url"
OUTPUT_DIR = PROJECT_ROOT / "data" / "organized_images"
DB_PATH = PROJECT_ROOT / "data" / "role_images.db"
CONFIG_PATH = PROJECT_ROOT / "scripts" / "notification_config.json"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Referer': 'https://www.pixiv.net/'
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeishuNotifier:
    def __init__(self):
        self.app_id = None
        self.app_secret = None
        self.receive_id = None
        self.access_token = None
        self.token_expires = 0
        self._load_config()

    def _load_config(self):
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.app_id = config.get('feishu', {}).get('app_id')
                    self.app_secret = config.get('feishu', {}).get('app_secret')
                    self.receive_id = config.get('feishu', {}).get('receive_id')
                logger.info("飞书配置加载成功")
        except Exception as e:
            logger.warning(f"加载飞书配置失败: {e}")

    def _get_access_token(self) -> Optional[str]:
        if self.access_token and time.time() < self.token_expires:
            return self.access_token
        if not self.app_id or not self.app_secret:
            return None
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        try:
            response = requests.post(url, headers={"Content-Type": "application/json"},
                                     json={"app_id": self.app_id, "app_secret": self.app_secret}, timeout=10)
            result = response.json()
            if result.get("code") == 0:
                self.access_token = result.get("tenant_access_token")
                self.token_expires = time.time() + result.get("expire", 7200) - 300
                return self.access_token
        except Exception as e:
            logger.error(f"获取飞书 Access Token 失败: {e}")
        return None

    def send_message(self, text: str) -> bool:
        if not self.receive_id:
            return False
        access_token = self._get_access_token()
        if not access_token:
            return False
        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        params = {"receive_id_type": "chat_id"}
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        data = {"receive_id": self.receive_id, "msg_type": "text", "content": json.dumps({"text": text})}
        try:
            response = requests.post(url, headers=headers, params=params, json=data, timeout=10)
            return response.json().get("code") == 0
        except Exception as e:
            logger.error(f"发送飞书消息失败: {e}")
            return False


class Downloader:
    def __init__(self):
        self.notifier = FeishuNotifier()
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.lock = threading.Lock()
        self.running = True
        self.paused = False
        self.pause_event = threading.Event()

    def load_local_hashes(self) -> Set[str]:
        hashes = set()
        for role_dir in self.output_dir.iterdir():
            if role_dir.is_dir():
                for img in role_dir.glob("*"):
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                        try:
                            with open(img, 'rb') as f:
                                hashes.add(hashlib.md5(f.read()).hexdigest())
                        except:
                            pass
        logger.info(f"已加载 {len(hashes)} 个本地图片哈希")
        return hashes

    def get_all_urls_from_files(self) -> List[tuple]:
        all_urls = []
        for url_file in URL_DIR.glob("*_img.txt"):
            role_name = url_file.stem.replace('_img', '')
            with open(url_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            all_urls.extend([(role_name, url) for url in urls])
        return all_urls

    def download_image(self, role_name: str, url: str, local_hashes: Set[str]) -> bool:
        if not self.running:
            return False
        if self.paused:
            self.pause_event.wait()
        try:
            response = self.session.get(url, timeout=30, allow_redirects=True)
            if response.status_code != 200:
                return False
            content = response.content
            if len(content) < 1000:
                return False
            img_hash = hashlib.md5(content).hexdigest()
            if img_hash in local_hashes:
                with self.lock:
                    self.skipped_count += 1
                return False
            local_hashes.add(img_hash)
            role_dir = self.output_dir / role_name
            role_dir.mkdir(exist_ok=True)
            ext = Path(urlparse(url).path).suffix or '.jpg'
            if ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                ext = '.jpg'
            filename = f"{img_hash}{ext}"
            filepath = role_dir / filename
            with open(filepath, 'wb') as f:
                f.write(content)
            with self.lock:
                self.downloaded_count += 1
            return True
        except Exception as e:
            with self.lock:
                self.failed_count += 1
            return False

    def download_worker(self, role_name: str, urls: List[str], local_hashes: Set[str], progress_interval: int = 50):
        for i, url in enumerate(urls):
            if not self.running:
                break
            if self.download_image(role_name, url, local_hashes):
                if self.downloaded_count % progress_interval == 0:
                    logger.info(f"进度: 已下载 {self.downloaded_count}, 跳过 {self.skipped_count}, 失败 {self.failed_count}")
                    self.notifier.send_message(f"📥 下载进度: 已下载 {self.downloaded_count} 张")
            time.sleep(0.1)

    def run(self):
        logger.info("=" * 60)
        logger.info("🚀 从文件系统下载最新URL")
        logger.info("=" * 60)

        local_hashes = self.load_local_hashes()
        all_urls = self.get_all_urls_from_files()

        if not all_urls:
            logger.warning("未找到任何URL文件!")
            return

        role_urls = {}
        for role_name, url in all_urls:
            if role_name not in role_urls:
                role_urls[role_name] = []
            role_urls[role_name].append(url)

        total_urls = len(all_urls)
        total_roles = len(role_urls)
        logger.info(f"📊 总计: {total_urls} 个URL, {total_roles} 个角色")

        self.notifier.send_message(f"🚀 开始下载!\n📊 总计: {total_urls} 个URL, {total_roles} 个角色\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        threads = []
        for role_name, urls in role_urls.items():
            t = threading.Thread(target=self.download_worker, args=(role_name, urls, local_hashes))
            t.start()
            threads.append(t)
            time.sleep(0.05)

        for t in threads:
            t.join()

        logger.info("=" * 60)
        logger.info("✅ 下载完成!")
        logger.info(f"已下载: {self.downloaded_count}")
        logger.info(f"跳过(已存在): {self.skipped_count}")
        logger.info(f"失败: {self.failed_count}")
        logger.info("=" * 60)

        self.notifier.send_message(
            f"✅ 下载完成!\n"
            f"📥 已下载: {self.downloaded_count}\n"
            f"⏭️ 跳过: {self.skipped_count}\n"
            f"❌ 失败: {self.failed_count}\n"
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )


def main():
    downloader = Downloader()
    downloader.run()


if __name__ == '__main__':
    main()

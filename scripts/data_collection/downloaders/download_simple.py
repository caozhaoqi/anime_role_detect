#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""直接从文件系统下载最新URL - 带进度显示"""

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
from typing import Set, List
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
URL_DIR = PROJECT_ROOT / "spider_image_system" / "data" / "img_url"
OUTPUT_DIR = PROJECT_ROOT / "data" / "organized_images"
CONFIG_PATH = PROJECT_ROOT / "scripts" / "notification_config.json"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Referer': 'https://www.pixiv.net/'
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

notifier_enabled = True


def send_feishu_notification(text: str):
    """发送飞书通知"""
    global notifier_enabled
    if not notifier_enabled:
        return
    try:
        if not CONFIG_PATH.exists():
            return
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        app_id = config.get('feishu', {}).get('app_id')
        app_secret = config.get('feishu', {}).get('app_secret')
        receive_id = config.get('feishu', {}).get('receive_id')
        if not all([app_id, app_secret, receive_id]):
            return

        token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        token_resp = requests.post(token_url, json={"app_id": app_id, "app_secret": app_secret}, timeout=5)
        token_data = token_resp.json()
        if token_data.get("code") != 0:
            return
        access_token = token_data.get("tenant_access_token")

        msg_url = "https://open.feishu.cn/open-apis/im/v1/messages"
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        data = {"receive_id": receive_id, "msg_type": "text", "content": json.dumps({"text": text})}
        requests.post(msg_url, headers=headers, params={"receive_id_type": "chat_id"}, json=data, timeout=5)
    except Exception as e:
        logger.warning(f"飞书通知失败: {e}")


class Downloader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.lock = threading.Lock()
        self.running = True
        self.last_progress_time = time.time()
        self.progress_interval = 30

    def load_local_hashes(self) -> Set[str]:
        logger.info("正在加载本地图片哈希...")
        hashes = set()
        count = 0
        start = time.time()
        for role_dir in self.output_dir.iterdir():
            if role_dir.is_dir():
                for img in role_dir.glob("*"):
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                        try:
                            with open(img, 'rb') as f:
                                hashes.add(hashlib.md5(f.read()).hexdigest())
                            count += 1
                            if count % 1000 == 0:
                                logger.info(f"  已加载 {count} 个哈希...")
                        except:
                            pass
        logger.info(f"已加载 {len(hashes)} 个本地图片哈希, 耗时: {time.time()-start:.1f}秒")
        return hashes

    def get_all_urls_from_files(self) -> List[tuple]:
        logger.info("正在读取URL文件...")
        all_urls = []
        for url_file in URL_DIR.glob("*_img.txt"):
            role_name = url_file.stem.replace('_img', '')
            with open(url_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            all_urls.extend([(role_name, url) for url in urls])
        logger.info(f"已读取 {len(all_urls)} 个URL")
        return all_urls

    def download_image(self, role_name: str, url: str, local_hashes: Set[str]) -> bool:
        if not self.running:
            return False
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
            filepath = role_dir / f"{img_hash}{ext}"
            with open(filepath, 'wb') as f:
                f.write(content)
            with self.lock:
                self.downloaded_count += 1
            return True
        except Exception:
            with self.lock:
                self.failed_count += 1
            return False

    def download_worker(self, role_name: str, urls: List[str], local_hashes: Set[str]):
        for i, url in enumerate(urls):
            if not self.running:
                break
            self.download_image(role_name, url, local_hashes)
            if (i + 1) % 100 == 0:
                logger.info(f"  [{role_name}] 已处理 {i+1}/{len(urls)}")

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
        logger.info(f"📁 输出目录: {self.output_dir}")

        try:
            send_feishu_notification(f"🚀 开始下载!\n📊 总计: {total_urls} 个URL, {total_roles} 个角色")
        except:
            pass

        logger.info("开始下载线程...")
        threads = []
        for role_name, urls in role_urls.items():
            t = threading.Thread(target=self.download_worker, args=(role_name, urls, local_hashes))
            t.start()
            threads.append(t)
            time.sleep(0.02)

        last_log_time = time.time()
        while any(t.is_alive() for t in threads):
            time.sleep(2)
            if time.time() - last_log_time > 30:
                logger.info(f"进度: 已下载 {self.downloaded_count}, 跳过 {self.skipped_count}, 失败 {self.failed_count}")
                last_log_time = time.time()

        for t in threads:
            t.join()

        logger.info("=" * 60)
        logger.info("✅ 下载完成!")
        logger.info(f"已下载: {self.downloaded_count}")
        logger.info(f"跳过(已存在): {self.skipped_count}")
        logger.info(f"失败: {self.failed_count}")
        logger.info("=" * 60)

        try:
            send_feishu_notification(
                f"✅ 下载完成!\n"
                f"📥 已下载: {self.downloaded_count}\n"
                f"⏭️ 跳过: {self.skipped_count}\n"
                f"❌ 失败: {self.failed_count}"
            )
        except:
            pass


if __name__ == '__main__':
    Downloader().run()

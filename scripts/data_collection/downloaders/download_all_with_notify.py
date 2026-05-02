#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能图片下载器 - 带飞书通知
从 URL 文件下载图片，自动去重，发送飞书通知
"""

import os
import sys
import json
import time
import hashlib
import sqlite3
import logging
import requests
import threading
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
from typing import Optional, Dict, Set, List

# 配置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # 脚本在 scripts/data_collection/downloaders/ 目录
URL_DIR = PROJECT_ROOT / "spider_image_system" / "data" / "img_url"
OUTPUT_DIR = PROJECT_ROOT / "data" / "organized_images"
DB_PATH = PROJECT_ROOT / "data" / "role_images.db"
CONFIG_PATH = PROJECT_ROOT / "scripts" / "notification_config.json"

# 请求配置
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36',
    'Referer': 'https://www.pixiv.net/'
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeishuNotifier:
    """飞书通知器"""
    
    def __init__(self):
        self.app_id = None
        self.app_secret = None
        self.receive_id = None
        self.receive_id_type = "chat_id"
        self.access_token = None
        self.token_expires = 0
        self._load_config()
    
    def _load_config(self):
        """加载飞书配置"""
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.app_id = config.get('feishu', {}).get('app_id')
                    self.app_secret = config.get('feishu', {}).get('app_secret')
                    self.receive_id = config.get('feishu', {}).get('receive_id')
                    self.receive_id_type = config.get('feishu', {}).get('receive_id_type', 'chat_id')
                    logger.info("飞书配置加载成功")
        except Exception as e:
            logger.warning(f"加载飞书配置失败: {e}")
    
    def _get_access_token(self) -> Optional[str]:
        """获取飞书 Access Token"""
        if self.access_token and time.time() < self.token_expires:
            return self.access_token
        
        if not self.app_id or not self.app_secret:
            return None
        
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        headers = {"Content-Type": "application/json"}
        data = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            result = response.json()
            if result.get("code") == 0:
                self.access_token = result.get("tenant_access_token")
                self.token_expires = time.time() + result.get("expire", 7200) - 300
                return self.access_token
        except Exception as e:
            logger.error(f"获取飞书 Access Token 失败: {e}")
        return None
    
    def send_message(self, text: str) -> bool:
        """发送飞书消息"""
        if not self.receive_id:
            logger.warning("未配置飞书 receive_id")
            return False
        
        access_token = self._get_access_token()
        if not access_token:
            logger.warning("无法获取飞书 Access Token")
            return False
        
        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        params = {"receive_id_type": self.receive_id_type}
        data = {
            "receive_id": self.receive_id,
            "msg_type": "text",
            "content": json.dumps({"text": text})
        }
        try:
            response = requests.post(url, headers=headers, json=data, params=params, timeout=10)
            result = response.json()
            return result.get("code") == 0
        except Exception as e:
            logger.error(f"发送飞书消息失败: {e}")
            return False
    
    def send_download_start(self, total_urls: int, total_roles: int):
        """发送下载开始通知"""
        msg = f"""📥 **图片下载任务开始**

📊 任务统计:
• 总URL数: {total_urls:,}
• 角色数: {total_roles}
• 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⏳ 正在下载..."""
        return self.send_message(msg)
    
    def send_download_progress(self, downloaded: int, total: int, current_role: str, success_rate: float):
        """发送下载进度通知"""
        pct = (downloaded / total * 100) if total > 0 else 0
        msg = f"""📥 **下载进度更新**

📊 进度: {downloaded}/{total} ({pct:.1f}%)
✅ 成功率: {success_rate:.1f}%
🔄 当前角色: {current_role}

⏰ {datetime.now().strftime('%H:%M:%S')}"""
        return self.send_message(msg)
    
    def send_download_complete(self, stats: Dict):
        """发送下载完成通知"""
        msg = f"""✅ **图片下载任务完成**

📊 下载统计:
• 总URL数: {stats['total']:,}
• 成功: {stats['success']:,}
• 失败: {stats['failed']:,}
• 跳过(重复): {stats['skipped']:,}
• 成功率: {stats['success_rate']:.1f}%

📁 输出目录: {stats['output_dir']}
⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        return self.send_message(msg)


class SmartDownloader:
    """智能图片下载器"""
    
    def __init__(self, notify: bool = True):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.local_hashes: Set[str] = set()
        self.db_hashes: Set[str] = set()
        self.notifier = FeishuNotifier() if notify else None
        
        # 统计
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # 控制标志
        self.stop_flag = False
        self.pause_flag = False
        
        # 初始化
        self._init_db()
        self._load_local_hashes()
        self._load_db_hashes()
    
    def _init_db(self):
        """初始化数据库表"""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS downloaded_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                file_path TEXT,
                md5_hash TEXT UNIQUE NOT NULL,
                role_name TEXT,
                download_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'success'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_local_hashes(self):
        """加载本地已存在图片的哈希"""
        if OUTPUT_DIR.exists():
            for dirpath, dirnames, filenames in os.walk(OUTPUT_DIR):
                for filename in filenames:
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                        file_path = os.path.join(dirpath, filename)
                        try:
                            file_hash = self._compute_file_hash(file_path)
                            self.local_hashes.add(file_hash)
                        except Exception:
                            pass
        logger.info(f"已加载 {len(self.local_hashes)} 个本地图片哈希")
    
    def _load_db_hashes(self):
        """加载数据库中已记录的哈希"""
        if DB_PATH.exists():
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute('SELECT md5_hash FROM downloaded_images WHERE status = "success"')
            for row in cursor.fetchall():
                self.db_hashes.add(row[0])
            conn.close()
        logger.info(f"已加载 {len(self.db_hashes)} 个数据库哈希")
    
    def _compute_file_hash(self, file_path: str) -> str:
        """计算文件MD5哈希"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _get_extension(self, url: str) -> str:
        """从URL获取文件扩展名"""
        path = urlparse(url).path
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.jpg', '.jpeg', '.png', '.webp', '.gif'):
            return ext
        return '.jpg'
    
    def _download_image(self, url: str, role_name: str) -> bool:
        """下载单个图片"""
        try:
            response = self.session.get(url, timeout=30, stream=True)
            if response.status_code != 200:
                return False
            
            # 计算哈希
            image_data = response.content
            file_hash = hashlib.md5(image_data).hexdigest()
            
            # 检查重复
            if file_hash in self.local_hashes or file_hash in self.db_hashes:
                self.stats['skipped'] += 1
                return True
            
            # 保存图片
            role_dir = OUTPUT_DIR / role_name
            role_dir.mkdir(parents=True, exist_ok=True)
            
            ext = self._get_extension(url)
            filename = f"{file_hash}{ext}"
            file_path = role_dir / filename
            
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            # 记录到数据库
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO downloaded_images (url, file_path, md5_hash, role_name, status)
                    VALUES (?, ?, ?, ?, 'success')
                ''', (url, str(file_path), file_hash, role_name))
                conn.commit()
            except sqlite3.IntegrityError:
                pass
            finally:
                conn.close()
            
            self.local_hashes.add(file_hash)
            self.db_hashes.add(file_hash)
            self.stats['success'] += 1
            return True
            
        except Exception as e:
            logger.debug(f"下载失败 {url}: {e}")
            self.stats['failed'] += 1
            return False
    
    def get_all_url_files(self) -> List[tuple]:
        """获取所有URL（优先从数据库，其次从文件）"""
        files = []
        
        # 尝试从数据库读取
        try:
            import sqlite3
            db_path = PROJECT_ROOT / "data" / "role_images.db"
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute('SELECT url, role_name FROM raw_urls WHERE status = "pending"')
                rows = cursor.fetchall()
                conn.close()
                
                if rows:
                    # 按角色分组
                    role_urls = {}
                    for url, role_name in rows:
                        if role_name not in role_urls:
                            role_urls[role_name] = []
                        role_urls[role_name].append(url)
                    
                    for role_name, urls in role_urls.items():
                        files.append((role_name, urls))
                    
                    logger.info(f"从数据库加载了 {len(rows)} 个URL，共 {len(files)} 个角色")
                    return files
        except Exception as e:
            logger.warning(f"从数据库读取失败: {e}")
        
        # 从文件读取
        if URL_DIR.exists():
            for file in URL_DIR.glob("*_img.txt"):
                role_name = file.stem.replace("_img", "")
                with open(file, 'r', encoding='utf-8') as f:
                    urls = [line.strip() for line in f if line.strip()]
                if urls:
                    files.append((role_name, urls))
            logger.info(f"从文件加载了 {sum(len(u) for _, u in files)} 个URL，共 {len(files)} 个角色")
        
        return files
    
    def download_all(self, progress_interval: int = 100):
        """下载所有URL"""
        files = self.get_all_url_files()
        if not files:
            logger.warning("没有找到URL文件")
            return
        
        total_urls = sum(len(urls) for _, urls in files)
        total_roles = len(files)
        
        self.stats['total'] = total_urls
        
        # 发送开始通知
        if self.notifier:
            self.notifier.send_download_start(total_urls, total_roles)
        
        logger.info(f"开始下载 {total_urls} 个URL，共 {total_roles} 个角色")
        
        processed = 0
        for role_name, urls in files:
            if self.stop_flag:
                logger.warning("收到停止信号，终止下载")
                break
            
            while self.pause_flag:
                time.sleep(1)
            
            logger.info(f"下载角色: {role_name} ({len(urls)} 个URL)")
            
            for url in urls:
                if self.stop_flag:
                    break
                
                self._download_image(url, role_name)
                processed += 1
                
                # 进度通知
                if processed % progress_interval == 0 and self.notifier:
                    success_rate = (self.stats['success'] / processed * 100) if processed > 0 else 0
                    self.notifier.send_download_progress(
                        processed, total_urls, role_name, success_rate
                    )
        
        # 计算成功率
        success_rate = (self.stats['success'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
        self.stats['success_rate'] = success_rate
        self.stats['output_dir'] = str(OUTPUT_DIR)
        
        # 发送完成通知
        if self.notifier:
            self.notifier.send_download_complete(self.stats)
        
        logger.info(f"下载完成: {self.stats}")
    
    def stop(self):
        """停止下载"""
        self.stop_flag = True
    
    def pause(self):
        """暂停下载"""
        self.pause_flag = True
    
    def resume(self):
        """恢复下载"""
        self.pause_flag = False


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='智能图片下载器')
    parser.add_argument('--no-notify', action='store_true', help='禁用飞书通知')
    parser.add_argument('--progress-interval', type=int, default=100, help='进度通知间隔')
    args = parser.parse_args()
    
    downloader = SmartDownloader(notify=not args.no_notify)
    
    try:
        downloader.download_all(progress_interval=args.progress_interval)
    except KeyboardInterrupt:
        logger.info("用户中断下载")
        downloader.stop()


if __name__ == '__main__':
    main()

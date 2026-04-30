#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能图片下载器
从 URL 文件下载图片，自动去重（本地+数据库），记录下载状态
"""

import os
import sys
import hashlib
import sqlite3
import logging
import requests
from urllib.parse import urlparse

# 配置
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'
OUTPUT_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'
DB_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/role_images.db'

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


class SmartDownloader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.local_hashes = set()
        self.db_hashes = set()
        
        # 初始化数据库
        self._init_db()
        # 加载本地哈希
        self._load_local_hashes()
        # 加载数据库哈希
        self._load_db_hashes()
    
    def _init_db(self):
        """初始化数据库表"""
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 创建图片记录表
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
        if os.path.exists(OUTPUT_DIR):
            for dirpath, dirnames, filenames in os.walk(OUTPUT_DIR):
                for filename in filenames:
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        file_path = os.path.join(dirpath, filename)
                        try:
                            file_hash = self._compute_file_hash(file_path)
                            self.local_hashes.add(file_hash)
                        except Exception:
                            pass
        logger.info(f"已加载 {len(self.local_hashes)} 个本地图片哈希")
    
    def _load_db_hashes(self):
        """加载数据库中已记录的哈希"""
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('SELECT md5_hash FROM downloaded_images WHERE status = "success"')
            for row in cursor.fetchall():
                self.db_hashes.add(row[0])
            conn.close()
        logger.info(f"已加载 {len(self.db_hashes)} 个数据库哈希")
    
    def _compute_file_hash(self, file_path):
        """计算文件MD5哈希"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _compute_url_hash(self, url):
        """计算URL的MD5哈希（用于去重）"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _is_duplicate(self, url):
        """检查是否重复（URL或内容哈希）"""
        url_hash = self._compute_url_hash(url)
        if url_hash in self.db_hashes:
            return True, "URL已在数据库中"
        
        return False, ""
    
    def _download_image(self, url, role_dir):
        """下载单张图片"""
        try:
            response = self.session.get(url, timeout=15, stream=True)
            response.raise_for_status()
            
            # 获取文件名
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or '.' not in filename:
                filename = f"{self._compute_url_hash(url)}.jpg"
            
            # 确保有正确的扩展名
            name, ext = os.path.splitext(filename)
            if ext.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
                ext = '.jpg'
            filename = f"{name}{ext}"
            
            # 保存图片
            file_path = os.path.join(role_dir, filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 验证下载的图片
            if os.path.getsize(file_path) < 10:
                os.remove(file_path)
                return None, "文件太小"
            
            return file_path, None
            
        except Exception as e:
            return None, str(e)
    
    def _save_to_db(self, url, file_path, role_name, status='success'):
        """保存下载记录到数据库"""
        try:
            md5_hash = self._compute_file_hash(file_path) if status == 'success' else self._compute_url_hash(url)
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO downloaded_images 
                (url, file_path, md5_hash, role_name, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (url, file_path, md5_hash, role_name, status))
            
            conn.commit()
            conn.close()
            
            # 更新本地和数据库哈希集合
            if status == 'success':
                self.local_hashes.add(md5_hash)
                self.db_hashes.add(md5_hash)
            
            return True
        except Exception as e:
            logger.error(f"保存数据库失败 {url}: {str(e)}")
            return False
    
    def download_role_images(self, role_name, url_file_path):
        """下载单个角色的图片"""
        # 创建角色目录
        role_dir = os.path.join(OUTPUT_DIR, role_name)
        os.makedirs(role_dir, exist_ok=True)
        
        # 读取URL文件
        if not os.path.exists(url_file_path):
            logger.warning(f"URL文件不存在: {url_file_path}")
            return {'downloaded': 0, 'skipped': 0, 'failed': 0}
        
        with open(url_file_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        stats = {'downloaded': 0, 'skipped': 0, 'failed': 0}
        
        logger.info(f"开始下载角色 {role_name}，共 {len(urls)} 个URL")
        
        for url in urls:
            # 检查是否重复
            is_dup, reason = self._is_duplicate(url)
            if is_dup:
                stats['skipped'] += 1
                continue
            
            # 下载图片
            file_path, error = self._download_image(url, role_dir)
            
            if file_path:
                # 保存到数据库
                self._save_to_db(url, file_path, role_name)
                stats['downloaded'] += 1
                logger.debug(f"下载成功: {url}")
            else:
                # 记录失败
                self._save_to_db(url, None, role_name, status='failed')
                stats['failed'] += 1
                logger.warning(f"下载失败 {url}: {error}")
        
        logger.info(f"角色 {role_name} 下载完成: 成功 {stats['downloaded']}, 跳过 {stats['skipped']}, 失败 {stats['failed']}")
        return stats
    
    def download_all(self):
        """下载所有角色的图片"""
        if not os.path.exists(URL_DIR):
            logger.error(f"URL目录不存在: {URL_DIR}")
            return
        
        # 获取所有URL文件
        url_files = [f for f in os.listdir(URL_DIR) if f.endswith('_img.txt')]
        logger.info(f"找到 {len(url_files)} 个URL文件")
        
        total_stats = {'total_downloaded': 0, 'total_skipped': 0, 'total_failed': 0}
        
        for url_file in url_files:
            # 从文件名提取角色名
            role_name = url_file.replace('_img.txt', '')
            
            url_file_path = os.path.join(URL_DIR, url_file)
            
            # 下载该角色的图片
            stats = self.download_role_images(role_name, url_file_path)
            
            total_stats['total_downloaded'] += stats['downloaded']
            total_stats['total_skipped'] += stats['skipped']
            total_stats['total_failed'] += stats['failed']
        
        # 输出汇总统计
        logger.info("\n=== 全部下载完成 ===")
        logger.info(f"总下载: {total_stats['total_downloaded']} 张")
        logger.info(f"总跳过: {total_stats['total_skipped']} 张")
        logger.info(f"总失败: {total_stats['total_failed']} 张")
        
        # 保存下载统计
        stats_path = os.path.join(OUTPUT_DIR, 'download_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"下载时间: {__import__('datetime').datetime.now()}\n")
            f.write(f"总下载: {total_stats['total_downloaded']} 张\n")
            f.write(f"总跳过: {total_stats['total_skipped']} 张\n")
            f.write(f"总失败: {total_stats['total_failed']} 张\n")
        
        logger.info(f"下载统计已保存: {stats_path}")


def main():
    """主函数"""
    downloader = SmartDownloader()
    downloader.download_all()


if __name__ == '__main__':
    main()

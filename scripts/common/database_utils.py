#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共数据库工具模块
提供统一的数据库操作接口，支持SQLite
"""

import os
import sys
import sqlite3
import hashlib
import logging
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class ImageDatabase:
    """
    图片数据库操作类
    用于记录下载状态、去重等功能
    """
    
    def __init__(self, db_path: str):
        """
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
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
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_url ON downloaded_images(url)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_md5_hash ON downloaded_images(md5_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_role_name ON downloaded_images(role_name)')
        
        conn.commit()
        conn.close()
    
    def _compute_file_hash(self, file_path: str) -> str:
        """
        计算文件MD5哈希
        
        Args:
            file_path: 文件路径
        
        Returns:
            MD5哈希值
        """
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _compute_url_hash(self, url: str) -> str:
        """
        计算URL的MD5哈希
        
        Args:
            url: URL字符串
        
        Returns:
            MD5哈希值
        """
        return hashlib.md5(url.encode()).hexdigest()
    
    def get_existing_hashes(self, status: str = 'success') -> set:
        """
        获取已存在的图片哈希集合
        
        Args:
            status: 状态过滤，默认'success'
        
        Returns:
            哈希值集合
        """
        hashes = set()
        
        if not os.path.exists(self.db_path):
            return hashes
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT md5_hash FROM downloaded_images WHERE status = ?', (status,))
        
        for row in cursor.fetchall():
            hashes.add(row[0])
        
        conn.close()
        return hashes
    
    def is_url_downloaded(self, url: str) -> bool:
        """
        检查URL是否已下载
        
        Args:
            url: 图片URL
        
        Returns:
            True表示已下载，False表示未下载
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM downloaded_images WHERE url = ? AND status = "success"', (url,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result is not None
    
    def save_download_record(self, url: str, file_path: str, role_name: str, status: str = 'success') -> bool:
        """
        保存下载记录到数据库
        
        Args:
            url: 图片URL
            file_path: 保存路径
            role_name: 角色名称
            status: 状态 (success/failed)
        
        Returns:
            True表示成功，False表示失败
        """
        try:
            if status == 'success':
                md5_hash = self._compute_file_hash(file_path)
            else:
                md5_hash = self._compute_url_hash(url)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO downloaded_images 
                (url, file_path, md5_hash, role_name, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (url, file_path, md5_hash, role_name, status))
            
            conn.commit()
            conn.close()
            
            return True
        
        except Exception as e:
            logger.error(f"保存数据库失败 {url}: {str(e)}")
            return False
    
    def get_download_stats(self, role_name: Optional[str] = None) -> Dict[str, int]:
        """
        获取下载统计
        
        Args:
            role_name: 角色名称（可选）
        
        Returns:
            统计字典 {'success': int, 'failed': int}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if role_name:
            cursor.execute('''
                SELECT status, COUNT(*) FROM downloaded_images 
                WHERE role_name = ? GROUP BY status
            ''', (role_name,))
        else:
            cursor.execute('''
                SELECT status, COUNT(*) FROM downloaded_images 
                GROUP BY status
            ''')
        
        stats = {'success': 0, 'failed': 0}
        for status, count in cursor.fetchall():
            if status in stats:
                stats[status] = count
        
        conn.close()
        return stats
    
    def get_role_list(self) -> List[str]:
        """
        获取所有角色名称
        
        Returns:
            角色名称列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT role_name FROM downloaded_images ORDER BY role_name')
        
        roles = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return roles
    
    def get_downloaded_urls(self, role_name: Optional[str] = None) -> List[str]:
        """
        获取已下载的URL列表
        
        Args:
            role_name: 角色名称（可选）
        
        Returns:
            URL列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if role_name:
            cursor.execute('SELECT url FROM downloaded_images WHERE role_name = ? AND status = "success"', (role_name,))
        else:
            cursor.execute('SELECT url FROM downloaded_images WHERE status = "success"')
        
        urls = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return urls
    
    def delete_record(self, url: str) -> bool:
        """
        删除下载记录
        
        Args:
            url: 图片URL
        
        Returns:
            True表示成功，False表示失败
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM downloaded_images WHERE url = ?', (url,))
            conn.commit()
            conn.close()
            
            return True
        
        except Exception as e:
            logger.error(f"删除记录失败 {url}: {str(e)}")
            return False


__all__ = ['ImageDatabase']

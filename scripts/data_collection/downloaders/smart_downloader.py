#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能图片下载器（使用公共模块重构版）
从 URL 文件下载图片，自动去重（本地+数据库），记录下载状态
"""

import os
import sys
import glob

# 添加公共模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))

from download_utils import (
    setup_logger,
    download_image,
    load_urls_from_file,
    load_local_hashes,
    DownloadStats
)
from database_utils import ImageDatabase

# 配置
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'
OUTPUT_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'
DB_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/role_images.db'


class SmartDownloader:
    def __init__(self):
        self.logger = setup_logger("smart_downloader")
        self.db = ImageDatabase(DB_PATH)
        self.local_hashes = load_local_hashes(OUTPUT_DIR)
        self.db_hashes = self.db.get_existing_hashes()
        
        self.logger.info(f"已加载 {len(self.local_hashes)} 个本地图片哈希")
        self.logger.info(f"已加载 {len(self.db_hashes)} 个数据库哈希")
    
    def download_role_images(self, role_name: str, url_file_path: str) -> dict:
        """
        下载单个角色的图片
        
        Args:
            role_name: 角色名称
            url_file_path: URL文件路径
        
        Returns:
            统计字典
        """
        # 创建角色目录
        role_dir = os.path.join(OUTPUT_DIR, role_name)
        os.makedirs(role_dir, exist_ok=True)
        
        # 读取URL文件
        if not os.path.exists(url_file_path):
            self.logger.warning(f"URL文件不存在: {url_file_path}")
            return {'downloaded': 0, 'skipped': 0, 'failed': 0}
        
        urls = load_urls_from_file(url_file_path)
        stats = {'downloaded': 0, 'skipped': 0, 'failed': 0}
        
        self.logger.info(f"开始下载角色 {role_name}，共 {len(urls)} 个URL")
        
        for url in urls:
            # 检查是否已下载（通过数据库）
            if self.db.is_url_downloaded(url):
                stats['skipped'] += 1
                continue
            
            # 下载图片
            success, message = download_image(url, role_dir)
            
            if success:
                # 保存到数据库
                file_path = os.path.join(role_dir, message)
                self.db.save_download_record(url, file_path, role_name)
                stats['downloaded'] += 1
                self.logger.debug(f"下载成功: {url}")
            elif message == "文件已存在":
                # 文件已存在但数据库没有记录，补充记录
                file_path = os.path.join(role_dir, message)
                self.db.save_download_record(url, file_path, role_name)
                stats['skipped'] += 1
            else:
                # 记录失败
                self.db.save_download_record(url, None, role_name, status='failed')
                stats['failed'] += 1
                self.logger.warning(f"下载失败 {url}: {message}")
        
        self.logger.info(f"角色 {role_name} 下载完成: 成功 {stats['downloaded']}, 跳过 {stats['skipped']}, 失败 {stats['failed']}")
        return stats
    
    def download_all(self):
        """下载所有角色的图片"""
        if not os.path.exists(URL_DIR):
            self.logger.error(f"URL目录不存在: {URL_DIR}")
            return
        
        # 获取所有URL文件
        url_files = [f for f in os.listdir(URL_DIR) if f.endswith('_img.txt')]
        self.logger.info(f"找到 {len(url_files)} 个URL文件")
        
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
        self.logger.info("\n=== 全部下载完成 ===")
        self.logger.info(f"总下载: {total_stats['total_downloaded']} 张")
        self.logger.info(f"总跳过: {total_stats['total_skipped']} 张")
        self.logger.info(f"总失败: {total_stats['total_failed']} 张")
        
        # 保存下载统计
        stats_path = os.path.join(OUTPUT_DIR, 'download_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"下载时间: {__import__('datetime').datetime.now()}\n")
            f.write(f"总下载: {total_stats['total_downloaded']} 张\n")
            f.write(f"总跳过: {total_stats['total_skipped']} 张\n")
            f.write(f"总失败: {total_stats['total_failed']} 张\n")
        
        self.logger.info(f"下载统计已保存: {stats_path}")


def main():
    """主函数"""
    downloader = SmartDownloader()
    downloader.download_all()


if __name__ == '__main__':
    main()

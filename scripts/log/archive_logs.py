#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志归档脚本
定期将旧日志压缩归档，节省磁盘空间

用法:
    python archive_logs.py --log-dir logs --archive-days 7 --retention-days 180
"""

import os
import sys
import gzip
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('log_archiver')


class LogArchiver:
    """日志归档器"""
    
    def __init__(self, log_dir: str, archive_dir: str, retention_days: int = 180):
        """
        初始化归档器
        
        Args:
            log_dir: 日志根目录
            archive_dir: 归档目录
            retention_days: 归档保留天数
        """
        self.log_dir = Path(log_dir)
        self.archive_dir = Path(archive_dir)
        self.retention_days = retention_days
        
        # 确保归档目录存在
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
    def find_old_logs(self, days: int = 7) -> list[Path]:
        """
        查找指定天数前的日志文件
        
        Args:
            days: 天数阈值
            
        Returns:
            需要归档的日志文件列表
        """
        old_logs = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for log_file in self.log_dir.rglob('*.log'):
            # 跳过归档目录中的文件
            if self.archive_dir in log_file.parents:
                continue
                
            # 跳过备份目录
            if 'backup_' in str(log_file.parent):
                continue
            
            # 检查修改时间
            mod_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            if mod_time < cutoff_date:
                old_logs.append(log_file)
        
        return sorted(old_logs)
    
    def compress_log(self, log_file: Path) -> Path:
        """
        压缩单个日志文件
        
        Args:
            log_file: 要压缩的日志文件
            
        Returns:
            压缩后的文件路径
        """
        # 构建归档文件名: service_name_YYYYMMDD_HHMMSS.log.gz
        timestamp = datetime.fromtimestamp(log_file.stat().st_mtime).strftime('%Y%m%d_%H%M%S')
        relative_path = log_file.relative_to(self.log_dir)
        archive_name = f"{relative_path.with_suffix('').as_posix()}_{timestamp}.log.gz"
        archive_path = self.archive_dir / "compressed" / archive_name
        
        # 确保子目录存在
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(log_file, 'rb') as f_in:
                with gzip.open(archive_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            logger.info(f"✓ 压缩: {log_file.name} -> {archive_path.name}")
            return archive_path
        except Exception as e:
            logger.error(f"✗ 压缩失败 {log_file.name}: {e}")
            return None
    
    def delete_old_archives(self):
        """删除超过保留期的归档文件"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0
        
        for archive_file in self.archive_dir.rglob('*.gz'):
            mod_time = datetime.fromtimestamp(archive_file.stat().st_mtime)
            if mod_time < cutoff_date:
                try:
                    archive_file.unlink()
                    deleted_count += 1
                    logger.info(f"删除过期归档: {archive_file.name}")
                except Exception as e:
                    logger.error(f"删除失败 {archive_file.name}: {e}")
        
        if deleted_count > 0:
            logger.info(f"共删除 {deleted_count} 个过期归档文件")
        
        return deleted_count
    
    def cleanup_empty_dirs(self):
        """清理空目录"""
        for dirpath in sorted(self.log_dir.rglob('*'), reverse=True):
            if dirpath.is_dir() and not any(dirpath.iterdir()):
                try:
                    dirpath.rmdir()
                    logger.info(f"删除空目录: {dirpath}")
                except Exception as e:
                    logger.error(f"删除目录失败 {dirpath}: {e}")
    
    def get_disk_usage(self) -> dict:
        """获取磁盘使用情况"""
        total_size = 0
        log_count = 0
        
        for log_file in self.log_dir.rglob('*.log'):
            if self.archive_dir in log_file.parents:
                continue
            total_size += log_file.stat().st_size
            log_count += 1
        
        return {
            'total_size_mb': total_size / (1024 * 1024),
            'log_count': log_count
        }
    
    def run_archive(self, archive_days: int = 7):
        """
        执行归档操作
        
        Args:
            archive_days: 归档多少天前的日志
        """
        logger.info("=" * 60)
        logger.info("开始日志归档")
        logger.info("=" * 60)
        
        # 显示当前使用情况
        usage = self.get_disk_usage()
        logger.info(f"当前日志文件数: {usage['log_count']}")
        logger.info(f"当前日志总大小: {usage['total_size_mb']:.2f} MB")
        logger.info("")
        
        # 查找旧日志
        old_logs = self.find_old_logs(archive_days)
        
        if not old_logs:
            logger.info(f"没有{archive_days}天前的日志需要归档")
            return
        
        logger.info(f"找到 {len(old_logs)} 个需要归档的日志文件")
        logger.info("")
        
        # 压缩归档
        archived_count = 0
        for log_file in old_logs:
            archive_path = self.compress_log(log_file)
            if archive_path:
                # 删除原文件
                try:
                    log_file.unlink()
                    archived_count += 1
                except Exception as e:
                    logger.error(f"删除原文件失败 {log_file.name}: {e}")
        
        logger.info("")
        logger.info(f"成功归档 {archived_count}/{len(old_logs)} 个文件")
        
        # 删除过期归档
        logger.info("")
        deleted = self.delete_old_archives()
        
        # 清理空目录
        self.cleanup_empty_dirs()
        
        # 显示新的使用情况
        new_usage = self.get_disk_usage()
        logger.info("")
        logger.info("=" * 60)
        logger.info("归档完成")
        logger.info(f"剩余日志文件数: {new_usage['log_count']}")
        logger.info(f"剩余日志总大小: {new_usage['total_size_mb']:.2f} MB")
        logger.info(f"释放空间: {usage['total_size_mb'] - new_usage['total_size_mb']:.2f} MB")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='日志归档工具')
    parser.add_argument('--log-dir', default='logs', help='日志根目录 (默认: logs)')
    parser.add_argument('--archive-dir', default=None, help='归档目录 (默认: logs/archive)')
    parser.add_argument('--archive-days', type=int, default=7, help='归档多少天前的日志 (默认: 7)')
    parser.add_argument('--retention-days', type=int, default=180, help='归档保留天数 (默认: 180)')
    parser.add_argument('--dry-run', action='store_true', help='仅显示将要归档的文件，不实际执行')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        logger.error(f"日志目录不存在: {log_dir}")
        sys.exit(1)
    
    archive_dir = Path(args.archive_dir) if args.archive_dir else log_dir / 'archive'
    
    archiver = LogArchiver(
        log_dir=str(log_dir),
        archive_dir=str(archive_dir),
        retention_days=args.retention_days
    )
    
    if args.dry_run:
        logger.info("Dry run模式 - 显示将要归档的文件:")
        old_logs = archiver.find_old_logs(args.archive_days)
        if old_logs:
            for log_file in old_logs:
                size_mb = log_file.stat().st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(log_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"  {log_file} ({size_mb:.2f} MB, 修改时间: {mod_time})")
        else:
            logger.info(f"  没有{args.archive_days}天前的日志需要归档")
    else:
        archiver.run_archive(archive_days=args.archive_days)


if __name__ == '__main__':
    main()

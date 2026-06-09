#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志分析工具
提供多种日志查询和分析功能

用法:
    python analyze_logs.py --service api-service --level ERROR --since 1h
    python analyze_logs.py --search "timeout" --services api-service,model-service
    python analyze_logs.py --stats
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import argparse


class LogAnalyzer:
    """日志分析器"""
    
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = Path(log_dir)
        
        # 服务到日志目录的映射
        self.service_map = {
            'api-service': 'services/api-service',
            'model-service': 'services/model-service',
            'api-gateway': 'services/api-gateway',
            'multimedia-service': 'services/multimedia-service',
            'search-service': 'services/search-service',
            'inference-worker': 'services/inference-worker',
            'frontend': 'services/frontend',
            'monitoring': 'services/monitoring',
        }
    
    def find_log_files(self, service: str = None, include_error: bool = True) -> list[Path]:
        """
        查找日志文件
        
        Args:
            service: 指定服务名称，None表示所有服务
            include_error: 是否包含错误日志
            
        Returns:
            日志文件列表
        """
        log_files = []
        
        if service:
            if service in self.service_map:
                service_dir = self.log_dir / self.service_map[service]
                if service_dir.exists():
                    for log_file in service_dir.glob('*.log'):
                        if not include_error and '.err.' in str(log_file):
                            continue
                        log_files.append(log_file)
            else:
                print(f"警告: 未知服务 '{service}'")
        else:
            # 所有服务的日志
            for service_dir in (self.log_dir / 'services').rglob('*'):
                if service_dir.is_dir():
                    for log_file in service_dir.glob('*.log'):
                        if not include_error and '.err.' in str(log_file):
                            continue
                        log_files.append(log_file)
        
        return sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def parse_log_line(self, line: str) -> dict:
        """
        解析日志行
        
        支持的格式:
        - 2024-06-09 10:30:45 | INFO | message
        - 2024-06-09 10:30:45.123 [INFO] message
        - [2024-06-09 10:30:45] INFO: message
        
        Returns:
            包含timestamp, level, message的字典
        """
        patterns = [
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\|\s*(\w+)\s*\|\s*(.*)',
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\[(\w+)\]\s*(.*)',
            r'\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]\s*(\w+):\s*(.*)',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                timestamp_str, level, message = match.groups()
                try:
                    timestamp = datetime.strptime(timestamp_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    timestamp = None
                
                return {
                    'timestamp': timestamp,
                    'level': level.upper(),
                    'message': message.strip()
                }
        
        return None
    
    def filter_logs(self, log_files: list[Path], 
                   level: str = None, 
                   keyword: str = None,
                   since: str = None,
                   until: str = None) -> list[tuple[str, dict]]:
        """
        过滤日志
        
        Args:
            log_files: 日志文件列表
            level: 日志级别过滤 (ERROR, WARNING, INFO等)
            keyword: 关键词搜索
            since: 起始时间 (如: 1h, 30m, 2d)
            until: 结束时间
            
        Returns:
            (文件名, 日志条目) 列表
        """
        results = []
        
        # 解析时间范围
        cutoff_time = None
        if since:
            cutoff_time = self.parse_relative_time(since)
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        parsed = self.parse_log_line(line)
                        
                        if not parsed:
                            continue
                        
                        # 级别过滤
                        if level and parsed['level'] != level.upper():
                            continue
                        
                        # 关键词过滤
                        if keyword and keyword.lower() not in parsed['message'].lower():
                            continue
                        
                        # 时间过滤
                        if cutoff_time and parsed['timestamp']:
                            if parsed['timestamp'] < cutoff_time:
                                continue
                        
                        results.append((str(log_file), parsed))
            except Exception as e:
                print(f"读取文件失败 {log_file}: {e}")
        
        return results
    
    def parse_relative_time(self, time_str: str) -> datetime:
        """
        解析相对时间字符串
        
        Args:
            time_str: 如 "1h", "30m", "2d", "7d"
            
        Returns:
            datetime对象
        """
        match = re.match(r'(\d+)([hmd])', time_str)
        if not match:
            raise ValueError(f"无效的时间格式: {time_str}")
        
        value = int(match.group(1))
        unit = match.group(2)
        
        now = datetime.now()
        if unit == 'h':
            return now - timedelta(hours=value)
        elif unit == 'm':
            return now - timedelta(minutes=value)
        elif unit == 'd':
            return now - timedelta(days=value)
        
        return now
    
    def show_stats(self, service: str = None):
        """显示日志统计信息"""
        log_files = self.find_log_files(service)
        
        if not log_files:
            print("未找到日志文件")
            return
        
        print("=" * 80)
        print("日志统计信息")
        print("=" * 80)
        print()
        
        total_lines = 0
        level_counts = Counter()
        error_messages = []
        
        for log_file in log_files:
            file_lines = 0
            file_levels = Counter()
            
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parsed = self.parse_log_line(line)
                        if parsed:
                            file_lines += 1
                            file_levels[parsed['level']] += 1
                            
                            if parsed['level'] == 'ERROR':
                                error_messages.append({
                                    'file': str(log_file),
                                    'message': parsed['message'],
                                    'timestamp': parsed['timestamp']
                                })
                
                total_lines += file_lines
                level_counts.update(file_levels)
                
                rel_path = log_file.relative_to(self.log_dir)
                size_mb = log_file.stat().st_size / (1024 * 1024)
                print(f"📄 {rel_path}")
                print(f"   行数: {file_lines:,}, 大小: {size_mb:.2f} MB")
                if file_levels:
                    print(f"   级别分布: {dict(file_levels)}")
                print()
                
            except Exception as e:
                print(f"❌ 读取失败 {log_file}: {e}")
                print()
        
        print("-" * 80)
        print(f"总计:")
        print(f"  文件数: {len(log_files)}")
        print(f"  总行数: {total_lines:,}")
        print(f"  级别分布: {dict(level_counts)}")
        print()
        
        if error_messages:
            print(f"最近10个错误:")
            for err in error_messages[-10:]:
                ts = err['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if err['timestamp'] else 'N/A'
                print(f"  [{ts}] {err['message'][:100]}")
            print()
    
    def search_logs(self, keyword: str, services: list[str] = None, 
                   level: str = None, limit: int = 50):
        """搜索日志"""
        print("=" * 80)
        print(f"搜索关键词: '{keyword}'")
        if services:
            print(f"服务: {', '.join(services)}")
        if level:
            print(f"级别: {level}")
        print("=" * 80)
        print()
        
        log_files = []
        if services:
            for service in services:
                log_files.extend(self.find_log_files(service))
        else:
            log_files = self.find_log_files()
        
        results = self.filter_logs(log_files, level=level, keyword=keyword)
        
        if not results:
            print("未找到匹配的日志")
            return
        
        print(f"找到 {len(results)} 条匹配记录\n")
        
        for file_path, parsed in results[:limit]:
            rel_path = Path(file_path).relative_to(self.log_dir)
            ts = parsed['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if parsed['timestamp'] else 'N/A'
            print(f"[{ts}] [{parsed['level']}] [{rel_path.name}]")
            print(f"  {parsed['message']}")
            print()
        
        if len(results) > limit:
            print(f"... 还有 {len(results) - limit} 条记录未显示")


def main():
    parser = argparse.ArgumentParser(description='日志分析工具')
    
    # 模式选择
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--stats', action='store_true', help='显示统计信息')
    mode_group.add_argument('--search', type=str, help='搜索关键词')
    
    # 通用参数
    parser.add_argument('--service', type=str, help='指定服务名称')
    parser.add_argument('--services', type=str, help='多个服务名称，逗号分隔')
    parser.add_argument('--level', type=str, help='日志级别 (ERROR, WARNING, INFO)')
    parser.add_argument('--since', type=str, help='起始时间 (如: 1h, 30m, 2d)')
    parser.add_argument('--limit', type=int, default=50, help='显示条数限制')
    parser.add_argument('--log-dir', default='logs', help='日志目录')
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(log_dir=args.log_dir)
    
    if args.stats:
        analyzer.show_stats(service=args.service)
    
    elif args.search:
        services = args.services.split(',') if args.services else None
        analyzer.search_logs(
            keyword=args.search,
            services=services,
            level=args.level,
            limit=args.limit
        )


if __name__ == '__main__':
    main()

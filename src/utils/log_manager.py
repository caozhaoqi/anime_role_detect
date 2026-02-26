#!/usr/bin/env python3
"""
日志管理器类，实现更详细的日志记录和分析功能
"""
import os
import logging
import json
import time
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class LogManager:
    """日志管理器类"""
    
    def __init__(self, log_dir='logs', log_level=logging.INFO, max_bytes=10*1024*1024, backup_count=5):
        """
        初始化日志管理器
        
        Args:
            log_dir: 日志目录
            log_level: 日志级别
            max_bytes: 单个日志文件最大大小
            backup_count: 日志文件备份数量
        """
        self.log_dir = log_dir
        self.log_level = log_level
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.loggers = {}
        self.log_stats = {
            'info_count': 0,
            'warning_count': 0,
            'error_count': 0,
            'critical_count': 0,
            'total_count': 0
        }
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 配置根日志
        self._configure_root_logger()
    
    def _configure_root_logger(self):
        """配置根日志"""
        # 创建根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # 创建文件处理器
        log_file = os.path.join(self.log_dir, f'application_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # 添加处理器
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        logger.info(f"根日志器配置完成，日志文件: {log_file}")
    
    def get_logger(self, name):
        """
        获取指定名称的日志器
        
        Args:
            name: 日志器名称
            
        Returns:
            logging.Logger: 日志器实例
        """
        if name not in self.loggers:
            # 创建新的日志器
            logger_instance = logging.getLogger(name)
            logger_instance.setLevel(self.log_level)
            
            # 创建文件处理器
            log_file = os.path.join(self.log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            
            # 添加处理器
            logger_instance.addHandler(file_handler)
            self.loggers[name] = logger_instance
            
            logger.info(f"创建日志器: {name}，日志文件: {log_file}")
        
        return self.loggers[name]
    
    def log_with_context(self, logger_name, level, message, context=None):
        """
        带上下文的日志记录
        
        Args:
            logger_name: 日志器名称
            level: 日志级别
            message: 日志消息
            context: 上下文信息
        """
        logger_instance = self.get_logger(logger_name)
        
        # 构建日志消息
        log_message = message
        if context:
            log_message = f"{message} (Context: {json.dumps(context, ensure_ascii=False)})"
        
        # 记录日志
        if level == logging.INFO:
            logger_instance.info(log_message)
            self.log_stats['info_count'] += 1
        elif level == logging.WARNING:
            logger_instance.warning(log_message)
            self.log_stats['warning_count'] += 1
        elif level == logging.ERROR:
            logger_instance.error(log_message)
            self.log_stats['error_count'] += 1
        elif level == logging.CRITICAL:
            logger_instance.critical(log_message)
            self.log_stats['critical_count'] += 1
        
        self.log_stats['total_count'] += 1
    
    def log_collection_stats(self, collector_name, stats):
        """
        记录采集统计信息
        
        Args:
            collector_name: 采集器名称
            stats: 统计信息
        """
        # 创建统计日志文件
        stats_file = os.path.join(self.log_dir, f'collection_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        # 构建统计数据
        collection_stats = {
            'timestamp': datetime.now().isoformat(),
            'collector': collector_name,
            'stats': stats,
            'log_stats': self.log_stats.copy()
        }
        
        # 保存统计数据
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(collection_stats, f, ensure_ascii=False, indent=2)
            logger.info(f"采集统计信息保存成功: {stats_file}")
        except Exception as e:
            logger.error(f"保存采集统计信息失败: {e}")
    
    def analyze_logs(self, log_file=None, days=7):
        """
        分析日志文件
        
        Args:
            log_file: 日志文件路径
            days: 分析最近几天的日志
            
        Returns:
            dict: 日志分析结果
        """
        analysis_result = {
            'total_logs': 0,
            'level_distribution': {
                'INFO': 0,
                'WARNING': 0,
                'ERROR': 0,
                'CRITICAL': 0
            },
            'top_errors': [],
            'time_distribution': {},
            'duration': 0
        }
        
        start_time = time.time()
        
        # 收集日志文件
        log_files = []
        if log_file:
            if os.path.exists(log_file):
                log_files.append(log_file)
        else:
            # 收集最近几天的日志文件
            for file in os.listdir(self.log_dir):
                if file.endswith('.log'):
                    file_path = os.path.join(self.log_dir, file)
                    file_mtime = os.path.getmtime(file_path)
                    if time.time() - file_mtime <= days * 24 * 3600:
                        log_files.append(file_path)
        
        # 分析日志文件
        error_patterns = {}
        
        for file_path in log_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        analysis_result['total_logs'] += 1
                        
                        # 分析日志级别
                        if 'INFO' in line:
                            analysis_result['level_distribution']['INFO'] += 1
                        elif 'WARNING' in line:
                            analysis_result['level_distribution']['WARNING'] += 1
                        elif 'ERROR' in line:
                            analysis_result['level_distribution']['ERROR'] += 1
                            # 提取错误信息
                            error_msg = line.split('ERROR - ')[1] if 'ERROR - ' in line else line
                            error_patterns[error_msg] = error_patterns.get(error_msg, 0) + 1
                        elif 'CRITICAL' in line:
                            analysis_result['level_distribution']['CRITICAL'] += 1
                        
                        # 分析时间分布
                        if ' - ' in line:
                            time_str = line.split(' - ')[0]
                            hour = time_str.split(' ')[1].split(':')[0]
                            analysis_result['time_distribution'][hour] = analysis_result['time_distribution'].get(hour, 0) + 1
            except Exception as e:
                logger.error(f"分析日志文件失败 {file_path}: {e}")
        
        # 提取Top错误
        sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        for error_msg, count in sorted_errors:
            analysis_result['top_errors'].append({
                'message': error_msg.strip(),
                'count': count
            })
        
        analysis_result['duration'] = time.time() - start_time
        
        # 保存分析结果
        analysis_file = os.path.join(self.log_dir, f'log_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        try:
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2)
            logger.info(f"日志分析结果保存成功: {analysis_file}")
        except Exception as e:
            logger.error(f"保存日志分析结果失败: {e}")
        
        return analysis_result
    
    def get_log_stats(self):
        """
        获取日志统计信息
        
        Returns:
            dict: 日志统计信息
        """
        return self.log_stats
    
    def reset_log_stats(self):
        """
        重置日志统计信息
        """
        self.log_stats = {
            'info_count': 0,
            'warning_count': 0,
            'error_count': 0,
            'critical_count': 0,
            'total_count': 0
        }
        logger.info("日志统计信息已重置")
    
    def close(self):
        """
        关闭所有日志处理器
        """
        for logger_name, logger_instance in self.loggers.items():
            for handler in logger_instance.handlers[:]:
                handler.close()
                logger_instance.removeHandler(handler)
        
        # 关闭根日志处理器
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        
        logger.info("所有日志处理器已关闭")


# 全局日志管理器实例
log_manager = LogManager()

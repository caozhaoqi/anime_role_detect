#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式数据采集命令行工具

使用分布式任务管理器进行大规模数据采集
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.distributed_manager import DistributedManager
from src.data_collection.keyword_based_collector import KeywordBasedDataCollector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('distributed_collector')


def load_keywords(keyword_file):
    """
    从关键词文件加载角色列表
    
    Args:
        keyword_file: 关键词文件路径
        
    Returns:
        角色列表
    """
    keywords = []
    
    try:
        with open(keyword_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    keywords.append(line)
        
        logger.info(f"从 {keyword_file} 加载了 {len(keywords)} 个关键词")
    except Exception as e:
        logger.error(f"加载关键词文件失败 {keyword_file}: {e}")
    
    return keywords


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='分布式数据采集命令行工具')
    
    parser.add_argument('--action', type=str, required=True,
                       choices=['start', 'add_task', 'monitor', 'stats'],
                       help='操作类型')
    parser.add_argument('--manager-id', type=str, default='manager_001',
                       help='管理器ID')
    parser.add_argument('--queue-path', type=str, default='./task_queue',
                       help='任务队列存储路径')
    parser.add_argument('--max-workers', type=int, default=5,
                       help='最大工作线程数')
    
    # 添加任务相关参数
    parser.add_argument('--series', type=str,
                       help='系列名称')
    parser.add_argument('--keyword-file', type=str,
                       help='关键词文件路径')
    parser.add_argument('--characters', type=str, nargs='+',
                       help='角色列表')
    parser.add_argument('--max-images', type=int, default=50,
                       help='每个角色的最大图像数')
    parser.add_argument('--output-dir', type=str, default='data/train',
                       help='输出目录')
    
    # 监控相关参数
    parser.add_argument('--interval', type=int, default=10,
                       help='监控间隔（秒）')
    
    args = parser.parse_args()
    
    # 创建分布式管理器
    manager = DistributedManager(
        manager_id=args.manager_id,
        queue_path=args.queue_path,
        max_workers=args.max_workers
    )
    
    if args.action == 'start':
        # 启动分布式管理器
        logger.info(f"启动分布式管理器，ID: {args.manager_id}")
        manager.start()
        
        # 进入监控模式
        try:
            manager.monitor(interval=args.interval)
        except KeyboardInterrupt:
            logger.info("监控已停止")
        finally:
            manager.stop()
    
    elif args.action == 'add_task':
        # 添加任务
        if not args.series:
            logger.error("添加任务时必须指定 --series 参数")
            return
        
        characters = []
        
        # 从关键词文件加载角色
        if args.keyword_file:
            characters.extend(load_keywords(args.keyword_file))
        
        # 从命令行参数加载角色
        if args.characters:
            characters.extend(args.characters)
        
        if not characters:
            logger.error("没有指定角色，请使用 --keyword-file 或 --characters 参数")
            return
        
        # 添加图像采集任务
        task_ids = manager.add_image_collection_tasks(
            series=args.series,
            characters=characters,
            max_images=args.max_images
        )
        
        logger.info(f"成功添加了 {len(task_ids)} 个任务")
        logger.info(f"任务ID列表: {task_ids}")
    
    elif args.action == 'monitor':
        # 监控任务执行状态
        logger.info("开始监控任务执行状态")
        try:
            manager.monitor(interval=args.interval)
        except KeyboardInterrupt:
            logger.info("监控已停止")
    
    elif args.action == 'stats':
        # 获取统计信息
        stats = manager.get_stats()
        logger.info("分布式管理器统计信息:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

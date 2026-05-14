#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定时训练脚本
定期运行自动化训练流程
"""

import schedule
import time
import os
import sys
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduled_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("scheduled_training")

# 脚本路径
AUTOMATED_TRAINING_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'automated_training.py')


def run_training():
    """运行训练流程"""
    logger.info("=" * 60)
    logger.info(f"开始定时训练任务: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    try:
        # 运行自动化训练脚本
        result = os.system(f'python3 {AUTOMATED_TRAINING_SCRIPT}')
        
        if result == 0:
            logger.info("定时训练任务执行成功")
        else:
            logger.error(f"定时训练任务执行失败，返回码: {result}")
            
    except Exception as e:
        logger.error(f"执行训练任务时出错: {e}")
    
    logger.info("=" * 60)
    logger.info(f"定时训练任务完成: {datetime.now().isoformat()}")
    logger.info("=" * 60)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("定时训练系统启动")
    logger.info("=" * 60)
    
    # 设置定时任务
    # 每天凌晨2点运行训练
    schedule.every().day.at("02:00").do(run_training)
    
    # 每小时检查一次（用于测试）
    # schedule.every().hour.do(run_training)
    
    # 立即运行一次（用于测试）
    # run_training()
    
    logger.info("定时任务已设置")
    logger.info("每天凌晨2点自动运行训练流程")
    logger.info("按 Ctrl+C 退出")
    
    # 运行调度器
    while True:
        schedule.run_pending()
        time.sleep(60)  # 每分钟检查一次

if __name__ == '__main__':
    main()

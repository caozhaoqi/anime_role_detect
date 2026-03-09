#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试全局日志系统
"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.logging.global_logger import get_logger

# 测试不同模块的日志
logger1 = get_logger('module1')
logger2 = get_logger('module2')
logger3 = get_logger('module3')

print("开始测试全局日志系统...")
print("\n1. 测试信息日志")
logger1.info("这是模块1的信息日志")
logger2.info("这是模块2的信息日志")

print("\n2. 测试警告日志")
logger1.warning("这是模块1的警告日志")
logger2.warning("这是模块2的警告日志")

print("\n3. 测试错误日志")
try:
    1 / 0
except Exception as e:
    logger3.error(f"这是模块3的错误日志: {e}")

print("\n4. 测试异常日志")
try:
    raise ValueError("测试异常")
except Exception as e:
    logger1.exception(f"这是模块1的异常日志: {e}")

print("\n5. 测试调试日志")
logger2.debug("这是模块2的调试日志")

print("\n日志测试完成！请检查 logs 目录下的日志文件。")

# 检查日志目录是否创建
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
if os.path.exists(log_dir):
    print(f"\n日志目录已创建: {log_dir}")
    # 列出日志目录下的文件
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            print(f"  - {os.path.join(root, file)}")
else:
    print(f"\n警告: 日志目录未创建: {log_dir}")
#!/usr/bin/env python3
"""
测试日志系统
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.logging.global_logger import get_logger, log_system, log_inference, log_training, log_error

# 测试日志系统
def test_logging():
    print("=== 测试日志系统 ===")
    
    # 获取日志记录器
    logger = get_logger("test")
    
    # 测试不同级别的日志
    logger.debug("这是一条调试日志")
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    logger.critical("这是一条严重错误日志")
    
    # 测试分类日志
    log_system("系统启动成功")
    log_inference("推理完成: 角色=测试角色, 相似度=0.95")
    log_training("训练完成: 准确率=0.98")
    log_error("发生错误: 无法加载模型")
    
    print("日志测试完成，请检查 logs 目录下的日志文件")

if __name__ == "__main__":
    test_logging()

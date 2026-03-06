#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过API接口下载采集到的图片
"""

import requests
import time
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('download_images')

# API地址
API_URL = "http://localhost:33333/api/v1.2.5.260305/sis/download_all_image/start/"

def download_images():
    """调用API下载图片"""
    try:
        logger.info("开始下载图片...")
        # 发送API请求
        response = requests.get(API_URL)
        
        # 检查响应
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                logger.info("图片下载任务已启动")
                return True
            else:
                logger.error(f"图片下载任务启动失败: {result.get('msg')}")
                return False
        else:
            logger.error(f"图片下载任务启动失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"图片下载任务启动出错: {e}")
        return False

def main():
    """主函数"""
    # 启动下载任务
    if download_images():
        logger.info("图片下载任务已成功启动，请等待下载完成")
        # 等待一段时间，让下载任务开始
        time.sleep(5)
        logger.info("下载任务正在进行中...")
    else:
        logger.error("图片下载任务启动失败")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量通过API接口采集角色图片
"""

import requests
import time
import logging
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('batch_spider_roles')

# API地址
API_URL = "http://localhost:33333/api/v1.2.5.260305/sis/spider_start/single?key_word="

# 角色列表文件路径
ROLES_FILE = "auto_spider_img/blda_spider_img_keyword.txt"

def load_roles():
    """加载角色列表"""
    roles = []
    try:
        with open(ROLES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                role = line.strip()
                if role:
                    roles.append(role)
        logger.info(f"成功加载 {len(roles)} 个角色")
        return roles
    except Exception as e:
        logger.error(f"加载角色列表失败: {e}")
        return []

def collect_role_images(role_name):
    """采集单个角色的图片"""
    try:
        logger.info(f"开始采集角色: {role_name}")
        # 发送API请求
        response = requests.post(f"{API_URL}{role_name}")
        
        # 检查响应
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                logger.success(f"角色 {role_name} 采集成功")
                # 添加延迟，避免请求过快
                time.sleep(10)  # 增加等待时间
                return True
            else:
                logger.error(f"角色 {role_name} 采集失败: {result.get('msg')}")
                # 添加延迟，避免请求过快
                time.sleep(5)  # 增加等待时间
                return False
        else:
            logger.error(f"角色 {role_name} 采集失败，状态码: {response.status_code}")
            # 添加延迟，避免请求过快
            time.sleep(5)  # 增加等待时间
            return False
    except Exception as e:
        logger.error(f"角色 {role_name} 采集出错: {e}")
        # 添加延迟，避免请求过快
        time.sleep(5)  # 增加等待时间
        return False

def main():
    """主函数"""
    # 加载角色列表
    all_roles = load_roles()
    
    if not all_roles:
        logger.error("没有找到角色列表")
        return
    
    # 只采集日奈、亚子、伊织这三个角色，以便测试
    test_roles = ["日奈", "亚子", "伊织"]
    logger.info(f"测试采集 {len(test_roles)} 个角色: {test_roles}")
    
    # 开始采集
    success_count = 0
    fail_count = 0
    
    for role in test_roles:
        if collect_role_images(role):
            success_count += 1
        else:
            fail_count += 1
    
    # 打印统计信息
    logger.info(f"采集完成: 成功 {success_count}, 失败 {fail_count}")
    
if __name__ == "__main__":
    main()

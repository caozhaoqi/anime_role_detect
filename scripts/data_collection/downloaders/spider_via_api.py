#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过API接口爬取缺失角色的URL
"""

import os
import sys
import time
import requests
import logging
from pypinyin import lazy_pinyin, Style

# 配置 - 使用正确的API路径
API_BASE_URL = "http://localhost:33333/api/v1.2.5.260305"
ROLE_LIST_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_all_roles():
    """获取完整角色列表"""
    roles = []
    with open(ROLE_LIST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ')
                chinese_name = parts[0]
                roles.append(chinese_name)
    return roles


def get_missing_roles():
    """获取需要爬取的角色列表 - 根据拼音生成文件名检查"""
    all_roles = get_all_roles()
    existing_files = set()
    
    if os.path.exists(URL_DIR):
        for filename in os.listdir(URL_DIR):
            if filename.endswith('_img.txt'):
                role_pinyin = filename.replace('_img.txt', '')
                existing_files.add(role_pinyin)
    
    missing = []
    for role in all_roles:
        # 生成拼音文件名
        pinyin = ''.join(lazy_pinyin(role, style=Style.TONE3))
        # 检查是否存在对应的URL文件
        if f"{pinyin}_img.txt" not in existing_files:
            missing.append(role)
    
    return missing, len(all_roles)


def start_spider_via_api(keyword):
    """通过API接口启动爬虫"""
    try:
        url = f"{API_BASE_URL}/sis/spider_start/single"
        params = {"key_word": keyword}
        response = requests.post(url, params=params, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                return True
            else:
                logger.error(f"API返回错误: {result.get('msg')}")
                return False
        else:
            logger.error(f"API返回错误状态码: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"API调用失败: {str(e)}")
        return False


def check_api_status():
    """检查API服务是否可用"""
    try:
        response = requests.get(f"{API_BASE_URL}/sis/spider_image/config", timeout=5)
        return response.status_code == 200
    except Exception as e:
        return False


def spider_via_api():
    """主函数：通过API爬取缺失角色"""
    logger.info("检查API服务状态...")
    if not check_api_status():
        logger.error("API服务不可用，请先启动爬虫服务！")
        return
    
    logger.info("获取缺失角色列表...")
    missing_roles, total = get_missing_roles()
    
    if not missing_roles:
        logger.info("所有角色的URL都已采集完成！")
        return
    
    logger.info(f"总角色数: {total}")
    logger.info(f"已采集: {total - len(missing_roles)}")
    logger.info(f"缺少URL: {len(missing_roles)}")
    logger.info(f"缺失角色: {', '.join(missing_roles[:10])}{'...' if len(missing_roles) > 10 else ''}")
    
    # 逐个爬取
    success_count = 0
    for i, role in enumerate(missing_roles, 1):
        logger.info(f"[{i}/{len(missing_roles)}] 开始爬取角色: {role}")
        
        if start_spider_via_api(role):
            logger.info(f"✓ 成功启动爬虫: {role}")
            success_count += 1
        else:
            logger.error(f"✗ 启动爬虫失败: {role}")
        
        # 间隔30秒再爬取下一个
        time.sleep(30)
    
    logger.info(f"\n=== 爬取启动完成 ===")
    logger.info(f"成功: {success_count}/{len(missing_roles)}")


if __name__ == '__main__':
    spider_via_api()

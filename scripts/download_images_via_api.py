#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过API接口下载图片 - 使用英文名下载
支持超时处理，避免下载卡死
"""

import os
import sys
import time
import requests
import logging
import threading

# 配置
API_BASE_URL = "http://localhost:33333/api/v1.2.5.260305"
DATASET_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/merged_english_dataset'
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url_english'

# 下载配置
TARGET_COUNT = 100  # 每个角色目标图片数量
TIMEOUT_PER_URL = 30  # 单URL下载超时（秒）
TIMEOUT_TOTAL = 300  # 总超时（秒）

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_api_status():
    """检查API服务是否可用"""
    try:
        response = requests.get(f"{API_BASE_URL}/sis/spider_image/config", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"API服务不可用: {e}")
        return False


def download_image_via_api(keyword, save_dir, target_count=100):
    """
    通过API下载图片
    :param keyword: 搜索关键词（英文名）
    :param save_dir: 保存目录
    :param target_count: 目标下载数量
    :return: (成功数, 失败数)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查当前已有图片数量
    current_count = len([f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    if current_count >= target_count:
        logger.info(f"⏭️ {keyword}: 已有 {current_count} 张图片，无需补充")
        return 0, 0, current_count
    
    need_count = target_count - current_count
    logger.info(f"🔄 {keyword}: 开始下载，需补充 {need_count} 张图片")
    
    # 调用API下载
    try:
        url = f"{API_BASE_URL}/sis/spider_image/download"
        params = {
            "key_word": keyword,
            "count": need_count,
            "save_path": save_dir
        }
        
        response = requests.post(url, params=params, timeout=TIMEOUT_TOTAL)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                success_count = result.get("data", {}).get("success_count", 0)
                fail_count = result.get("data", {}).get("fail_count", 0)
                logger.info(f"✅ {keyword}: 下载完成 - 成功 {success_count} 张, 失败 {fail_count} 张")
                return success_count, fail_count, current_count + success_count
            else:
                logger.error(f"❌ {keyword}: API返回错误 - {result.get('msg', '未知错误')}")
                return 0, 0, current_count
        else:
            logger.error(f"❌ {keyword}: HTTP错误 - {response.status_code}")
            return 0, 0, current_count
    except requests.Timeout:
        logger.error(f"❌ {keyword}: 下载超时")
        return 0, 0, current_count
    except Exception as e:
        logger.error(f"❌ {keyword}: 下载异常 - {e}")
        return 0, 0, current_count


def get_roles_needing_download():
    """获取需要下载图片的角色列表"""
    roles = []
    
    for role_dir in os.listdir(DATASET_DIR):
        role_path = os.path.join(DATASET_DIR, role_dir)
        if not os.path.isdir(role_path):
            continue
        
        # 检查当前图片数量
        current_count = len([f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        
        # 如果数量不足100，需要补充
        if current_count < TARGET_COUNT:
            roles.append({
                'english': role_dir,
                'current_count': current_count,
                'need_count': TARGET_COUNT - current_count
            })
    
    # 按需要补充的数量排序（优先补充数量少的）
    roles.sort(key=lambda x: x['current_count'])
    return roles


def main():
    logger.info("=" * 60)
    logger.info("          通过API补充角色图片")
    logger.info("=" * 60)
    
    # 检查API服务
    if not check_api_status():
        logger.error("API服务不可用，请先启动爬虫服务！")
        return
    
    # 获取需要下载的角色列表
    roles = get_roles_needing_download()
    
    if not roles:
        logger.info("🎉 所有角色图片数量已达标！")
        return
    
    logger.info(f"\n待补充角色: {len(roles)} 个")
    for role in roles[:10]:
        logger.info(f"  - {role['english']}: {role['current_count']}/{TARGET_COUNT} (需补充 {role['need_count']})")
    if len(roles) > 10:
        logger.info(f"  ... 还有 {len(roles) - 10} 个角色")
    
    # 逐个下载
    total_success = 0
    total_fail = 0
    
    for i, role in enumerate(roles, 1):
        english_name = role['english']
        save_dir = os.path.join(DATASET_DIR, english_name)
        
        logger.info(f"\n[{i}/{len(roles)}] 处理: {english_name}")
        
        success, fail, final_count = download_image_via_api(english_name, save_dir, TARGET_COUNT)
        total_success += success
        total_fail += fail
        
        # 等待5秒再下载下一个
        time.sleep(5)
    
    logger.info("\n" + "=" * 60)
    logger.info("下载完成!")
    logger.info(f"  成功下载: {total_success} 张")
    logger.info(f"  下载失败: {total_fail} 张")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

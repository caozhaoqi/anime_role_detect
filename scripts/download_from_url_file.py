#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从URL文件直接下载图片到数据集目录
使用英文名命名，支持超时处理
"""

import os
import sys
import time
import requests
import logging
import hashlib
from PIL import Image

# 配置
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url_english'
DATASET_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/merged_english_dataset'

# 下载配置
TARGET_COUNT = 100  # 每个角色目标图片数量
TIMEOUT_PER_URL = 30  # 单URL下载超时（秒）
MAX_RETRIES = 2  # 重试次数

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_single_image(url, save_path):
    """下载单张图片"""
    try:
        response = requests.get(url, timeout=TIMEOUT_PER_URL, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            
            # 验证图片是否有效
            try:
                img = Image.open(save_path)
                img.verify()
                img.close()
                return True
            except:
                os.remove(save_path)
                return False
        return False
    except Exception as e:
        logger.debug(f"下载失败: {url} - {e}")
        return False


def download_role_images(role_name):
    """下载单个角色的图片"""
    # 检查URL文件是否存在
    url_file = os.path.join(URL_DIR, f"{role_name}_img.txt")
    if not os.path.exists(url_file):
        logger.warning(f"⏭️ {role_name}: 未找到URL文件")
        return 0, 0
    
    # 创建保存目录
    save_dir = os.path.join(DATASET_DIR, role_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查当前已有图片数量
    existing_images = [f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    current_count = len(existing_images)
    
    if current_count >= TARGET_COUNT:
        logger.info(f"⏭️ {role_name}: 已有 {current_count} 张图片，无需补充")
        return 0, 0
    
    need_count = TARGET_COUNT - current_count
    logger.info(f"🔄 {role_name}: 开始下载，需补充 {need_count} 张图片")
    
    # 读取URL列表
    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    if not urls:
        logger.error(f"❌ {role_name}: URL文件为空")
        return 0, 0
    
    # 记录已下载的图片哈希（去重）
    existing_hashes = set()
    for img_file in existing_images:
        img_path = os.path.join(save_dir, img_file)
        try:
            with open(img_path, 'rb') as f:
                existing_hashes.add(hashlib.md5(f.read()).hexdigest())
        except:
            pass
    
    success_count = 0
    fail_count = 0
    
    for url in urls:
        if success_count >= need_count:
            break
        
        # 下载并重试
        success = False
        for retry in range(MAX_RETRIES):
            try:
                # 生成文件名（使用URL哈希）
                url_hash = hashlib.md5(url.encode()).hexdigest()
                save_path = os.path.join(save_dir, f"{url_hash}.jpg")
                
                # 检查是否已存在
                if os.path.exists(save_path):
                    continue
                
                # 下载图片
                if download_single_image(url, save_path):
                    # 验证下载的图片
                    try:
                        with open(save_path, 'rb') as f:
                            new_hash = hashlib.md5(f.read()).hexdigest()
                        if new_hash in existing_hashes:
                            os.remove(save_path)
                            continue
                        existing_hashes.add(new_hash)
                    except:
                        os.remove(save_path)
                        continue
                    
                    success_count += 1
                    success = True
                    break
                else:
                    fail_count += 1
            except Exception as e:
                logger.debug(f"下载异常: {url} - {e}")
                fail_count += 1
        
        if success:
            logger.debug(f"✅ {role_name}: 已下载 {success_count}/{need_count}")
    
    logger.info(f"✅ {role_name}: 下载完成 - 成功 {success_count} 张, 失败 {fail_count} 张")
    return success_count, fail_count


def get_roles_needing_download():
    """获取需要下载图片的角色列表"""
    roles = []
    
    # 获取所有URL文件对应的角色
    url_roles = set()
    if os.path.exists(URL_DIR):
        for filename in os.listdir(URL_DIR):
            if filename.endswith('_img.txt'):
                role_name = filename.replace('_img.txt', '')
                url_roles.add(role_name)
    
    # 获取数据集中已有角色
    dataset_roles = set()
    if os.path.exists(DATASET_DIR):
        for dirname in os.listdir(DATASET_DIR):
            dir_path = os.path.join(DATASET_DIR, dirname)
            if os.path.isdir(dir_path):
                dataset_roles.add(dirname)
    
    # 合并并检查每个角色的图片数量
    all_roles = url_roles.union(dataset_roles)
    
    for role_name in all_roles:
        role_path = os.path.join(DATASET_DIR, role_name)
        if os.path.isdir(role_path):
            current_count = len([f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        else:
            current_count = 0
        
        if current_count < TARGET_COUNT:
            roles.append({
                'name': role_name,
                'current_count': current_count,
                'need_count': TARGET_COUNT - current_count
            })
    
    # 按需要补充的数量排序
    roles.sort(key=lambda x: x['current_count'])
    return roles


def main():
    logger.info("=" * 60)
    logger.info("          从URL文件下载图片")
    logger.info("=" * 60)
    
    # 获取需要下载的角色列表
    roles = get_roles_needing_download()
    
    if not roles:
        logger.info("🎉 所有角色图片数量已达标！")
        return
    
    logger.info(f"\n待补充角色: {len(roles)} 个")
    for role in roles[:10]:
        logger.info(f"  - {role['name']}: {role['current_count']}/{TARGET_COUNT} (需补充 {role['need_count']})")
    if len(roles) > 10:
        logger.info(f"  ... 还有 {len(roles) - 10} 个角色")
    
    # 逐个下载
    total_success = 0
    total_fail = 0
    
    for i, role in enumerate(roles, 1):
        role_name = role['name']
        
        logger.info(f"\n[{i}/{len(roles)}] 处理: {role_name}")
        
        success, fail = download_role_images(role_name)
        total_success += success
        total_fail += fail
        
        # 等待2秒再下载下一个
        time.sleep(2)
    
    logger.info("\n" + "=" * 60)
    logger.info("下载完成!")
    logger.info(f"  成功下载: {total_success} 张")
    logger.info(f"  下载失败: {total_fail} 张")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

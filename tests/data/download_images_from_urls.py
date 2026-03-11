#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据URL文件下载图片
"""

import os
import requests
import time
import logging
import json
import random
from urllib.parse import urlparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('download_images_from_urls')

# 数据目录
DATA_DIR = "data"
IMG_URL_DIR = os.path.join(DATA_DIR, "img_url")
DOWNLOADED_DIR = os.path.join(DATA_DIR, "downloaded_images")
ATTRIBUTES_DIR = os.path.join(DATA_DIR, "attributes")

# 角色拼音到名称的映射
ROLE_MAPPING = {
    "a1luo2na4": "阿罗娜",
    "pu3la1na4": "普拉娜",
    "ri4nai4": "日奈",
    "xiang3": "亚子",
    "yi1zhi1": "伊织",
    "qian1xia4": "千夏",
    "feng1xiang1": "枫香"
}

# 请求头信息
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com/"
}

# 代理服务器列表（可选）
PROXIES = [
    # 示例代理，实际使用时需要替换为有效的代理
    # {"http": "http://proxy1:port", "https": "https://proxy1:port"},
    # {"http": "http://proxy2:port", "https": "https://proxy2:port"},
]

# 下载重试次数
MAX_RETRIES = 3

# 随机延迟范围（秒）
MIN_DELAY = 0.5
MAX_DELAY = 2.0

def load_character_attributes():
    """加载角色属性配置"""
    config_path = os.path.join("config", "character_attributes.json")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"无法加载角色属性配置: {e}，将使用默认配置")
        return {"characters": {}, "attribute_mappings": {}, "attribute_order": []}


def get_attribute_labels(character_name, attribute_config):
    """获取角色的属性标签
    
    Args:
        character_name: 角色名称
        attribute_config: 属性配置
    
    Returns:
        list: 属性标签列表
    """
    characters = attribute_config.get("characters", {})
    attribute_order = attribute_config.get("attribute_order", [])
    attribute_mappings = attribute_config.get("attribute_mappings", {})
    
    if character_name not in characters:
        logger.warning(f"角色 {character_name} 不在属性配置中")
        return [0] * len(attribute_order)
    
    character_attrs = characters[character_name]
    attribute_labels = []
    
    for attr_name in attribute_order:
        attr_value = character_attrs.get(attr_name, "unknown")
        if isinstance(attr_value, bool):
            attr_value = str(attr_value).lower()
        
        mapping = attribute_mappings.get(attr_name, {})
        label = mapping.get(attr_value, 0)
        attribute_labels.append(label)
    
    return attribute_labels

def download_image(url, save_path):
    """下载单个图片
    
    Args:
        url: 图片URL
        save_path: 保存路径
    
    Returns:
        bool: 是否下载成功
    """
    for attempt in range(MAX_RETRIES):
        try:
            # 选择随机代理（如果有）
            proxies = random.choice(PROXIES) if PROXIES else None
            
            # 发送请求
            response = requests.get(
                url, 
                headers=HEADERS, 
                proxies=proxies, 
                timeout=15, 
                allow_redirects=True
            )
            response.raise_for_status()
            
            # 保存图片
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.debug(f"下载成功: {url} -> {save_path}")
            return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"下载失败 (尝试 {attempt+1}/{MAX_RETRIES}): {url}, 错误: {e}")
            if attempt < MAX_RETRIES - 1:
                # 重试前随机延迟
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                time.sleep(delay)
            else:
                logger.error(f"下载失败（达到最大重试次数）: {url}, 错误: {e}")
                return False

def process_url_file(file_name, role_name):
    """处理URL文件
    
    Args:
        file_name: URL文件名称
        role_name: 角色名称
    """
    file_path = os.path.join(IMG_URL_DIR, file_name)
    save_dir = os.path.join(DOWNLOADED_DIR, role_name)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载角色属性配置
    attribute_config = load_character_attributes()
    attribute_labels = get_attribute_labels(role_name, attribute_config)
    attribute_order = attribute_config.get("attribute_order", [])
    
    # 读取URL文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"读取URL文件失败: {file_path}, 错误: {e}")
        return
    
    logger.info(f"开始下载 {role_name} 的图片，共 {len(urls)} 张")
    logger.info(f"角色属性: {dict(zip(attribute_order, attribute_labels))}")
    
    # 下载图片
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # 属性注释文件
    annotations = []
    
    for i, url in enumerate(urls, 1):
        # 生成文件名
        parsed_url = urlparse(url)
        file_ext = os.path.splitext(parsed_url.path)[1]
        
        # 过滤SVG格式
        if file_ext.lower() == ".svg":
            logger.info(f"跳过SVG格式图片: {url}")
            skip_count += 1
            continue
        
        if not file_ext:
            file_ext = ".jpg"
        file_name = f"{role_name}_{i}{file_ext}"
        save_path = os.path.join(save_dir, file_name)
        
        # 检查文件是否已存在
        if os.path.exists(save_path):
            logger.info(f"图片已存在，跳过: {file_name}")
            skip_count += 1
            # 添加属性注释
            annotations.append({
                "image_path": os.path.join(role_name, file_name),
                "character": role_name,
                "attributes": dict(zip(attribute_order, attribute_labels)),
                "attribute_labels": attribute_labels
            })
            continue
        
        # 下载图片
        if download_image(url, save_path):
            success_count += 1
            # 添加属性注释
            annotations.append({
                "image_path": os.path.join(role_name, file_name),
                "character": role_name,
                "attributes": dict(zip(attribute_order, attribute_labels)),
                "attribute_labels": attribute_labels
            })
        else:
            fail_count += 1
        
        # 避免请求过快，使用随机延迟
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(delay)
    
    # 保存属性注释文件
    os.makedirs(ATTRIBUTES_DIR, exist_ok=True)
    annotation_file = os.path.join(ATTRIBUTES_DIR, f"{role_name}_annotations.json")
    try:
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        logger.info(f"属性注释已保存到: {annotation_file}")
    except Exception as e:
        logger.error(f"保存属性注释失败: {e}")
    
    logger.info(f"{role_name} 图片下载完成，成功: {success_count}, 失败: {fail_count}, 跳过: {skip_count}")

def main():
    """主函数"""
    # 处理每个角色的URL文件
    for url_file, role_name in ROLE_MAPPING.items():
        file_name = f"{url_file}_img.txt"
        file_path = os.path.join(IMG_URL_DIR, file_name)
        if os.path.exists(file_path):
            logger.info(f"开始处理 {role_name} 的图片URL文件: {file_name}")
            process_url_file(file_name, role_name)
        else:
            logger.warning(f"URL文件不存在: {file_name}")
    
    # 检查是否有未在映射中的URL文件
    existing_files = [f for f in os.listdir(IMG_URL_DIR) if f.endswith('_img.txt')]
    mapped_files = [f"{url_file}_img.txt" for url_file in ROLE_MAPPING.keys()]
    
    for file_name in existing_files:
        if file_name not in mapped_files:
            logger.warning(f"未在角色映射中找到 {file_name} 对应的角色名称")
    
    logger.info("所有图片下载任务完成")

if __name__ == "__main__":
    main()

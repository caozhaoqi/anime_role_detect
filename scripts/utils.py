#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一工具函数
"""

import os
import hashlib
import requests
import time
import urllib.parse
from .config import *

def get_image_count(role_dir):
    """获取角色当前图片数量"""
    if not os.path.exists(role_dir):
        return 0
    return len([f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])

def get_image_files(role_dir):
    """获取角色当前已有的图片文件名集合"""
    if not os.path.exists(role_dir):
        return set()
    return set([f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])

def find_url_file(identifier):
    """查找角色的图片URL文件"""
    patterns = [
        f"{identifier}_img.txt",
        f"{urllib.parse.quote(identifier)}_img.txt"
    ]
    
    for pattern in patterns:
        filepath = os.path.join(URL_DIR, pattern)
        if os.path.exists(filepath):
            return filepath
    
    for filename in os.listdir(URL_DIR):
        if identifier in filename and filename.endswith('_img.txt'):
            return os.path.join(URL_DIR, filename)
    
    return None

def download_image_from_url(img_url):
    """从URL下载单张图片"""
    try:
        response = requests.get(img_url, timeout=10)
        if response.status_code == 200:
            md5_hash = hashlib.md5(response.content).hexdigest()
            ext = '.jpg'
            if 'png' in img_url.lower():
                ext = '.png'
            elif 'webp' in img_url.lower():
                ext = '.webp'
            return md5_hash, ext, response.content
    except Exception:
        pass
    return None, None, None

def download_images(url_file, role_dir, need_count, existing_files):
    """批量下载图片"""
    downloaded = 0
    
    if not os.path.exists(url_file):
        return downloaded
    
    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    for img_url in urls:
        if downloaded >= need_count:
            break
        
        md5_hash, ext, content = download_image_from_url(img_url)
        if md5_hash and content:
            filename = f'{md5_hash}{ext}'
            if filename not in existing_files:
                with open(os.path.join(role_dir, filename), 'wb') as f:
                    f.write(content)
                downloaded += 1
                existing_files.add(filename)
    
    return downloaded

def wait_for_spider(timeout=180):
    """等待爬虫完成"""
    wait_time = 0
    while wait_time < timeout:
        try:
            status = requests.get(f"{API_BASE}/spider/status", timeout=10)
            data = status.json()
            if data.get('code') == 0 and not data.get('data', {}).get('is_running', True):
                return True
            time.sleep(3)
            wait_time += 3
        except Exception:
            return False
    return False

def spider_single_role(keyword):
    """调用爬虫API采集角色URL"""
    encoded_keyword = urllib.parse.quote(keyword)
    url = f"{API_BASE}/spider_start/single?key_word={encoded_keyword}"
    try:
        response = requests.post(url, timeout=30)
        result = response.json()
        if result.get('code') == 0:
            wait_for_spider()
            return True
    except Exception:
        pass
    return False

def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def read_role_file(filepath):
    """读取角色名单文件"""
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def parse_role_line(line):
    """解析角色行"""
    parts = line.split()
    if len(parts) >= 4:
        return {
            'name': parts[0],
            'work': parts[1],
            'en': parts[2],
            'jp': ' '.join(parts[3:])
        }
    return {'name': parts[0], 'work': '', 'en': '', 'jp': ''}

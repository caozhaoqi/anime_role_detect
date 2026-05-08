#!/usr/bin/env python3
"""继续采集芙丽希娅的图片"""
import os
import sys
import requests
import hashlib
import time
from pathlib import Path

sys.path.insert(0, 'spider_image_system/src/run')
from constants import PINYIN_MAPPING

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.pixiv.net/',
    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

def download_image(url, save_path, max_retries=3):
    """下载单张图片"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            else:
                print(f"   ⚠️ 尝试 {attempt+1}/{max_retries}: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ⚠️ 尝试 {attempt+1}/{max_retries}: {e}")
        time.sleep(1)
    return False

def spider_furisia():
    """采集芙丽希娅的图片"""
    role_name = '芙丽希娅'
    pinyin = PINYIN_MAPPING.get(role_name)
    if not pinyin:
        print(f"❌ 未找到 {role_name} 的拼音映射")
        return
    
    # 目标目录
    target_dir = Path(f'data/organized_images/{pinyin}')
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取已有图片数量
    existing_count = len(list(target_dir.glob('*.jpg'))) + len(list(target_dir.glob('*.png'))) + len(list(target_dir.glob('*.webp')))
    print(f"📋 {role_name}: 当前 {existing_count} 张图片")
    
    # 需要采集的数量
    needed = max(0, 50 - existing_count)
    print(f"🎯 需要再采集 {needed} 张图片")
    
    if needed <= 0:
        print("✅ 已达到目标数量")
        return
    
    # API配置
    API_BASE = 'http://localhost:33334/api/v1.2.5.260305/sis'
    
    # 启动URL采集
    print(f"🚀 开始采集 {role_name} 的URL...")
    url = f"{API_BASE}/spider_start/single?key_word={role_name}"
    try:
        response = requests.post(url, timeout=300)
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                print(f"✅ URL采集任务已启动")
            else:
                print(f"❌ URL采集失败: {result.get('msg', '未知错误')}")
                return
        else:
            print(f"❌ URL采集请求失败: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ URL采集异常: {e}")
        return
    
    # 等待采集完成
    print("⏳ 等待URL采集完成...")
    while True:
        try:
            response = requests.get(f"{API_BASE}/spider/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                if status.get("code") == 0:
                    data = status.get("data", {})
                    if data.get("is_running") == False:
                        print("✅ URL采集完成")
                        break
            time.sleep(3)
        except Exception as e:
            print(f"⚠️ 检查状态异常: {e}")
            time.sleep(3)
    
    # 读取新采集的URL
    img_url_file = f'spider_image_system/data/img_url/{pinyin}_img.txt'
    if not os.path.exists(img_url_file):
        print(f"❌ 图片URL文件不存在")
        return
    
    with open(img_url_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    urls = list(set(urls))  # 去重
    print(f"📥 找到 {len(urls)} 个图片URL")
    
    # 下载图片
    downloaded = 0
    failed = 0
    
    for i, url in enumerate(urls, 1):
        # 生成文件名
        url_hash = hashlib.md5(url.encode()).hexdigest()
        ext = '.jpg' if '.jpg' in url or 'jpg' in url else '.png'
        save_path = target_dir / f"{url_hash}{ext}"
        
        # 如果文件已存在，跳过
        if save_path.exists():
            continue
        
        print(f"   [{i}/{len(urls)}] 下载中...", end='\r')
        
        if download_image(url, save_path):
            downloaded += 1
            if downloaded >= needed:
                break
        else:
            failed += 1
        
        time.sleep(0.5)
    
    # 最终统计
    final_count = len(list(target_dir.glob('*.jpg'))) + len(list(target_dir.glob('*.png'))) + len(list(target_dir.glob('*.webp')))
    print(f"\n✅ 成功下载: {downloaded}张, 失败: {failed}张, 总计: {final_count}张")

def main():
    role_name = '芙丽希娅'
    print("=" * 60)
    print(f"📷 采集 {role_name} 的图片")
    print("=" * 60)
    
    spider_furisia()
    
    print("\n" + "=" * 60)
    print("📊 采集完成")
    print("=" * 60)

if __name__ == '__main__':
    main()

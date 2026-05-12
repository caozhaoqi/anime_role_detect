#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优先从URL下载补充图片脚本 - 修复版
"""

import os
import requests
import hashlib
import time
import urllib.parse

DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'
API_BASE = "http://localhost:33333/api/v1.2.5.260305/sis"
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/href_url'
TARGET_COUNT = 100

PINYIN_MAPPING = {
    'a1luo4na4': '阿洛娜',
    'a1ni4ya4': '阿妮娅',
    'an1ke3': '安可',
    'bai2shang4chui1xue3': '白上吹雪',
    'bu4luo4ni2ya4': '布洛妮娅',
    'de2li4sha1': '德丽莎',
    'di2ao4na4': '迪奥娜',
    'duo1li4': '多莉',
    'fei1mi3li4si1': '菲米莉丝',
    'fu2lan2': '芙兰',
    'fu2xuan2': '符玄',
    'hua1huo3': '火花',
    'ka3qi2na4': '卡琪娜',
    'kai3lu4': '开萝',
    'kang1na4': '康娜',
    'ke1xie4ni2ya4': '克谢尼娅',
    'ke3li4': '刻晴',
    'ke3lin2': '克林',
    'ke4luo2li4ke1': '克罗丽克',
    'kou4er3fu2': '蔻尔芙',
    'la1mu3': '拉姆',
    'lei2mu3': '雷姆',
    'lei3bei4': '蕾贝',
    'li4ta3la1': '莉塔菈',
    'luo4qian4': '洛茜',
    'mao1gong1you4nai4': '猫宫又奈',
    'mi2dou4zi': '蜜豆子',
    'na4gan1': '娜甘',
    'na4xi1da2': '纳西妲',
    'pei4li3ti2ya4': '佩丽缇娅',
    'qi1qi1': '七七',
    'san1yue4qi1': '三月七',
    'si4mi4nai3': '四宫奈',
    'ti2bao3': '提宝',
    'tian1tong2ai4li4si1': '天音爱莉丝',
    'wei2li3nai4': '薇莉奈',
    'wei2pu3lei3': '薇普蕾',
    'xi1ge2wen2': '希格雯',
    'xia4ke4li3': '夏可莉',
    'xiao3mai2': '小麦',
    'xiao3mei3yan4': '小美焰',
    'xing4': '星',
    'xue4xiao3ban3': '学校版',
    'yi1se4lin2': '伊瑟琳',
    'you4hu2': '釉瑚',
    'yue4qian1ye4': '月千夜',
    'zao3wu4': '早雾',
    'zhi4nai3': '智乃'
}

def get_current_count(pinyin):
    """获取角色当前图片数量"""
    role_dir = os.path.join(DATA_DIR, pinyin)
    if not os.path.exists(role_dir):
        return 0
    return len([f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])

def get_current_images(pinyin):
    """获取角色当前已有的图片文件名集合"""
    role_dir = os.path.join(DATA_DIR, pinyin)
    if not os.path.exists(role_dir):
        return set()
    return set([f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])

def wait_for_spider():
    """等待爬虫完成"""
    while True:
        try:
            status = requests.get(f"{API_BASE}/spider/status", timeout=10)
            data = status.json()
            if data.get('code') == 0 and not data.get('data', {}).get('is_running', True):
                return True
            time.sleep(2)
        except Exception:
            return False

def spider_single_role(role_name):
    """调用爬虫API采集角色URL"""
    encoded_name = urllib.parse.quote(role_name)
    url = f"{API_BASE}/spider_start/single?key_word={encoded_name}"
    try:
        response = requests.post(url, timeout=30)
        result = response.json()
        if result.get('code') == 0:
            print(f"  🕷️ 启动采集: {result.get('msg')}")
            wait_for_spider()
            return True
        else:
            print(f"  ❌ 采集失败: {result.get('msg')}")
            return False
    except Exception as e:
        print(f"  ❌ 采集异常: {str(e)}")
        return False

def find_url_file(pinyin, role_name):
    """查找角色的URL文件"""
    # 尝试多种文件名格式
    patterns = [
        f"{pinyin}_result_url.txt",
        f"{pinyin}_url.txt",
        f"{urllib.parse.quote(role_name)}_result_url.txt",
        f"{urllib.parse.quote(role_name)}_url.txt",
        f"{role_name}_result_url.txt",
        f"{role_name}_url.txt"
    ]
    
    for pattern in patterns:
        filepath = os.path.join(URL_DIR, pattern)
        if os.path.exists(filepath):
            return filepath
    
    # 也尝试查找包含pinyin的文件
    for filename in os.listdir(URL_DIR):
        if pinyin in filename and filename.endswith('_url.txt'):
            return os.path.join(URL_DIR, filename)
        if role_name in filename and filename.endswith('_url.txt'):
            return os.path.join(URL_DIR, filename)
    
    return None

def download_images(pinyin, role_name, need_count):
    """下载图片"""
    role_dir = os.path.join(DATA_DIR, pinyin)
    current_imgs = get_current_images(pinyin)
    downloaded = 0
    
    # 查找URL文件
    url_file = find_url_file(pinyin, role_name)
    
    if url_file:
        print(f"  📋 找到URL文件: {os.path.basename(url_file)}")
        with open(url_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        print(f"  📋 读取到 {len(urls)} 个URL")
        
        for img_url in urls:
            if downloaded >= need_count:
                break
            
            try:
                img_response = requests.get(img_url, timeout=10)
                if img_response.status_code == 200:
                    md5_hash = hashlib.md5(img_response.content).hexdigest()
                    ext = '.jpg'
                    if 'png' in img_url.lower():
                        ext = '.png'
                    elif 'webp' in img_url.lower():
                        ext = '.webp'
                    
                    filename = f'{md5_hash}{ext}'
                    
                    if filename not in current_imgs:
                        with open(os.path.join(role_dir, filename), 'wb') as f:
                            f.write(img_response.content)
                        downloaded += 1
                        current_imgs.add(filename)
            except Exception:
                continue
    else:
        print(f"  ⚠️ 未找到URL文件")
    
    return downloaded

def main():
    print("=" * 70)
    print("🚀 优先从URL下载补充图片")
    print("目标: 每个角色100张")
    print("API: ", API_BASE)
    print("URL目录: ", URL_DIR)
    print("=" * 70)
    
    total_downloaded = 0
    remaining_roles = []
    
    # 检查爬虫服务
    try:
        response = requests.get(f"{API_BASE}/spider/status", timeout=5)
        print("✅ 爬虫服务连接成功")
    except Exception as e:
        print(f"❌ 爬虫服务未运行: {str(e)}")
        return
    
    for pinyin, name in PINYIN_MAPPING.items():
        current = get_current_count(pinyin)
        if current >= TARGET_COUNT:
            print(f"  ✅ {name}: {current} 张 (已达标)")
            continue
        
        need_count = TARGET_COUNT - current
        print(f"\n📦 {name} ({pinyin}): 当前 {current} 张, 需要补充 {need_count} 张")
        
        # 1. 检查是否已有URL文件
        url_file = find_url_file(pinyin, name)
        if url_file:
            print("  步骤1: URL文件已存在，跳过采集")
        else:
            print("  步骤1: 采集URL...")
            spider_single_role(name)
        
        # 2. 下载图片
        print("  步骤2: 下载图片...")
        downloaded = download_images(pinyin, name, need_count)
        
        if downloaded > 0:
            print(f"  📥 成功下载: {downloaded} 张")
            total_downloaded += downloaded
        
        current = get_current_count(pinyin)
        
        if current >= TARGET_COUNT:
            print(f"  ✅ 当前: {current} 张 (已达标)")
        else:
            print(f"  ⚠️ 当前: {current} 张 (仍需 {TARGET_COUNT - current} 张)")
            remaining_roles.append((name, current))
    
    print("\n" + "=" * 70)
    print(f"总共下载: {total_downloaded} 张")
    print("=" * 70)
    
    total_images = sum(get_current_count(p) for p in PINYIN_MAPPING.keys())
    
    print(f"\n📊 最终统计:")
    print(f"角色总数: {len(PINYIN_MAPPING)}")
    print(f"图片总数: {total_images}")
    print(f"平均每角色: {total_images / len(PINYIN_MAPPING):.1f} 张")
    
    if remaining_roles:
        print(f"\n⚠️ 仍有 {len(remaining_roles)} 个角色未达标")
        for name, count in remaining_roles[:5]:
            print(f"  {name}: {count}/{TARGET_COUNT}")
        if len(remaining_roles) > 5:
            print(f"  ...还有 {len(remaining_roles) - 5} 个")
    else:
        print("\n🎉 所有角色均已达标！")

if __name__ == '__main__':
    main()

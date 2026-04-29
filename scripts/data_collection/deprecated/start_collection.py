#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据采集入口脚本
根据 loli-role.txt 中的角色列表开始数据采集
"""

import os
import sys
import requests
from PIL import Image
import io
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置参数
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOLI_ROLE_FILE = BASE_DIR / "auto_spider_img" / "loli-role.txt"
DATA_DIR = BASE_DIR / "data" / "loli_training_data"
MAX_IMAGES_PER_ROLE = 100
TIMEOUT = 15
DELAY = 0.5
MAX_WORKERS = 5

# 创建数据目录
os.makedirs(DATA_DIR, exist_ok=True)

# 图片源配置
IMAGE_SOURCES = [
    ("sd.vv50.de", "https://sd.vv50.de/search?q={keyword}"),
    ("waifu.pics", "https://waifu.pics/api/sfw?tag={keyword}"),
]

def is_valid_image(content):
    """检查是否为有效图片"""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except:
        return False

def parse_loli_role_file(filepath):
    """解析萝莉角色文件"""
    roles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                chinese_name = parts[0]
                source = parts[1]
                english_name = parts[2] if len(parts) > 2 else ""
                japanese_name = parts[3] if len(parts) > 3 else ""
                roles.append({
                    "chinese": chinese_name,
                    "source": source,
                    "english": english_name,
                    "japanese": japanese_name
                })
    return roles

def get_search_keywords(role):
    """获取角色的搜索关键词列表"""
    keywords = []
    if role["japanese"]:
        keywords.append(role["japanese"])
    if role["english"]:
        keywords.append(role["english"])
    if role["chinese"]:
        keywords.append(role["chinese"])
    return keywords

def create_role_directory(role_name, source):
    """创建角色目录"""
    dir_name = f"{role_name}_{source}"
    role_dir = DATA_DIR / dir_name
    os.makedirs(role_dir, exist_ok=True)
    return role_dir

def download_image(url, save_dir, role_name, timeout=15):
    """下载单张图片"""
    try:
        headers = {
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
            ])
        }

        response = requests.get(url, headers=headers, timeout=timeout)

        if response.status_code == 200:
            if is_valid_image(response.content):
                url_hash = abs(hash(url)) % 1000000
                filename = f"{url_hash:06d}.jpg"
                filepath = save_dir / filename

                if filepath.exists():
                    return False, "文件已存在"

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                return True, filename
            else:
                return False, "无效图片"
        else:
            return False, f"HTTP {response.status_code}"

    except Exception as e:
        return False, str(e)

def collect_from_sdvv50(role, role_dir, max_images=50):
    """从 sd.vv50.de 采集图片"""
    keywords = get_search_keywords(role)
    collected = 0

    for keyword in keywords:
        if collected >= max_images:
            break

        url = f"https://sd.vv50.de/search?q={requests.utils.quote(keyword)}"
        try:
            # 使用 Selenium 处理 JavaScript 渲染
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from webdriver_manager.chrome import ChromeDriverManager

            # 配置 Chrome 选项
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)

            # 启动浏览器
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(60)

            # 访问页面
            driver.get(url)
            
            # 等待页面加载
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # 等待 JavaScript 渲染
            time.sleep(5)

            # 查找图片元素
            # 尝试不同的选择器
            img_elements = []
            
            # 查找所有 img 标签
            img_elements = driver.find_elements(By.TAG_NAME, "img")
            
            # 也尝试查找包含图片的 div 元素
            div_elements = driver.find_elements(By.CSS_SELECTOR, "div[class*='image']")
            for div in div_elements:
                try:
                    img = div.find_element(By.TAG_NAME, "img")
                    img_elements.append(img)
                except:
                    pass

            logger.info(f"找到 {len(img_elements)} 个图片元素")

            for img in img_elements:
                if collected >= max_images:
                    break

                try:
                    # 尝试获取不同的图片属性
                    img_url = None
                    for attr in ['src', 'data-src', 'data-original']:
                        try:
                            img_url = img.get_attribute(attr)
                            if img_url and img_url.startswith('http'):
                                break
                        except:
                            pass

                    if img_url and img_url.startswith('http'):
                        # 过滤掉小图标和头像
                        if 'avatar' in img_url.lower() or 'icon' in img_url.lower():
                            continue
                        
                        success, result = download_image(img_url, role_dir, role["chinese"])
                        if success:
                            collected += 1
                            logger.info(f"✓ {role['chinese']} 下载成功: {result}")
                        time.sleep(DELAY)
                except Exception as e:
                    logger.error(f"处理图片错误: {e}")
                    continue

            # 关闭浏览器
            driver.quit()

        except Exception as e:
            logger.error(f"采集错误 {role['chinese']} ({keyword}): {e}")
            continue

    return collected

def collect_from_existing_urls(role, role_dir, max_images=50):
    """从现有的URL文件中采集图片"""
    collected = 0
    
    # 查找相关的URL文件
    url_files = [
        Path(BASE_DIR) / "data" / "img_url" / f"{role['chinese']}_img.txt",
        Path(BASE_DIR) / "data" / "img_url" / "arona_img.txt",  # 通用URL文件
    ]
    
    for url_file in url_files:
        if not url_file.exists():
            continue
        
        try:
            with open(url_file, 'r', encoding='utf-8') as f:
                urls = f.readlines()
            
            for url in urls:
                if collected >= max_images:
                    break
                
                url = url.strip()
                if not url:
                    continue
                
                # 过滤掉无效URL和图标
                if url.endswith('.svg') or 'icon' in url.lower() or 'logo' in url.lower():
                    continue
                
                # 只处理jpg图片
                if not url.endswith('.jpg'):
                    continue
                
                success, result = download_image(url, role_dir, role["chinese"])
                if success:
                    collected += 1
                    logger.info(f"✓ {role['chinese']} 下载成功: {result}")
                time.sleep(DELAY)
                
        except Exception as e:
            logger.error(f"读取URL文件错误 {url_file}: {e}")
            continue
    
    return collected

def collect_role_images(role):
    """采集单个角色的图片"""
    role_dir = create_role_directory(role["chinese"], role["source"])

    existing_count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if existing_count >= MAX_IMAGES_PER_ROLE:
        logger.info(f"⏭ {role['chinese']} 已采集 ({existing_count}/{MAX_IMAGES_PER_ROLE})")
        return existing_count

    logger.info(f"🎯 开始采集: {role['chinese']} ({role['source']})")
    logger.info(f"   关键词: {get_search_keywords(role)}")

    total_collected = 0
    
    # 首先从现有URL文件中采集
    collected = collect_from_existing_urls(role, role_dir, max_images=MAX_IMAGES_PER_ROLE - total_collected)
    total_collected += collected
    
    if total_collected < MAX_IMAGES_PER_ROLE:
        # 如果现有URL不够，再尝试从sd.vv50.de采集
        for keyword in get_search_keywords(role):
            collected = collect_from_sdvv50(role, role_dir, max_images=MAX_IMAGES_PER_ROLE - total_collected)
            total_collected += collected
            if total_collected >= MAX_IMAGES_PER_ROLE:
                break

    final_count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    logger.info(f"✅ {role['chinese']} 采集完成 ({final_count}/{MAX_IMAGES_PER_ROLE})")

    return final_count

def main():
    """主函数"""
    print("=" * 60)
    print("🎯 萝莉角色数据采集系统")
    print("=" * 60)

    if not LOLI_ROLE_FILE.exists():
        logger.error(f"角色文件不存在: {LOLI_ROLE_FILE}")
        return

    roles = parse_loli_role_file(LOLI_ROLE_FILE)
    logger.info(f"📋 加载了 {len(roles)} 个角色")

    print()
    print("角色列表:")
    for i, role in enumerate(roles, 1):
        print(f"  {i}. {role['chinese']} ({role['source']}) - {role.get('english', '')}")
    print()

    print("开始采集数据...")
    print()

    success_count = 0
    fail_count = 0

    for i, role in enumerate(roles, 1):
        logger.info(f"[{i}/{len(roles)}] 正在处理: {role['chinese']}")
        try:
            count = collect_role_images(role)
            if count > 0:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"处理角色失败 {role['chinese']}: {e}")
            fail_count += 1

        time.sleep(1)

    print()
    print("=" * 60)
    print("📊 采集统计")
    print("=" * 60)
    print(f"  总角色数: {len(roles)}")
    print(f"  成功采集: {success_count}")
    print(f"  采集失败: {fail_count}")
    print(f"  数据目录: {DATA_DIR}")
    print()

    print("各角色数据统计:")
    for role in roles:
        role_dir = DATA_DIR / f"{role['chinese']}_{role['source']}"
        if role_dir.exists():
            count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {role['chinese']}: {count} 张")
    print()

    print("=" * 60)
    print("✅ 数据采集完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
角色图片下载脚本
根据characters目录中的txt文件下载角色图片
参考 spider_image_system 项目设计
"""
import os
import json
import requests
import time
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import cv2
import psutil
from PIL import Image
import hashlib

# 加载配置
with open('config.json', 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)

# 配置参数
MAX_IMAGES = CONFIG['download']['max_images']
DOWNLOAD_RETRIES = CONFIG['download']['retries']
DOWNLOAD_DELAY = CONFIG['download']['delay']
ANIME_DELAY = CONFIG['download']['anime_delay']
PIXIV_MAX_IMAGES = CONFIG['search']['pixiv_max_images']
PIXIV_RETRIES = CONFIG['search']['pixiv_retries']
BING_RETRIES = CONFIG['search']['bing_retries']
MAX_WORKERS = CONFIG['threading']['max_workers']
CHARACTERS_DIR = CONFIG['paths']['characters_dir']
OUTPUT_DIR = CONFIG['paths']['output_dir']
LOG_DIR = CONFIG['paths']['log_dir']
HEADERS = CONFIG['headers']

# 配置日志
logger.add(f"{LOG_DIR}/download_character_images_{time}.log", rotation="10 MB", encoding="utf-8")



# 配置 Selenium
CHROME_OPTIONS = Options()
CHROME_OPTIONS.add_argument("--headless")
CHROME_OPTIONS.add_argument("--no-sandbox")
CHROME_OPTIONS.add_argument("--disable-dev-shm-usage")
CHROME_OPTIONS.add_argument(f"user-agent={HEADERS['User-Agent']}")

def get_system_stats():
    """获取系统性能统计信息"""
    try:
        # CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / (1024 * 1024 * 1024)  # 转换为 GB
        memory_total = memory.total / (1024 * 1024 * 1024)  # 转换为 GB
        
        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used = disk.used / (1024 * 1024 * 1024)  # 转换为 GB
        disk_total = disk.total / (1024 * 1024 * 1024)  # 转换为 GB
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_used": round(memory_used, 2),
            "memory_total": round(memory_total, 2),
            "disk_percent": disk_percent,
            "disk_used": round(disk_used, 2),
            "disk_total": round(disk_total, 2)
        }
    except Exception as e:
        logger.error(f"获取系统性能统计信息失败: {e}")
        return {}

def log_system_stats():
    """记录系统性能统计信息"""
    stats = get_system_stats()
    if stats:
        logger.info(f"系统性能: CPU={stats['cpu_percent']}% | 内存={stats['memory_used']}/{stats['memory_total']}GB ({stats['memory_percent']}%) | 磁盘={stats['disk_used']}/{stats['disk_total']}GB ({stats['disk_percent']}%)")

def get_image_hash(image_path):
    """计算图片的哈希值"""
    try:
        img = Image.open(image_path)
        # 调整图片大小以统一尺寸
        img = img.resize((16, 16), Image.Resampling.LANCZOS)
        # 转换为灰度图
        img = img.convert('L')
        # 计算平均灰度
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        # 生成哈希值
        hash_str = ''.join(['1' if pixel > avg else '0' for pixel in pixels])
        # 转换为十六进制
        return hashlib.md5(hash_str.encode()).hexdigest()
    except Exception as e:
        logger.error(f"计算图片哈希值失败: {e}")
        return None

def detect_face(image_path):
    """检测图片中的人脸"""
    try:
        # 加载预训练的人脸检测模型
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # 如果检测到人脸，返回 True
        return len(faces) > 0
    except Exception as e:
        logger.error(f"面部检测失败: {e}")
        return False

def load_downloaded_urls(character_dir):
    """加载已下载的URL列表"""
    status_file = os.path.join(character_dir, 'download_status.json')
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except Exception as e:
            logger.error(f"加载下载状态文件失败: {e}")
    return set()

def save_downloaded_urls(character_dir, downloaded_urls):
    """保存已下载的URL列表"""
    status_file = os.path.join(character_dir, 'download_status.json')
    try:
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(list(downloaded_urls), f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存下载状态文件失败: {e}")

def download_image(url, save_path, retries=DOWNLOAD_RETRIES):
    """下载单个图片"""
    # 检查是否已经下载过
    character_dir = os.path.dirname(save_path)
    downloaded_urls = load_downloaded_urls(character_dir)
    
    if url in downloaded_urls:
        logger.info(f"URL 已下载过，跳过: {url}")
        return False
    
    for i in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                # 获取文件类型
                content_type = response.headers.get('Content-Type', '')
                extension = '.jpg'
                if 'png' in content_type:
                    extension = '.png'
                elif 'gif' in content_type:
                    extension = '.gif'
                elif 'webp' in content_type:
                    extension = '.webp'
                elif 'svg' in content_type:
                    # 跳过 SVG 文件，因为它们不是真正的图片
                    return False
                
                # 使用正确的扩展名
                save_path = save_path.rsplit('.', 1)[0] + extension
                
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                
                # 检测人脸
                if not detect_face(save_path):
                    # 如果没有检测到人脸，删除图片
                    os.remove(save_path)
                    return False
                
                # 计算图片哈希值
                img_hash = get_image_hash(save_path)
                if img_hash:
                    # 检查是否已经存在相同的图片
                    existing_hashes = set()
                    
                    for file in os.listdir(character_dir):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            file_path = os.path.join(character_dir, file)
                            if file_path != save_path:
                                existing_hash = get_image_hash(file_path)
                                if existing_hash == img_hash:
                                    # 发现重复图片，删除新下载的
                                    os.remove(save_path)
                                    logger.info(f"发现重复图片，已删除: {save_path}")
                                    return False
                
                # 记录已下载的URL
                downloaded_urls.add(url)
                save_downloaded_urls(character_dir, downloaded_urls)
                
                return True
        except Exception:
            time.sleep(1)
    return False

def configure_chrome_options():
    """配置 Chrome 选项，参考 spider_image_system 项目"""
    chrome_options = Options()
    chrome_options.add_argument(f"user-agent={HEADERS['User-Agent']}")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--disable-javascript")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    return chrome_options


def search_images_sdvv50(character_name, anime_name, max_images=MAX_IMAGES, retries=BING_RETRIES):
    """从 sd.vv50.de 搜索角色图片（使用 Selenium，参考 spider_image_system 实现）"""
    images = []
    search_query = f"{character_name}"
    search_url = f"https://sd.vv50.de/search?q={search_query}"
    
    logger.info(f"开始从 sd.vv50.de 搜索: {search_query}")
    
    for retry in range(retries):
        driver = None
        try:
            # 配置 Chrome 选项
            chrome_options = configure_chrome_options()
            
            # 启动浏览器
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)
            
            # 访问搜索页面
            driver.get(search_url)
            logger.debug(f"已访问: {search_url}")
            
            # 等待页面加载完成
            time.sleep(5)
            
            # 检查页面标题
            logger.debug(f"页面标题: {driver.title}")
            
            # 滚动页面以加载更多图片（参考 spider_image_system 的 slider_page_down）
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
            max_scroll_attempts = 5
            
            while scroll_attempts < max_scroll_attempts:
                # 滚动到页面底部
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
                
                # 检查是否有新内容加载
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                scroll_attempts += 1
                logger.debug(f"滚动页面 {scroll_attempts}/{max_scroll_attempts}")
            
            # 获取页面源代码并解析
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # 查找所有图片元素（参考 spider_image_system 的 save_img_element）
            img_elements = driver.find_elements(By.CSS_SELECTOR, "img")
            logger.debug(f"找到 {len(img_elements)} 个图片元素")
            
            for img in img_elements:
                try:
                    # 尝试从不同属性获取图片URL
                    img_url = img.get_attribute("src") or img.get_attribute("data-src") or img.get_attribute("data-original")
                    
                    if not img_url:
                        continue
                    
                    # 执行 JavaScript 获取完整 URL
                    try:
                        driver.execute_script("return arguments[0].src;", img)
                    except:
                        pass
                    
                    # 确保URL是完整的
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url
                    elif img_url.startswith('/'):
                        img_url = f"https://sd.vv50.de{img_url}"
                    elif not img_url.startswith('http'):
                        continue
                    
                    # 过滤掉小图标和无效图片
                    if any(skip in img_url.lower() for skip in ['thumb', 'icon', 'avatar', 'logo', 'button', 'gif']):
                        continue
                    
                    # 只保留图片文件
                    if not any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        continue
                    
                    if img_url not in images:
                        images.append(img_url)
                        logger.debug(f"找到图片: {img_url}")
                        
                    if len(images) >= max_images:
                        break
                        
                except Exception as e:
                    logger.warning(f"处理图片元素时出错: {e}")
                    continue
            
            # 如果找到图片，跳出重试循环
            if images:
                logger.success(f"成功找到 {len(images)} 张图片")
                break
            else:
                logger.warning("未找到任何图片，尝试重试...")
                
        except Exception as e:
            logger.error(f"sdvv50.de 搜索失败 (尝试 {retry+1}/{retries}): {e}")
            if retry < retries - 1:
                time.sleep(5)
        finally:
            if driver:
                try:
                    driver.quit()
                    logger.debug("浏览器已关闭")
                except Exception as e:
                    logger.warning(f"关闭浏览器时出错: {e}")
    
    logger.info(f"从 sd.vv50.de 搜索到 {len(images)} 张图片")
    return images


def search_images_danbooru(character_name, anime_name, max_images=MAX_IMAGES, retries=BING_RETRIES):
    """从 Danbooru 搜索角色图片"""
    images = []
    search_queries = [
        _get_character_tags(character_name, anime_name),
        [character_name, 'highres'],
        [character_name, 'blue_archive'],
        [character_name, 'solo'],
    ]
    
    for tags in search_queries:
        if images:
            break
        tags_str = '+'.join(tags) if tags else character_name
        search_url = f"https://danbooru.donmai.us/posts.json?tags={urllib.parse.quote(tags_str)}&limit={max_images}"
        
        for retry in range(retries):
            try:
                response = requests.get(search_url, headers=HEADERS, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    for post in data:
                        if 'file_url' in post and post['file_url']:
                            img_url = post['file_url']
                            if not img_url.startswith('http'):
                                img_url = f"https://danbooru.donmai.us{img_url}"
                            if any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                                images.append(img_url)
                                if len(images) >= max_images:
                                    break
                    if images:
                        break
            except Exception as e:
                logger.error(f"Danbooru 搜索失败 (尝试 {retry+1}/{retries}): {e}")
                if retry < retries - 1:
                    time.sleep(3)
    
    logger.info(f"从 Danbooru 搜索到 {len(images)} 张图片")
    return images

def search_images_safebooru(character_name, anime_name, max_images=MAX_IMAGES, retries=BING_RETRIES):
    """从 Safebooru 搜索角色图片"""
    images = []
    search_queries = [
        _get_character_tags(character_name, anime_name),
        [character_name, 'highres'],
        [character_name, 'blue_archive'],
        [character_name, 'solo'],
    ]
    
    for tags in search_queries:
        if images:
            break
        tags_str = '+'.join(tags) if tags else character_name
        search_url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&tags={urllib.parse.quote(tags_str)}&limit={max_images}"
        
        for retry in range(retries):
            try:
                response = requests.get(search_url, headers=HEADERS, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'xml')
                    posts = soup.find_all('post')
                    for post in posts:
                        img_url = post.get('file_url')
                        if img_url and not img_url.startswith('http'):
                            img_url = f"https://safebooru.org{img_url}"
                        if img_url:
                            images.append(img_url)
                            if len(images) >= max_images:
                                break
                    if images:
                        break
            except Exception as e:
                logger.error(f"Safebooru 搜索失败 (尝试 {retry+1}/{retries}): {e}")
                if retry < retries - 1:
                    time.sleep(3)
    
    logger.info(f"从 Safebooru 搜索到 {len(images)} 张图片")
    return images

def _get_character_tags(character_name, anime_name):
    """根据角色名和动漫名获取搜索标签"""
    tags = []
    anime_tag = _get_anime_tag(anime_name)
    if anime_tag:
        tags.append(anime_tag)
    
    # 获取角色的英文标签
    character_tag = _get_character_tag(character_name)
    if character_tag:
        tags.append(character_tag)
    else:
        tags.append(character_name)
    
    tags.append('highres')
    return tags

def _get_anime_tag(anime_name):
    """根据动漫名获取对应的标签"""
    anime_tags = {
        'blda_spider_img_keyword': 'blue_archive',
        '蔚蓝档案': 'blue_archive',
        '原神': 'genshin_impact',
        '崩坏星穹铁道': 'honkai_star_rail',
        '崩坏3': 'honkai_impact_3rd',
        '明日方舟': 'arknights',
        'fate': 'fate_(series)',
        'fate/grand_order': 'fate/grand_order',
        'fgo': 'fate/grand_order',
        '碧蓝航线': 'azur_lane',
        '舰娘': 'azur_lane',
        '偶像大师': 'the_idolmaster',
        'LoveLive': 'love_live',
        '赛马娘': 'umamusume',
        '公主连结': 'princess_connect',
        '公主连接': 'princess_connect',
        '尼尔机械纪元': 'nier:automata',
        '尼尔': 'nier:automata',
        '刀剑神域': 'sword_art_online',
        'SAO': 'sword_art_online',
        '进击的巨人': 'attack_on_titan',
        '巨人': 'attack_on_titan',
        '鬼灭之刃': 'demon_slayer',
        '原神': 'genshin_impact',
        '火影忍者': 'naruto',
        '海贼王': 'one_piece',
        '死神': 'bleach',
        '龙珠': 'dragon_ball',
    }
    return anime_tags.get(anime_name, '')

def _get_character_tag(character_name):
    """根据角色名获取对应的英文标签"""
    character_tags = {
        '阿罗娜': 'arona_(blue_archive)',
        '普拉娜': 'plana_(blue_archive)',
        '日奈': 'hina_(blue_archive)',
        '亚子': 'ako_(blue_archive)',
        '伊织': 'iori_(blue_archive)',
        '千夏': 'chinatsu_(blue_archive)',
        '伊吕波': 'iroha_(blue_archive)',
        '阿露': 'aru_(blue_archive)',
        '睦月': 'mutsuki_(blue_archive)',
        '佳代子': 'kayoko_(blue_archive)',
        '遥香': 'haruka_(blue_archive)',
        '晴奈': 'haruna_(blue_archive)',
        '淳子': 'junko_(blue_archive)',
        '明里': 'akari_(blue_archive)',
        '泉': 'izumi_(blue_archive)',
        '枫香': 'fuuka_(blue_archive)',
        '朱莉': 'juri_(blue_archive)',
        '濑名': 'serina_(blue_archive)',
        '惠': 'megu_(blue_archive)',
        '霞': 'kasumi_(blue_archive)',
        '优香': 'yuuka_(blue_archive)',
        '诺亚': 'noa_(blue_archive)',
        '小雪': 'koyuki_(blue_archive)',
        '尼禄': 'neru_(blue_archive)',
        '明日奈': 'asuna_(blue_archive)',
        '花凛': 'karin_(blue_archive)',
        '朱音': 'ayane_(blue_archive)',
        '时': 'toki_(blue_archive)',
        '歌原': 'utaha_(blue_archive)',
        '响': 'hibiki_(blue_archive)',
        '柯托莉': 'kotori_(blue_archive)',
        '柚子': 'yuzu_(blue_archive)',
        '桃井': 'momoi_(blue_archive)',
        '绿': 'midori_(blue_archive)',
        '爱丽丝': 'aris_(blue_archive)',
        '千寻': 'chihiro_(blue_archive)',
        '真纪': 'maki_(blue_archive)',
        '晴': 'hare_(blue_archive)',
        '小玉': 'kotama_(blue_archive)',
        '日鞠': 'himari_(blue_archive)',
        '艾米': 'emi_(blue_archive)',
        '菫': 'sumire_(blue_archive)',
        '未花': 'mika_(blue_archive)',
        '渚': 'nagisa_(blue_archive)',
        '鹤城': 'tsurugi_(blue_archive)',
    }
    return character_tags.get(character_name, '')

def search_images_bing(character_name, anime_name, max_images=MAX_IMAGES, retries=BING_RETRIES):
    """从 Bing 图片搜索角色图片（使用 Selenium）"""
    images = []
    search_query = f"{character_name} {anime_name} anime character"
    search_url = f"https://www.bing.com/images/search?q={urllib.parse.quote(search_query)}"
    
    logger.info(f"开始从 Bing 搜索: {search_query}")
    
    for retry in range(retries):
        driver = None
        try:
            # 配置 Chrome 选项
            chrome_options = Options()
            chrome_options.add_argument(f"user-agent={HEADERS['User-Agent']}")
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # 启动浏览器
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)
            
            # 访问搜索页面
            driver.get(search_url)
            logger.debug(f"已访问 Bing 图片搜索: {search_url}")
            
            # 等待页面加载
            time.sleep(5)
            
            # 滚动页面以加载更多图片
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
            max_scroll_attempts = 3
            
            while scroll_attempts < max_scroll_attempts and len(images) < max_images:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                scroll_attempts += 1
            
            # 查找所有图片元素
            img_elements = driver.find_elements(By.CSS_SELECTOR, "img.mimg")
            logger.debug(f"找到 {len(img_elements)} 个图片元素")
            
            for img in img_elements:
                try:
                    img_url = img.get_attribute("src") or img.get_attribute("data-src")
                    
                    if not img_url:
                        continue
                    
                    # 过滤掉 base64 图片和小图标
                    if img_url.startswith('data:'):
                        continue
                    
                    # 只保留图片文件
                    if not any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        # 尝试获取完整 URL
                        if 'bing.net' in img_url or 'microsoft.com' in img_url:
                            pass  # 保留 Bing 的图片 URL
                        else:
                            continue
                    
                    if img_url not in images:
                        images.append(img_url)
                        logger.debug(f"找到图片: {img_url[:100]}...")
                        
                    if len(images) >= max_images:
                        break
                        
                except Exception as e:
                    logger.warning(f"处理图片元素时出错: {e}")
                    continue
            
            # 如果找到图片，跳出重试循环
            if images:
                logger.success(f"成功从 Bing 找到 {len(images)} 张图片")
                break
            else:
                logger.warning("未找到任何图片，尝试重试...")
                
        except Exception as e:
            logger.error(f"Bing 搜索失败 (尝试 {retry+1}/{retries}): {e}")
            if retry < retries - 1:
                time.sleep(5)
        finally:
            if driver:
                try:
                    driver.quit()
                    logger.debug("浏览器已关闭")
                except Exception as e:
                    logger.warning(f"关闭浏览器时出错: {e}")
    
    logger.info(f"从 Bing 搜索到 {len(images)} 张图片")
    return images


def search_images(character_name, anime_name, max_images=MAX_IMAGES, retries=BING_RETRIES):
    """搜索角色图片"""
    images = []
    
    # 优先使用 Bing 图片搜索
    bing_images = search_images_bing(character_name, anime_name, max_images=max_images, retries=retries)
    images.extend(bing_images)
    
    # 如果 Bing 搜索结果不足，尝试 sdvv50.de
    if len(images) < max_images:
        sdvv50_images = search_images_sdvv50(character_name, anime_name, max_images=max_images - len(images), retries=retries)
        images.extend(sdvv50_images)
    
    # 如果还不够，尝试 Danbooru
    if len(images) < max_images:
        danbooru_images = search_images_danbooru(character_name, anime_name, max_images=max_images - len(images), retries=retries)
        images.extend(danbooru_images)
    
    # 最后尝试 Safebooru
    if len(images) < max_images:
        safebooru_images = search_images_safebooru(character_name, anime_name, max_images=max_images - len(images), retries=retries)
        images.extend(safebooru_images)
    
    seen = set()
    unique_images = []
    for img in images:
        if img not in seen:
            seen.add(img)
            unique_images.append(img)
    
    logger.info(f"总共找到 {len(unique_images)} 张图片")
    return unique_images[:max_images]

def process_character(character_name, anime_name, output_dir):
    """处理单个角色"""
    safe_character_name = re.sub(r'[\\/*?:"<>|]', '_', character_name)
    character_dir = os.path.join(output_dir, f"{anime_name}_{safe_character_name}")
    
    if not os.path.exists(character_dir):
        os.makedirs(character_dir)
    
    # 检查是否已有足够的图片
    existing_images = [f for f in os.listdir(character_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if len(existing_images) >= MAX_IMAGES:
        logger.info(f"{anime_name}_{character_name} 已有足够的图片，跳过")
        return
    
    logger.info(f"下载 {anime_name}_{character_name} 的图片...")
    
    # 搜索图片
    image_urls = search_images(character_name, anime_name)
    
    # 下载图片
    downloaded = 0
    for i, url in enumerate(image_urls):
        if downloaded >= 50:
            break
        
        save_path = os.path.join(character_dir, f"{anime_name}_{safe_character_name}_{i:04d}.jpg")
        if download_image(url, save_path):
                downloaded += 1
                logger.info(f"  下载成功 {downloaded}/50")
        time.sleep(DOWNLOAD_DELAY)  # 礼貌延时
    
    logger.info(f"{anime_name}_{character_name} 下载完成，共 {downloaded} 张图片")

def main():
    # 记录初始系统性能
    log_system_stats()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, OUTPUT_DIR)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 只处理指定的文件
    target_file = 'blda_spider_img_keyword.txt'
    txt_path = os.path.join(base_dir, target_file)
    
    if not os.path.exists(txt_path):
        logger.error(f"文件不存在: {target_file}")
        return
    
    # 从文件名中提取作品名称
    anime_name = target_file.replace('.txt', '')
    
    logger.info(f"\n=== 处理 {anime_name} ===")
    
    # 读取角色列表
    with open(txt_path, 'r', encoding='utf-8') as f:
        characters = [line.strip() for line in f if line.strip()]
    
    if not characters:
        logger.warning(f"{target_file} 中没有角色信息")
        return
    
    logger.info(f"发现 {len(characters)} 个角色")
    
    # 并发处理角色
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for character in characters:
            future = executor.submit(process_character, character, anime_name, output_dir)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"处理失败: {e}")
    
    time.sleep(ANIME_DELAY)  # 每个作品之间的延时
    
    # 记录结束时系统性能
    log_system_stats()

if __name__ == "__main__":
    main()

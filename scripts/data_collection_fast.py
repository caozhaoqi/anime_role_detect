#!/usr/bin/env python3
"""
快速数据采集脚本 - 优化版本
从网络上快速获取更多与现有角色相关的图像数据
"""
import os
import requests
import time
import random
import logging
import threading
from urllib.parse import quote
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_collection_fast')


class FastImageCollector:
    """快速图像采集器"""
    
    def __init__(self, output_dir, max_images_per_character=100, max_workers=5):
        """初始化
        
        Args:
            output_dir: 输出目录
            max_images_per_character: 每个角色最大采集图像数
            max_workers: 并发下载线程数
        """
        self.output_dir = output_dir
        self.max_images_per_character = max_images_per_character
        self.max_workers = max_workers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        }
    
    def search_images(self, query, num_images=100):
        """
        搜索图像 - 优化版本
        
        Args:
            query: 搜索关键词
            num_images: 搜索图像数量
            
        Returns:
            图像URL列表
        """
        image_urls = []
        
        # 先尝试从专门的二次元网站采集高质量图像
        anime_site_urls = self.search_anime_sites(query, num_images=num_images)
        image_urls.extend(anime_site_urls)
        
        # 如果还需要更多图像，使用Bing图像搜索
        if len(image_urls) < num_images:
            search_engines = [
                lambda q: f"https://www.bing.com/images/search?q={quote(q)}&count=100",
            ]
            
            for engine_url in search_engines:
                if len(image_urls) >= num_images:
                    break
                
                try:
                    url = engine_url(query)
                    
                    response = requests.get(url, headers=self.headers, timeout=8)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 提取图像URL
                    for img in soup.find_all('img'):
                        if len(image_urls) >= num_images:
                            break
                        
                        # 尝试不同的属性获取图像URL
                        img_url = None
                        for attr in ['src', 'data-src', 'data-original', 'data-image-src']:
                            if attr in img.attrs:
                                img_url = img[attr]
                                break
                        
                        if img_url and img_url.startswith('http') and not img_url.startswith('data:'):
                            # 确保URL是完整的
                            if not img_url.startswith('http://') and not img_url.startswith('https://'):
                                continue
                            image_urls.append(img_url)
                
                except Exception as e:
                    logger.error(f"搜索图像出错 {engine_url(query)}: {str(e)}")
                    continue
        
        # 去重
        image_urls = list(set(image_urls))
        return image_urls[:num_images]
    
    def search_anime_sites(self, query, num_images=50):
        """
        从专门的二次元网站搜索高质量图像
        
        Args:
            query: 搜索关键词
            num_images: 搜索图像数量
            
        Returns:
            高质量图像URL列表
        """
        image_urls = []
        
        # 二次元专门网站 - 只保留可访问的
        anime_sites = [
            # Gelbooru - 动漫图像
            lambda q: f"https://gelbooru.com/index.php?page=post&s=list&tags={quote(q)}",
            # Yande.re - 高质量动漫图像
            lambda q: f"https://yande.re/post?tags={quote(q)}"
        ]
        
        for site_url in anime_sites:
            if len(image_urls) >= num_images:
                break
            
            try:
                url = site_url(query)
                logger.debug(f"从二次元网站搜索: {url}")
                
                response = requests.get(url, headers=self.headers, timeout=8)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 提取图像URL - 适配不同网站
                for img in soup.find_all('img'):
                    if len(image_urls) >= num_images:
                        break
                    
                    # 尝试不同的属性获取图像URL
                    img_url = None
                    for attr in ['src', 'data-src', 'data-original', 'data-file-url', 'data-large-file']:
                        if attr in img.attrs:
                            img_url = img[attr]
                            break
                    
                    if img_url:
                        # 处理相对路径
                        if img_url.startswith('//'):
                            img_url = 'https:' + img_url
                        elif not img_url.startswith('http'):
                            continue
                        
                        # 过滤掉小缩略图
                        if 'thumbnail' in img_url.lower() or 'thumb' in img_url.lower():
                            continue
                        
                        # 确保是图像文件
                        if any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                            image_urls.append(img_url)
            
            except Exception as e:
                logger.debug(f"从二次元网站搜索出错 {site_url(query)}: {str(e)}")
                continue
        
        return image_urls[:num_images]
    
    def download_image(self, url, save_path):
        """
        下载图像 - 优化版本
        
        Args:
            url: 图像URL
            save_path: 保存路径
            
        Returns:
            是否下载成功
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            # 快速验证图像
            from PIL import Image
            img = Image.open(save_path)
            img.verify()
            
            return True
        
        except Exception as e:
            if os.path.exists(save_path):
                os.remove(save_path)
            return False
    
    def download_image_batch(self, urls, char_dir, character, existing_images):
        """
        批量下载图像 - 使用并发
        
        Args:
            urls: 图像URL列表
            char_dir: 角色目录
            character: 角色名
            existing_images: 现有图像数量
            
        Returns:
            成功下载的图像数量
        """
        collected = 0
        collected_lock = threading.Lock()
        
        def download_single(url, idx):
            nonlocal collected
            try:
                img_name = f"{character}_{existing_images + idx}_{int(time.time())}.jpg"
                save_path = os.path.join(char_dir, img_name)
                
                if self.download_image(url, save_path):
                    with collected_lock:
                        collected += 1
                        return True
            except:
                pass
            return False
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(download_single, url, i) for i, url in enumerate(urls)]
            for future in as_completed(futures):
                pass
        
        return collected
    
    def collect_images(self, characters):
        """
        为多个角色采集图像 - 优化版本
        
        Args:
            characters: 角色列表
        """
        for character in tqdm(characters, desc="采集角色图像"):
            # 创建角色目录
            char_dir = os.path.join(self.output_dir, character)
            os.makedirs(char_dir, exist_ok=True)
            
            # 计算现有图像数量
            existing_images = len([f for f in os.listdir(char_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            # 需要采集的图像数量
            need_images = max(0, self.max_images_per_character - existing_images)
            
            if need_images <= 0:
                logger.info(f"角色 {character} 已有足够图像 ({existing_images} 张)")
                continue
            
            logger.info(f"开始采集角色 {character} 的图像，需要 {need_images} 张")
            
            # 简化搜索关键词
            search_queries = [
                character,
                f"{character} anime",
                f"{character} character",
                f"{character} artwork",
            ]
            
            collected = 0
            for query in search_queries:
                if collected >= need_images:
                    break
                
                # 搜索图像
                image_urls = self.search_images(query, num_images=need_images - collected)
                
                # 批量下载
                batch_collected = self.download_image_batch(image_urls, char_dir, character, existing_images + collected)
                collected += batch_collected
                
                logger.info(f"搜索关键词: {query}, 成功下载 {batch_collected} 张图像，总计 {collected}/{need_images}")
                
                # 极短延迟
                time.sleep(random.uniform(0.3, 0.8))
            
            logger.info(f"角色 {character} 采集完成，新增 {collected} 张图像，总计 {existing_images + collected} 张")
            
            # 角色之间短延迟
            time.sleep(random.uniform(0.5, 1.0))


def main():
    """主函数"""
    # 输出目录
    output_dir = 'data/all_characters'
    os.makedirs(output_dir, exist_ok=True)
    
    # 扩展角色列表
    characters = [
        # 原神角色
        '原神_琴', '原神_空', '原神_荧', '原神_丽莎', '原神_凯亚', '原神_安柏', 
        '原神_芭芭拉', '原神_迪卢克', '原神_雷泽', '原神_温迪', '原神_可莉', 
        '原神_班尼特', '原神_诺艾尔', '原神_菲谢尔', '原神_砂糖', '原神_莫娜', 
        '原神_迪奥娜', '原神_阿贝多', '原神_甘雨', '原神_魈', '原神_胡桃', 
        '原神_钟离', '原神_烟绯', '原神_罗莎莉亚', '原神_优菈', '原神_阿瑠', 
        
        # 鬼灭之刃角色
        '鬼灭之刃_灶门炭治郎', '鬼灭之刃_灶门祢豆子', '鬼灭之刃_我妻善逸', 
        '鬼灭之刃_嘴平伊之助', '鬼灭之刃_富冈义勇', '鬼灭之刃_蝴蝶忍', 
        
        # 进击的巨人角色
        '进击的巨人_艾伦耶格尔', '进击的巨人_三笠阿克曼', '进击的巨人_阿尔敏阿诺德', 
        
        # 海贼王角色
        '海贼王_蒙奇D路飞', '海贼王_罗罗诺亚索隆', '海贼王_娜美', 
        
        # 火影忍者角色
        '火影忍者_漩涡鸣人', '火影忍者_宇智波佐助', '火影忍者_春野樱', 
        
        # 东京复仇者角色
        '东京复仇者_花垣武道', '东京复仇者_佐野万次郎', '东京复仇者_龙宫寺坚', 
        
        # 我的英雄学院角色
        '我的英雄学院_绿谷出久', '我的英雄学院_爆豪胜己', '我的英雄学院_轰焦冻', 
        
        # 冰菓角色
        '冰菓_折木奉太郎', '冰菓_千反田爱瑠', 
        
        # 龙与虎角色
        '龙与虎_逢坂大河', '龙与虎_高须龙儿', 
        
        # 青春猪头少年不会梦到兔女郎学姐角色
        '青春猪头少年_樱岛麻衣', '青春猪头少年_梓川咲太', 
        
        # 从零开始的异世界生活角色
        'Re0_艾米莉亚', 'Re0_拉姆', 'Re0_雷姆', 'Re0_菜月昴', 
        
        # 约战角色
        '约会大作战_时崎狂三', '约会大作战_五河琴里', '约会大作战_夜刀神十香', 
        
        # 其他热门角色
        '初音未来', ' Kaguya Shinomiya', ' Chika Fujiwara', 
        'Rikka Takanashi', 'Taiga Aisaka', 'Misaka Mikoto'
    ]
    
    logger.info(f"准备采集 {len(characters)} 个角色的图像")
    logger.info(f"角色列表: {characters}")
    
    # 创建快速采集器
    collector = FastImageCollector(
        output_dir=output_dir,
        max_images_per_character=200,
        max_workers=8  # 并发下载线程数
    )
    
    # 开始采集
    logger.info('开始快速采集图像数据...')
    collector.collect_images(characters)
    logger.info('图像数据采集完成！')


if __name__ == "__main__":
    main()

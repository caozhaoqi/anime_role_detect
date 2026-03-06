#!/usr/bin/env python3
"""
数据采集脚本
从网络上获取更多与现有角色相关的图像数据
"""
import os
import requests
import time
import random
import logging
from urllib.parse import quote
from bs4 import BeautifulSoup
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_collection')


class ImageCollector:
    """图像采集器"""
    
    def __init__(self, output_dir, max_images_per_character=100):
        """初始化
        
        Args:
            output_dir: 输出目录
            max_images_per_character: 每个角色最大采集图像数
        """
        self.output_dir = output_dir
        self.max_images_per_character = max_images_per_character
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        }
    
    def search_images(self, query, num_images=100):
        """
        搜索图像
        
        Args:
            query: 搜索关键词
            num_images: 搜索图像数量
            
        Returns:
            图像URL列表
        """
        image_urls = []
        
        # 使用多个搜索引擎和不同的搜索策略
        search_engines = [
            # Bing图像搜索
            lambda q: f"https://www.bing.com/images/search?q={quote(q)}&count=100",
            # Google图像搜索
            lambda q: f"https://www.google.com/search?q={quote(q)}&tbm=isch",
            # DuckDuckGo图像搜索
            lambda q: f"https://duckduckgo.com/?q={quote(q)}&iax=images&ia=images",
            # Yahoo图像搜索
            lambda q: f"https://images.search.yahoo.com/search/images?p={quote(q)}"
        ]
        
        for engine_url in search_engines:
            if len(image_urls) >= num_images:
                break
            
            try:
                url = engine_url(query)
                logger.info(f"使用搜索引擎: {url}")
                
                response = requests.get(url, headers=self.headers, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 提取图像URL
                for img in soup.find_all('img'):
                    if len(image_urls) >= num_images:
                        break
                    
                    # 尝试不同的属性获取图像URL
                    img_url = None
                    for attr in ['src', 'data-src', 'data-original', 'data-image-src', 'lazy-src']:
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
                # 继续尝试下一个搜索引擎
                continue
        
        # 去重
        image_urls = list(set(image_urls))
        return image_urls[:num_images]
    
    def download_image(self, url, save_path):
        """
        下载图像
        
        Args:
            url: 图像URL
            save_path: 保存路径
            
        Returns:
            是否下载成功
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            # 验证图像是否有效
            from PIL import Image
            img = Image.open(save_path)
            img.verify()
            
            return True
        
        except Exception as e:
            logger.error(f"下载图像出错 {url}: {str(e)}")
            # 删除无效文件
            if os.path.exists(save_path):
                os.remove(save_path)
            return False
    
    def collect_images(self, characters):
        """
        为多个角色采集图像
        
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
            
            # 生成多样化的搜索关键词
            search_queries = [
                character,  # 基础关键词
                f"{character} anime",  # 英文关键词
                f"{character} character",  # 角色关键词
                f"{character} artwork",  #  artwork关键词
                f"{character} fanart",  # 同人关键词
                f"{character} illustration",  # 插画关键词
                f"{character} official art",  # 官方 artwork
                f"{character} HD",  # 高清图像
                f"{character} wallpaper",  # 壁纸
                f"{character} cosplay"  #  cosplay
            ]
            
            # 添加中文关键词（如果角色名不是纯英文）
            if any('一' <= c <= '鿿' for c in character):
                search_queries.extend([
                    f"{character} 动漫",
                    f"{character} 二次元",
                    f"{character} 同人",
                    f"{character} 插画",
                    f"{character} 官方",
                    f"{character} 高清",
                    f"{character} 壁纸",
                    f"{character}  cosplay"
                ])
            
            collected = 0
            for query in search_queries:
                if collected >= need_images:
                    break
                
                logger.info(f"搜索关键词: {query}")
                
                # 搜索图像
                image_urls = self.search_images(query, num_images=need_images - collected)
                
                # 下载图像
                for img_url in image_urls:
                    if collected >= need_images:
                        break
                    
                    # 生成文件名
                    img_name = f"{character}_{collected + existing_images}_{int(time.time())}.jpg"
                    save_path = os.path.join(char_dir, img_name)
                    
                    # 下载图像
                    if self.download_image(img_url, save_path):
                        collected += 1
                        logger.info(f"成功下载 {character} 的图像 {collected}/{need_images}")
                        
                        # 智能延迟，避免被封禁
                        # 随着下载量增加，延迟时间也增加
                        delay = random.uniform(0.8, 2.5) + (collected * 0.1)
                        time.sleep(min(delay, 5.0))  # 最大延迟5秒
            
            # 如果还需要更多图像，尝试使用不同的搜索策略
            if collected < need_images:
                logger.info(f"尝试使用更具体的搜索策略为 {character} 采集更多图像")
                
                # 使用特定的搜索策略
                specific_queries = [
                    # 添加数字后缀，获取不同的搜索结果
                    f"{character} 1",
                    f"{character} 2",
                    f"{character} 3",
                    # 添加时间限定
                    f"{character} 2025",
                    f"{character} 2024",
                    # 添加风格限定
                    f"{character} digital art",
                    f"{character} traditional art",
                    f"{character} chibi",
                    f"{character} realistic"
                ]
                
                for query in specific_queries:
                    if collected >= need_images:
                        break
                    
                    logger.info(f"使用特定搜索策略: {query}")
                    
                    # 搜索图像
                    image_urls = self.search_images(query, num_images=need_images - collected)
                    
                    # 下载图像
                    for img_url in image_urls:
                        if collected >= need_images:
                            break
                        
                        # 生成文件名
                        img_name = f"{character}_{collected + existing_images}_{int(time.time())}_special.jpg"
                        save_path = os.path.join(char_dir, img_name)
                        
                        # 下载图像
                        if self.download_image(img_url, save_path):
                            collected += 1
                            logger.info(f"成功下载 {character} 的图像 {collected}/{need_images}")
                            
                            # 增加延迟，避免被封禁
                            time.sleep(random.uniform(1.5, 3.0))
            
            logger.info(f"角色 {character} 采集完成，新增 {collected} 张图像，总计 {existing_images + collected} 张")
            
            # 角色之间增加更长的延迟，避免被封禁
            time.sleep(random.uniform(3.0, 5.0))


def main():
    """主函数"""
    # 输出目录
    output_dir = 'data/all_characters'
    os.makedirs(output_dir, exist_ok=True)
    
    # 扩展角色列表，包括更多动漫和游戏角色
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
    
    # 创建采集器
    collector = ImageCollector(
        output_dir=output_dir,
        max_images_per_character=200  # 每个角色采集200张图像，增加数据量
    )
    
    # 开始采集
    logger.info('开始采集图像数据...')
    collector.collect_images(characters)
    logger.info('图像数据采集完成！')


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化数据集扩充脚本

实现从多个来源自动收集更多角色和场景的训练数据
"""

import os
import argparse
import requests
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_expansion')

class AutomatedDataExpander:
    def __init__(self, output_dir='data/train', max_workers=10):
        """
        初始化自动化数据扩充器
        
        Args:
            output_dir: 输出目录
            max_workers: 最大并发数
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 支持的数据源
        self.data_sources = {
            'danbooru': self._fetch_from_danbooru,
            'gelbooru': self._fetch_from_gelbooru,
            'safebooru': self._fetch_from_safebooru
        }
        
        # 预设角色列表
        self.preset_characters = [
            # 原神角色
            {'name': '温迪', 'series': 'genshin_impact', 'tags': ['venti_(genshin_impact)', 'genshin_impact']},
            {'name': '迪卢克', 'series': 'genshin_impact', 'tags': ['diluc_(genshin_impact)', 'genshin_impact']},
            {'name': '刻晴', 'series': 'genshin_impact', 'tags': ['keqing_(genshin_impact)', 'genshin_impact']},
            {'name': '甘雨', 'series': 'genshin_impact', 'tags': ['ganyu_(genshin_impact)', 'genshin_impact']},
            {'name': '胡桃', 'series': 'genshin_impact', 'tags': ['hu_tao_(genshin_impact)', 'genshin_impact']},
            
            # 崩坏：星穹铁道角色
            {'name': '三月七', 'series': 'honkai_star_rail', 'tags': ['march_7th', 'honkai:_star_rail']},
            {'name': '丹恒', 'series': 'honkai_star_rail', 'tags': ['dan_heng', 'honkai:_star_rail']},
            {'name': '姬子', 'series': 'honkai_star_rail', 'tags': ['himeko', 'honkai:_star_rail']},
            {'name': '瓦尔特', 'series': 'honkai_star_rail', 'tags': ['welt', 'honkai:_star_rail']},
            
            # 崩坏3角色
            {'name': '琪亚娜', 'series': 'honkai_impact_3', 'tags': ['kiana_kaslana', 'honkai_impact_3rd']},
            {'name': '芽衣', 'series': 'honkai_impact_3', 'tags': ['raiden_mei', 'honkai_impact_3rd']},
            {'name': '布洛妮娅', 'series': 'honkai_impact_3', 'tags': ['bronya_zaychik', 'honkai_impact_3rd']},
            {'name': '德丽莎', 'series': 'honkai_impact_3', 'tags': ['theresa_apocalypse', 'honkai_impact_3rd']},
            
            # 其他热门动漫角色
            {'name': '鸣人', 'series': 'naruto', 'tags': ['uzumaki_naruto', 'naruto']},
            {'name': '佐助', 'series': 'naruto', 'tags': ['uchiha_sasuke', 'naruto']},
            {'name': '路飞', 'series': 'one_piece', 'tags': ['monkey_d._luffy', 'one_piece']},
            {'name': '索隆', 'series': 'one_piece', 'tags': ['roronoa_zoro', 'one_piece']},
            {'name': '炭治郎', 'series': 'demon_slayer', 'tags': ['tanjiro_kamado', 'kimetsu_no_yaiba']},
            {'name': '祢豆子', 'series': 'demon_slayer', 'tags': ['nezuko_kamado', 'kimetsu_no_yaiba']},
            
            # 更多热门动漫角色
            {'name': '孙悟空', 'series': 'dragon_ball', 'tags': ['son_goku', 'dragon_ball']},
            {'name': '贝吉塔', 'series': 'dragon_ball', 'tags': ['vegeta', 'dragon_ball']},
            {'name': '艾伦', 'series': 'attack_on_titan', 'tags': ['eren_yeager', 'shingeki_no_kyojin']},
            {'name': '三笠', 'series': 'attack_on_titan', 'tags': ['mikasa_ackerman', 'shingeki_no_kyojin']},
            {'name': '金木研', 'series': 'tokyo_ghoul', 'tags': ['ken_kaneki', 'tokyo_ghoul']},
            {'name': '董香', 'series': 'tokyo_ghoul', 'tags': ['touka_kirishima', 'tokyo_ghoul']},
            
            # 更多游戏角色
            {'name': '雷电将军', 'series': 'genshin_impact', 'tags': ['raiden_shogun', 'genshin_impact']},
            {'name': '神里绫华', 'series': 'genshin_impact', 'tags': ['kamisato_ayaka', 'genshin_impact']},
            {'name': '荒泷一斗', 'series': 'genshin_impact', 'tags': ['arataki_itto', 'genshin_impact']},
            {'name': '八重神子', 'series': 'genshin_impact', 'tags': ['yae_miko', 'genshin_impact']},
            {'name': '琪亚娜·卡斯兰娜', 'series': 'honkai_impact_3', 'tags': ['kiana_kaslana', 'honkai_impact_3rd']},
            {'name': '雷电芽衣', 'series': 'honkai_impact_3', 'tags': ['raiden_mei', 'honkai_impact_3rd']}
        ]
    
    def _fetch_from_danbooru(self, tags, limit=50):
        """
        从Danbooru获取图像
        """
        base_url = 'https://danbooru.donmai.us/posts.json'
        images = []
        
        try:
            # 构建标签查询
            tag_string = ' '.join(tags)
            params = {
                'tags': tag_string,
                'limit': limit,
                'random': 'true'
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            for post in data:
                if post.get('file_url') and post.get('rating') == 's':
                    images.append({
                        'url': post['file_url'],
                        'width': post.get('width', 0),
                        'height': post.get('height', 0)
                    })
            
            logger.info(f"从Danbooru获取了 {len(images)} 张图像")
        except Exception as e:
            logger.error(f"从Danbooru获取图像失败: {e}")
        
        return images
    
    def _fetch_from_gelbooru(self, tags, limit=50):
        """
        从Gelbooru获取图像
        """
        base_url = 'https://gelbooru.com/index.php'
        images = []
        
        try:
            tag_string = ' '.join(tags)
            params = {
                'page': 'dapi',
                's': 'post',
                'q': 'index',
                'tags': tag_string,
                'limit': limit,
                'json': '1'
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            for post in data:
                if post.get('file_url') and post.get('rating') == 's':
                    images.append({
                        'url': post['file_url'],
                        'width': post.get('width', 0),
                        'height': post.get('height', 0)
                    })
            
            logger.info(f"从Gelbooru获取了 {len(images)} 张图像")
        except Exception as e:
            logger.error(f"从Gelbooru获取图像失败: {e}")
        
        return images
    
    def _fetch_from_safebooru(self, tags, limit=50):
        """
        从Safebooru获取图像
        """
        base_url = 'https://safebooru.org/index.php'
        images = []
        
        try:
            tag_string = ' '.join(tags)
            params = {
                'page': 'dapi',
                's': 'post',
                'q': 'index',
                'tags': tag_string,
                'limit': limit,
                'json': '1'
            }
            
            # 添加User-Agent和Referer
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'https://safebooru.org/'
            }
            
            response = requests.get(base_url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            
            # 检查响应内容
            if not response.text:
                logger.error("Safebooru返回空响应")
                return images
            
            data = response.json()
            
            # 检查数据格式
            if isinstance(data, dict) and 'posts' in data:
                data = data['posts']
            
            for post in data:
                if post.get('file_url'):
                    images.append({
                        'url': post['file_url'],
                        'width': post.get('width', 0),
                        'height': post.get('height', 0)
                    })
            
            logger.info(f"从Safebooru获取了 {len(images)} 张图像")
        except Exception as e:
            logger.error(f"从Safebooru获取图像失败: {e}")
            # 尝试使用备用API格式
            try:
                alt_params = {
                    'page': 'post',
                    's': 'list',
                    'tags': tag_string,
                    'limit': limit
                }
                alt_response = requests.get(base_url, params=alt_params, headers=headers, timeout=15)
                alt_response.raise_for_status()
                logger.info("尝试使用备用API格式获取Safebooru图像")
            except Exception as alt_e:
                logger.error(f"备用API格式也失败: {alt_e}")
        
        return images
    
    def _download_image(self, url, save_path):
        """
        下载图像
        """
        try:
            response = requests.get(url, timeout=15, stream=True)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type:
                logger.warning(f"不是图像文件: {url}")
                return False
            
            # 保存图像
            image = Image.open(BytesIO(response.content))
            
            # 处理不同图像模式
            if image.mode == 'RGBA':
                # 转换为RGB模式
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])  # 3是alpha通道
                image = background
            elif image.mode == 'P':
                # 转换为RGB模式
                image = image.convert('RGB')
            
            # 调整图像大小，确保合适的尺寸
            max_size = 1024
            if max(image.width, image.height) > max_size:
                ratio = max_size / max(image.width, image.height)
                new_width = int(image.width * ratio)
                new_height = int(image.height * ratio)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            image.save(save_path, 'JPEG', quality=95)
            
            # 验证图像
            with Image.open(save_path) as img:
                img.verify()
            
            return True
        except Exception as e:
            logger.warning(f"下载图像失败 {url}: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            return False
    
    def _process_character(self, character, sources=['danbooru', 'gelbooru', 'safebooru'], max_images=100):
        """
        处理单个角色
        """
        name = character['name']
        series = character['series']
        tags = character['tags']
        
        # 创建角色目录
        character_dir = os.path.join(self.output_dir, f"{series}_{name}")
        os.makedirs(character_dir, exist_ok=True)
        
        # 获取已有的图像数量
        existing_images = [f for f in os.listdir(character_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        existing_count = len(existing_images)
        
        if existing_count >= max_images:
            logger.info(f"{series}_{name} 已有 {existing_count} 张图像，达到上限")
            return existing_count
        
        # 需要下载的图像数量
        needed_count = max_images - existing_count
        logger.info(f"开始为 {series}_{name} 收集 {needed_count} 张图像")
        
        # 从各个数据源获取图像
        all_images = []
        for source_name in sources:
            if source_name in self.data_sources:
                logger.info(f"从 {source_name} 获取图像")
                source_images = self.data_sources[source_name](tags, limit=needed_count)
                all_images.extend(source_images)
                
                # 去重
                seen_urls = set()
                unique_images = []
                for img in all_images:
                    if img['url'] not in seen_urls:
                        seen_urls.add(img['url'])
                        unique_images.append(img)
                all_images = unique_images
                
                if len(all_images) >= needed_count:
                    break
        
        # 下载图像
        downloaded_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {}
            
            for i, img in enumerate(all_images[:needed_count]):
                save_path = os.path.join(character_dir, f"{series}_{name}_{existing_count + i:04d}.jpg")
                future = executor.submit(self._download_image, img['url'], save_path)
                future_to_url[future] = img['url']
            
            # 处理结果
            for future in tqdm(as_completed(future_to_url), total=len(future_to_url), 
                              desc=f"下载 {series}_{name} 图像"):
                if future.result():
                    downloaded_count += 1
        
        final_count = existing_count + downloaded_count
        logger.info(f"{series}_{name} 数据收集完成，当前共有 {final_count} 张图像")
        return final_count
    
    def expand_preset_characters(self, sources=['danbooru', 'gelbooru', 'safebooru'], max_images=100):
        """
        扩充预设角色数据
        """
        logger.info(f"开始扩充预设角色数据，共 {len(self.preset_characters)} 个角色")
        
        results = {}
        for character in self.preset_characters:
            count = self._process_character(character, sources, max_images)
            results[f"{character['series']}_{character['name']}"] = count
        
        logger.info("预设角色数据扩充完成")
        return results
    
    def add_custom_character(self, name, series, tags, sources=['danbooru', 'gelbooru', 'safebooru'], max_images=100):
        """
        添加自定义角色
        """
        character = {
            'name': name,
            'series': series,
            'tags': tags
        }
        
        count = self._process_character(character, sources, max_images)
        return count
    
    def get_dataset_statistics(self):
        """
        获取数据集统计信息
        """
        statistics = {
            'total_characters': 0,
            'total_images': 0,
            'characters': {}
        }
        
        # 遍历角色目录
        for character_dir in os.listdir(self.output_dir):
            character_path = os.path.join(self.output_dir, character_dir)
            if os.path.isdir(character_path):
                # 统计图像数量
                images = [f for f in os.listdir(character_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                image_count = len(images)
                
                if image_count > 0:
                    statistics['total_characters'] += 1
                    statistics['total_images'] += image_count
                    statistics['characters'][character_dir] = image_count
        
        return statistics
    
    def validate_dataset(self):
        """
        验证数据集
        """
        logger.info("开始验证数据集")
        
        invalid_count = 0
        total_count = 0
        
        for character_dir in os.listdir(self.output_dir):
            character_path = os.path.join(self.output_dir, character_dir)
            if os.path.isdir(character_path):
                for img_name in os.listdir(character_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        total_count += 1
                        img_path = os.path.join(character_path, img_name)
                        
                        try:
                            with Image.open(img_path) as img:
                                img.verify()
                        except Exception as e:
                            logger.warning(f"无效图像: {img_path}, 错误: {e}")
                            os.remove(img_path)
                            invalid_count += 1
        
        logger.info(f"数据集验证完成，检查了 {total_count} 张图像，删除了 {invalid_count} 张无效图像")
        return invalid_count

def main():
    parser = argparse.ArgumentParser(description='自动化数据集扩充脚本')
    
    parser.add_argument('--output-dir', type=str, 
                       default='data/train',
                       help='输出目录')
    parser.add_argument('--max-workers', type=int, default=10,
                       help='最大并发数')
    parser.add_argument('--max-images', type=int, default=100,
                       help='每个角色的最大图像数')
    parser.add_argument('--sources', type=str, nargs='+',
                       default=['danbooru', 'gelbooru', 'safebooru'],
                       help='数据源')
    parser.add_argument('--add-character', type=str, nargs=3,
                       metavar=('NAME', 'SERIES', 'TAGS'),
                       help='添加自定义角色')
    parser.add_argument('--validate', action='store_true',
                       help='验证数据集')
    parser.add_argument('--stats', action='store_true',
                       help='获取数据集统计信息')
    
    args = parser.parse_args()
    
    # 初始化扩充器
    expander = AutomatedDataExpander(
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # 验证数据集
    if args.validate:
        expander.validate_dataset()
    
    # 获取统计信息
    if args.stats:
        stats = expander.get_dataset_statistics()
        print("\n" + "="*60)
        print("数据集统计信息")
        print("="*60)
        print(f"总角色数: {stats['total_characters']}")
        print(f"总图像数: {stats['total_images']}")
        print("\n各角色图像数:")
        for character, count in sorted(stats['characters'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {character}: {count}")
        print("="*60)
    
    # 添加自定义角色
    if args.add_character:
        name, series, tags_str = args.add_character
        tags = tags_str.split(',')
        count = expander.add_custom_character(name, series, tags, args.sources, args.max_images)
        print(f"\n自定义角色 {series}_{name} 添加完成，共 {count} 张图像")
    
    # 扩充预设角色
    else:
        results = expander.expand_preset_characters(args.sources, args.max_images)
        
        # 打印结果
        print("\n" + "="*60)
        print("数据集扩充结果")
        print("="*60)
        for character, count in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {character}: {count}")
        print("="*60)
        
        # 验证数据集
        expander.validate_dataset()

if __name__ == '__main__':
    main()

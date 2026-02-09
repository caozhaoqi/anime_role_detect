#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的数据集扩展脚本

基于当前项目需求，扩展数据集并增加数据多样性
"""

import os
import argparse
import requests
import json
import logging
import time
import random
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
logger = logging.getLogger('data_expander')


class ImprovedDataExpander:
    def __init__(self, output_dir='data/train', max_workers=5):
        """
        初始化改进的数据扩展器
        
        Args:
            output_dir: 输出目录
            max_workers: 最大并发数（降低以避免被封禁）
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        # 预设角色列表（基于当前项目需求）
        self.preset_characters = [
            # 原神角色
            {'name': '温迪', 'series': 'genshin_impact', 'tags': ['venti', 'genshin_impact']},
            {'name': '迪卢克', 'series': 'genshin_impact', 'tags': ['diluc', 'genshin_impact']},
            {'name': '刻晴', 'series': 'genshin_impact', 'tags': ['keqing', 'genshin_impact']},
            {'name': '甘雨', 'series': 'genshin_impact', 'tags': ['ganyu', 'genshin_impact']},
            {'name': '胡桃', 'series': 'genshin_impact', 'tags': ['hu_tao', 'genshin_impact']},
            {'name': '雷电将军', 'series': 'genshin_impact', 'tags': ['raiden_shogun', 'genshin_impact']},
            {'name': '神里绫华', 'series': 'genshin_impact', 'tags': ['kamisato_ayaka', 'genshin_impact']},
            {'name': '荒泷一斗', 'series': 'genshin_impact', 'tags': ['arataki_itto', 'genshin_impact']},
            {'name': '八重神子', 'series': 'genshin_impact', 'tags': ['yae_miko', 'genshin_impact']},
            
            # 崩坏：星穹铁道角色
            {'name': '三月七', 'series': 'honkai_star_rail', 'tags': ['march_7th', 'honkai_star_rail']},
            {'name': '丹恒', 'series': 'honkai_star_rail', 'tags': ['dan_heng', 'honkai_star_rail']},
            {'name': '姬子', 'series': 'honkai_star_rail', 'tags': ['himeko', 'honkai_star_rail']},
            {'name': '瓦尔特', 'series': 'honkai_star_rail', 'tags': ['welt', 'honkai_star_rail']},
            
            # 崩坏3角色
            {'name': '琪亚娜', 'series': 'honkai_impact_3', 'tags': ['kiana_kaslana', 'honkai_impact_3rd']},
            {'name': '芽衣', 'series': 'honkai_impact_3', 'tags': ['raiden_mei', 'honkai_impact_3rd']},
            {'name': '布洛妮娅', 'series': 'honkai_impact_3', 'tags': ['bronya_zaychik', 'honkai_impact_3rd']},
            {'name': '德丽莎', 'series': 'honkai_impact_3', 'tags': ['theresa_apocalypse', 'honkai_impact_3rd']},
            
            # 火影忍者角色
            {'name': '鸣人', 'series': 'naruto', 'tags': ['uzumaki_naruto', 'naruto']},
            {'name': '佐助', 'series': 'naruto', 'tags': ['uchiha_sasuke', 'naruto']},
            
            # 海贼王角色
            {'name': '路飞', 'series': 'one_piece', 'tags': ['monkey_d._luffy', 'one_piece']},
            {'name': '索隆', 'series': 'one_piece', 'tags': ['roronoa_zoro', 'one_piece']},
            
            # 龙珠角色
            {'name': '孙悟空', 'series': 'dragon_ball', 'tags': ['son_goku', 'dragon_ball']},
            {'name': '贝吉塔', 'series': 'dragon_ball', 'tags': ['vegeta', 'dragon_ball']},
            
            # 进击的巨人角色
            {'name': '艾伦', 'series': 'attack_on_titan', 'tags': ['eren_yeager', 'attack_on_titan']},
            {'name': '三笠', 'series': 'attack_on_titan', 'tags': ['mikasa_ackerman', 'attack_on_titan']},
            
            # 鬼灭之刃角色
            {'name': '炭治郎', 'series': 'demon_slayer', 'tags': ['tanjiro_kamado', 'demon_slayer']},
            {'name': '祢豆子', 'series': 'demon_slayer', 'tags': ['nezuko_kamado', 'demon_slayer']},
            
            # 东京食尸鬼角色
            {'name': '金木研', 'series': 'tokyo_ghoul', 'tags': ['ken_kaneki', 'tokyo_ghoul']},
            {'name': '董香', 'series': 'tokyo_ghoul', 'tags': ['touka_kirishima', 'tokyo_ghoul']}
        ]
    
    def _fetch_from_safebooru(self, tags, limit=50):
        """
        从Safebooru获取图像（使用更稳定的方法）
        """
        base_url = 'https://safebooru.org/index.php'
        images = []
        
        try:
            # 构建标签查询
            tag_string = ' '.join(tags)
            
            # 使用不同的参数组合
            params = {
                'page': 'dapi',
                's': 'post',
                'q': 'index',
                'tags': tag_string,
                'limit': limit,
                'json': '1'
            }
            
            # 添加延迟以避免被封禁
            time.sleep(random.uniform(1.0, 2.0))
            
            response = requests.get(base_url, params=params, headers=self.headers, timeout=20)
            response.raise_for_status()
            
            # 检查响应内容
            if not response.text:
                logger.warning(f"Safebooru返回空响应，标签: {tag_string}")
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
            
            logger.info(f"从Safebooru获取了 {len(images)} 张图像，标签: {tag_string}")
        except Exception as e:
            logger.error(f"从Safebooru获取图像失败，标签: {tag_string}, 错误: {e}")
        
        return images
    
    def _fetch_from_waifu_pics(self, tags, limit=50):
        """
        从Waifu.pics获取图像
        """
        base_url = 'https://api.waifu.pics/sfw/waifu'
        images = []
        
        try:
            # Waifu.pics API比较简单，每次返回一张随机图像
            for _ in range(limit):
                time.sleep(random.uniform(0.5, 1.0))
                response = requests.get(base_url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if data.get('url'):
                    images.append({
                        'url': data['url'],
                        'width': 0,
                        'height': 0
                    })
            
            logger.info(f"从Waifu.pics获取了 {len(images)} 张图像")
        except Exception as e:
            logger.error(f"从Waifu.pics获取图像失败: {e}")
        
        return images
    
    def _download_image(self, url, save_path):
        """
        下载图像
        """
        try:
            time.sleep(random.uniform(0.3, 0.8))
            response = requests.get(url, headers=self.headers, timeout=15, stream=True)
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
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode == 'P':
                image = image.convert('RGB')
            
            # 调整图像大小
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
    
    def _process_character(self, character, max_images=100):
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
        
        # 优先使用Safebooru
        logger.info(f"从 Safebooru 获取图像")
        source_images = self._fetch_from_safebooru(tags, limit=needed_count)
        all_images.extend(source_images)
        
        # 如果还不够，尝试其他数据源
        if len(all_images) < needed_count:
            logger.info(f"从 Waifu.pics 获取补充图像")
            additional_images = self._fetch_from_waifu_pics(tags, limit=needed_count - len(all_images))
            all_images.extend(additional_images)
        
        # 去重
        seen_urls = set()
        unique_images = []
        for img in all_images:
            if img['url'] not in seen_urls:
                seen_urls.add(img['url'])
                unique_images.append(img)
        all_images = unique_images
        
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
    
    def expand_dataset(self, max_images=100):
        """
        扩展数据集
        """
        logger.info(f"开始扩展数据集，共 {len(self.preset_characters)} 个角色")
        
        results = {}
        for i, character in enumerate(self.preset_characters):
            logger.info(f"处理角色 {i+1}/{len(self.preset_characters)}")
            count = self._process_character(character, max_images)
            results[f"{character['series']}_{character['name']}"] = count
            
            # 角色之间增加延迟
            if i < len(self.preset_characters) - 1:
                time.sleep(random.uniform(2.0, 4.0))
        
        logger.info("数据集扩展完成")
        return results
    
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
    parser = argparse.ArgumentParser(description='改进的数据集扩展脚本')
    
    parser.add_argument('--output-dir', type=str, 
                       default='data/train',
                       help='输出目录')
    parser.add_argument('--max-workers', type=int, default=5,
                       help='最大并发数')
    parser.add_argument('--max-images', type=int, default=100,
                       help='每个角色的最大图像数')
    parser.add_argument('--validate', action='store_true',
                       help='验证数据集')
    parser.add_argument('--stats', action='store_true',
                       help='获取数据集统计信息')
    
    args = parser.parse_args()
    
    # 初始化扩展器
    expander = ImprovedDataExpander(
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
    
    # 扩展数据集
    else:
        results = expander.expand_dataset(args.max_images)
        
        # 打印结果
        print("\n" + "="*60)
        print("数据集扩展结果")
        print("="*60)
        for character, count in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {character}: {count}")
        print("="*60)
        
        # 验证数据集
        expander.validate_dataset()
        
        # 显示最终统计信息
        stats = expander.get_dataset_statistics()
        print(f"\n最终统计: {stats['total_characters']} 个角色，{stats['total_images']} 张图像")


if __name__ == '__main__':
    main()

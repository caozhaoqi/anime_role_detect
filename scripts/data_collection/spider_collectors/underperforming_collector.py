#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""未达标角色补全脚本 - 专门处理资源匮乏的角色"""

import os
import sys
import time
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from loguru import logger
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import hashlib

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 未达标角色列表
UNDERPERFORMING_CHARACTERS = [
    {"name": "popukar", "tag": "popukar", "work": "arknights", "current": 9, "need": 91},
    {"name": "dori", "tag": "dori", "work": "genshin_impact", "current": 22, "need": 78},
    {"name": "sigewinne", "tag": "sigewinne", "work": "genshin_impact", "current": 22, "need": 78},
    {"name": "rosmontis", "tag": "rosmontis", "work": "arknights", "current": 24, "need": 76},
    {"name": "clara", "tag": "clara", "work": "honkai_star_rail", "current": 35, "need": 65},
    {"name": "collei", "tag": "collei", "work": "genshin_impact", "current": 53, "need": 47},
    {"name": "yunli", "tag": "yunli", "work": "honkai_star_rail", "current": 54, "need": 46},
    {"name": "yaoyao", "tag": "yaoyao", "work": "genshin_impact", "current": 60, "need": 40},
    {"name": "diona", "tag": "diona", "work": "genshin_impact", "current": 73, "need": 27},
    {"name": "ceobe", "tag": "ceobe", "work": "arknights", "current": 77, "need": 23},
    {"name": "sparkle", "tag": "sparkle", "work": "honkai_star_rail", "current": 92, "need": 8},
]

# 作品名映射
WORK_MAPPING = {
    '蔚蓝档案': 'blue_archive',
    '原神': 'genshin_impact',
    '崩坏星穹铁道': 'honkai_star_rail',
    '明日方舟': 'arknights',
    '崩坏3': 'honkai_impact_3rd',
    '碧蓝航线': 'azur_lane',
    '战舰少女': 'kancolle',
    '阴阳师': 'onmyoji',
    '绝区零': 'zenless_zone_zero',
    '鸣潮': 'wuthering_waves',
    '卡拉彼丘': 'lycoris',
}


class UnderperformingCollector:
    """未达标角色补全采集器"""
    
    def __init__(self, output_dir: str, target_count: int = 100, workers: int = 8, delay: float = 2.0):
        self.output_dir = Path(output_dir)
        self.target_count = target_count
        self.max_workers = workers
        self.delay = delay
        self.session = self._create_session()
        self._seen_hashes: Set[str] = set()
        self._lock = threading.Lock()
        
        # 站点配置
        self.SITES = {
            'yande.re': {
                'name': 'Yande.re',
                'api_url': 'https://yande.re/post.json',
                'format': 'json',
            },
            'safebooru': {
                'name': 'Safebooru',
                'api_url': 'https://safebooru.org/index.php?page=dapi&s=post&q=index',
                'format': 'xml',
            },
            'anime-pictures': {
                'name': 'Anime-Pictures',
                'api_url': 'https://anime-pictures.net/api/posts',
                'format': 'json',
            },
        }
        
        logger.info(f"未达标角色补全器初始化完成，目标: {target_count}张/角色")
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json, text/html',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        return session
    
    def _prepare_search_tags(self, tag: str) -> List[str]:
        """生成多种搜索标签策略"""
        tags_list = []
        
        # 如果标签包含作品名
        if '(' in tag and ')' in tag:
            idx = tag.rfind('(')
            char_name = tag[:idx].strip('_')
            work_name = tag[idx+1:-1]
            mapped_work = WORK_MAPPING.get(work_name, work_name)
            
            tags_list = [
                char_name,
                f"{char_name} {mapped_work}",
                char_name.replace('_', ' '),
                f"{char_name} {work_name}",
            ]
        else:
            tags_list = [
                tag,
                tag.replace('_', ' '),
                tag.replace('_', ''),
            ]
        
        return list(set(tags_list))  # 去重
    
    def _scan_existing_images(self, char_dir: Path) -> Set[str]:
        """扫描已存在的图片"""
        existing_ids = set()
        
        if not char_dir.exists():
            return existing_ids
        
        for img_file in char_dir.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                stem = img_file.stem
                parts = stem.split('_')
                if len(parts) >= 2 and parts[0] == 'id':
                    post_id = parts[1]
                    existing_ids.add(f"id_{post_id}")
        
        return existing_ids
    
    def _is_female_character(self, post: Dict) -> bool:
        """检查是否为女角色"""
        tags_str = ""
        if 'tags' in post and post['tags']:
            tags_str = str(post['tags']).lower()
        if 'tag_string' in post and post['tag_string']:
            tags_str = str(post['tag_string']).lower()
        
        female_indicators = ['female', '1girl', 'solo_female', 'girl', 'female_only']
        male_indicators = ['male', '1boy', 'solo_male', 'boy', 'male_only', 'gender_swap']
        
        for indicator in male_indicators:
            if indicator in tags_str:
                return False
        
        for indicator in female_indicators:
            if indicator in tags_str:
                return True
        
        return True
    
    def _get_unique_key(self, post: Dict) -> Optional[str]:
        if 'md5' in post and post['md5']:
            return f"md5_{post['md5']}"
        if 'id' in post and post['id']:
            return f"id_{post['id']}"
        if 'file_url' in post and post['file_url']:
            return hashlib.md5(post['file_url'].encode()).hexdigest()
        return None
    
    def search_all(self, tags: str, max_posts: int = 100) -> List[Dict]:
        """从所有站点搜索"""
        all_posts = []
        
        for site_id in self.SITES:
            if len(all_posts) >= max_posts:
                break
            
            try:
                posts = self._search_site(site_id, tags, max_posts - len(all_posts))
                
                for post in posts:
                    if not self._is_female_character(post):
                        continue
                    
                    unique_key = self._get_unique_key(post)
                    if unique_key and unique_key not in self._seen_hashes:
                        self._seen_hashes.add(unique_key)
                        all_posts.append(post)
                
                if posts:
                    logger.info(f"[{self.SITES[site_id]['name']}] 找到 {len(posts)} 张")
                    
            except Exception as e:
                logger.warning(f"[{self.SITES[site_id]['name']}] 搜索失败: {e}")
            
            time.sleep(random.uniform(0.5, 1.5))
        
        return all_posts[:max_posts]
    
    def _search_site(self, site_id: str, tags: str, max_posts: int) -> List[Dict]:
        site = self.SITES[site_id]
        
        if site_id == 'yande.re':
            return self._search_yandere(tags, max_posts)
        elif site_id == 'safebooru':
            return self._search_safebooru(tags, max_posts)
        elif site_id == 'anime-pictures':
            return self._search_anime_pictures(tags, max_posts)
        
        return []
    
    def _search_yandere(self, tags: str, max_posts: int) -> List[Dict]:
        """搜索 Yande.re"""
        all_posts = []
        page = 1
        
        while len(all_posts) < max_posts:
            params = {'tags': tags, 'page': page, 'limit': min(100, max_posts)}
            
            try:
                response = self.session.get(
                    'https://yande.re/post.json',
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                posts = response.json()
                
                if not posts:
                    break
                
                for post in posts:
                    post['source_site'] = 'yande'
                    all_posts.append(post)
                
                page += 1
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                logger.warning(f"[Yande.re] 搜索失败: {e}")
                break
        
        return all_posts
    
    def _search_safebooru(self, tags: str, max_posts: int) -> List[Dict]:
        """搜索 Safebooru"""
        all_posts = []
        page = 0
        
        while len(all_posts) < max_posts:
            params = {'tags': tags, 'page': page, 'limit': min(100, max_posts)}
            
            try:
                response = self.session.get(
                    'https://safebooru.org/index.php?page=dapi&s=post&q=index',
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(response.content)
                except ET.ParseError as e:
                    logger.warning(f"[Safebooru] XML解析错误: {e}")
                    break
                
                posts = []
                for post in root.findall('.//post'):
                    try:
                        p = {
                            'id': int(post.get('id', 0) or 0),
                            'md5': post.get('hash', ''),
                            'source_site': 'safebooru',
                            'file_url': post.get('file_url', ''),
                            'preview_url': post.get('preview_url', ''),
                            'tags': post.get('tags', ''),
                            'width': int(post.get('width', 0) or 0),
                            'height': int(post.get('height', 0) or 0),
                            'score': int(post.get('score', 0) or 0),
                        }
                        posts.append(p)
                    except (ValueError, TypeError):
                        continue
                
                if not posts:
                    break
                
                all_posts.extend(posts)
                page += 1
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                logger.warning(f"[Safebooru] 搜索失败: {e}")
                break
        
        return all_posts
    
    def _search_anime_pictures(self, tags: str, max_posts: int) -> List[Dict]:
        """搜索 Anime-Pictures.net
        
        注意: Anime-Pictures API需要注册获取API密钥
        这里使用网页搜索作为备选方案
        """
        try:
            # 尝试使用公开的搜索API
            response = self.session.get(
                'https://anime-pictures.net/api/posts',
                params={
                    'search_text': tags,
                    'lang': 'en',
                    'order_by': 'likes',
                    'page': 1,
                    'limit': max_posts
                },
                headers={'Accept': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    posts = []
                    for item in data.get('posts', [])[:max_posts]:
                        posts.append({
                            'id': item.get('id'),
                            'md5': item.get('md5'),
                            'source_site': 'anime-pictures',
                            'file_url': item.get('file_url'),
                            'preview_url': item.get('preview_url'),
                            'tags': item.get('tags', ''),
                        })
                    return posts
                except (ValueError, KeyError) as e:
                    logger.warning(f"[Anime-Pictures] JSON解析失败: {e}")
            else:
                logger.warning(f"[Anime-Pictures] API返回状态码: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"[Anime-Pictures] 搜索失败: {e}")
        
        return []
    
    def download_image(self, post: Dict, save_dir: Path) -> Tuple[bool, str]:
        """下载单张图片"""
        file_url = post.get('file_url', '')
        if not file_url:
            return False, "No URL"
        
        post_id = str(post.get('id', ''))
        md5 = post.get('md5', '')[:8]
        ext = 'jpg'
        
        if '.png' in file_url:
            ext = 'png'
        elif '.gif' in file_url:
            ext = 'gif'
        elif '.webp' in file_url:
            ext = 'webp'
        
        filename = f"id_{post['source_site']}_{post_id}_{md5}.{ext}"
        filepath = save_dir / filename
        
        if filepath.exists():
            return True, "Exists"
        
        try:
            response = self.session.get(file_url, timeout=60, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True, "Success"
            
        except Exception as e:
            if filepath.exists():
                filepath.unlink()
            return False, str(e)[:30]
    
    def collect_character(self, char_info: Dict) -> Tuple[int, int, int]:
        """采集单个角色"""
        tag = char_info['tag']
        work = char_info['work']
        current = char_info['current']
        need = char_info['need']
        
        full_tag = f"{tag}_({work})"
        character_dir = self.output_dir / full_tag
        
        # 扫描已存在图片
        existing_ids = self._scan_existing_images(character_dir)
        
        if character_dir.exists():
            actual_count = len(list(character_dir.glob("*.jpg"))) + len(list(character_dir.glob("*.png")))
        else:
            actual_count = 0
            character_dir.mkdir(parents=True, exist_ok=True)
        
        if actual_count >= self.target_count:
            logger.info(f"[{full_tag}] 已达标 ({actual_count}张)")
            return (0, 0, actual_count)
        
        need_count = self.target_count - actual_count
        logger.info(f"[{full_tag}] 当前{actual_count}张，需补充{need_count}张")
        
        # 生成多种搜索策略
        search_tags = self._prepare_search_tags(full_tag)
        
        # 搜索
        all_posts = []
        for st in search_tags:
            if len(all_posts) >= need_count:
                break
            
            posts = self.search_all(st, need_count - len(all_posts))
            
            # 过滤已存在
            for post in posts:
                post_id = str(post.get('id', ''))
                if post_id and f"id_{post_id}" in existing_ids:
                    continue
                all_posts.append(post)
            
            if posts:
                logger.info(f"[{full_tag}] 标签'{st}'找到 {len(posts)} 张")
        
        logger.info(f"[{full_tag}] 共找到 {len(all_posts)} 张图片")
        
        if not all_posts:
            return (0, 0, actual_count)
        
        # 下载
        success = 0
        fail = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_image, post, character_dir): post
                for post in all_posts
            }
            
            for future in as_completed(futures):
                ok, _ = future.result()
                if ok:
                    success += 1
                else:
                    fail += 1
        
        final_count = actual_count + success
        logger.success(f"[{full_tag}] 完成: +{success}张 (总计{final_count}张)")
        
        return (success, fail, final_count)
    
    def run(self):
        """运行采集"""
        total_chars = len(UNDERPERFORMING_CHARACTERS)
        
        logger.info(f"=" * 50)
        logger.info(f"开始补全 {total_chars} 个未达标角色")
        logger.info(f"=" * 50)
        
        total_success = 0
        total_fail = 0
        
        for i, char in enumerate(UNDERPERFORMING_CHARACTERS, 1):
            logger.info(f"[{i}/{total_chars}] 正在补全: {char['name']} ({char['work']})")
            
            success, fail, final = self.collect_character(char)
            total_success += success
            total_fail += fail
            
            # 发送进度通知
            logger.info(f"[{i}/{total_chars}] {char['name']} 补全完成: +{success}张")
            
            # 延迟
            if i < total_chars:
                time.sleep(random.uniform(2, 4))
        
        logger.success(f"=" * 50)
        logger.success(f"补全完成! 总计: +{total_success}张, 失败: {total_fail}张")
        logger.success(f"=" * 50)
        
        return total_success, total_fail


def main():
    import argparse
    parser = argparse.ArgumentParser(description='未达标角色补全采集器')
    parser.add_argument('--output-dir', type=str, default='data/danbooru_images', help='输出目录')
    parser.add_argument('--target-count', type=int, default=100, help='目标图片数')
    parser.add_argument('--workers', type=int, default=8, help='并发线程数')
    parser.add_argument('--delay', type=float, default=2.0, help='请求延迟')
    args = parser.parse_args()
    
    collector = UnderperformingCollector(
        output_dir=args.output_dir,
        target_count=args.target_count,
        workers=args.workers,
        delay=args.delay
    )
    
    collector.run()


if __name__ == '__main__':
    main()
